"""UnifiedHealthManager - Consolidated error recovery and health management (December 2025).

This module consolidates ErrorRecoveryCoordinator and RecoveryManager into a single
unified health management system. It provides:

1. Error tracking and statistics (from ErrorRecoveryCoordinator)
2. Circuit breaker management (from ErrorRecoveryCoordinator)
3. Recovery operations (from RecoveryManager)
4. Node/job health tracking (consolidated from both)

Event Integration:
- Subscribes to ERROR: Track error occurrences
- Subscribes to RECOVERY_INITIATED: Track recovery start
- Subscribes to RECOVERY_COMPLETED: Track successful recoveries
- Subscribes to RECOVERY_FAILED: Track failed recovery attempts
- Subscribes to TRAINING_FAILED: Track training failures
- Subscribes to TASK_FAILED: Track task failures
- Subscribes to REGRESSION_DETECTED: Track model regressions
- Subscribes to HOST_OFFLINE: Track offline hosts for recovery
- Subscribes to HOST_ONLINE: Track hosts coming online (Dec 2025)
- Subscribes to NODE_RECOVERED: Update recovery state
- Subscribes to PARITY_FAILURE_RATE_CHANGED: Alert on TS/Python parity issues (Dec 2025)
- Subscribes to COORDINATOR_HEALTH_DEGRADED: Track coordinator health issues (Dec 2025)
- Subscribes to DAEMON_STARTED: Track daemon lifecycle for health visibility (Dec 2025)
- Subscribes to DAEMON_STOPPED: Track daemon stops and detect unexpected failures (Dec 2025)

Usage:
    from app.coordination.unified_health_manager import (
        UnifiedHealthManager,
        get_health_manager,
        wire_health_events,
    )

    # Wire health events
    manager = wire_health_events()

    # Check circuit breaker
    if manager.is_circuit_broken("training"):
        print("Training circuit breaker open")

    # Recover stuck job
    result = await manager.recover_stuck_job(work_item, expected_timeout=300)

    # Get unified health stats
    stats = manager.get_health_stats()
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.coordination.coordinator_base import CoordinatorBase, CoordinatorStatus
from app.distributed.circuit_breaker import CircuitBreaker, CircuitState

# Event emission for node health feedback loops (Phase 21.2 - Dec 2025)
try:
    from app.coordination.event_router import emit_node_overloaded
    HAS_NODE_EVENTS = True
except ImportError:
    emit_node_overloaded = None
    HAS_NODE_EVENTS = False

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkItem

logger = logging.getLogger(__name__)


# =============================================================================
# Enums (consolidated from both modules)
# =============================================================================

# December 2025: Import ErrorSeverity from canonical source
from app.coordination.types import ErrorSeverity

# ErrorSeverity is now imported from app.coordination.types
# Canonical values: INFO, WARNING, ERROR, CRITICAL


class SystemHealthLevel(Enum):
    """System health levels for aggregate scoring."""

    HEALTHY = "healthy"  # 80-100
    DEGRADED = "degraded"  # 60-79
    UNHEALTHY = "unhealthy"  # 40-59
    CRITICAL = "critical"  # 0-39


class PipelineState(Enum):
    """Pipeline operational state."""

    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"


class RecoveryStatus(Enum):
    """Recovery attempt status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecoveryAction(str, Enum):
    """Types of job-level recovery actions.

    NOTE (Dec 2025): Renamed from RecoveryAction to avoid collision with
    SystemRecoveryAction in recovery_orchestrator.py and NodeRecoveryAction
    in node_recovery_daemon.py which have different semantics.
    """

    RESTART_JOB = "restart_job"
    KILL_JOB = "kill_job"
    RESTART_NODE_SERVICES = "restart_node_services"
    REBOOT_NODE = "reboot_node"
    REMOVE_NODE = "remove_node"
    ESCALATE_HUMAN = "escalate_human"
    NONE = "none"


# Backward-compat alias (deprecated)
RecoveryAction = JobRecoveryAction


class RecoveryResult(str, Enum):
    """Result of a recovery attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"


# =============================================================================
# Data Classes (consolidated from both modules)
# =============================================================================


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_id: str
    component: str
    error_type: str
    message: str
    node_id: str = ""
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    recovery_time: float = 0.0


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    recovery_id: str
    error_id: str
    component: str
    node_id: str
    strategy: str
    status: RecoveryStatus = RecoveryStatus.PENDING
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    success: bool = False
    message: str = ""
    attempt_number: int = 1

    @property
    def duration(self) -> float:
        """Get recovery duration in seconds."""
        if self.completed_at > 0:
            return self.completed_at - self.started_at
        return time.time() - self.started_at


@dataclass
class RecoveryEvent:
    """Record of a recovery action."""

    timestamp: float
    action: RecoveryAction
    target_type: str  # "job" or "node"
    target_id: str
    result: RecoveryResult
    reason: str
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class NodeRecoveryState:
    """Track recovery state for a node.

    December 2025: Renamed from NodeHealthState to avoid collision with
    the NodeHealthState enum in node_status.py which represents health grades
    (HEALTHY, DEGRADED, UNHEALTHY, etc.). This dataclass tracks recovery
    attempts and failure counts, not health grades.

    December 2025 Wave 2: Extended with additional fields for coordinator
    lifecycle tracking (heartbeats, responsiveness).
    """

    node_id: str
    is_online: bool = True
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0
    consecutive_failures: int = 0
    is_escalated: bool = False
    last_escalation_time: float = 0.0
    offline_since: float = 0.0
    # December 2025 Wave 2: Coordinator lifecycle tracking
    last_heartbeat: float = 0.0
    last_health_update: float = 0.0

    @property
    def is_healthy(self) -> bool:
        """Alias for is_online (backward compat)."""
        return self.is_online

    @is_healthy.setter
    def is_healthy(self, value: bool) -> None:
        """Set is_online via is_healthy alias."""
        self.is_online = value

    @property
    def is_responsive(self) -> bool:
        """Node is responsive if online and recently sent heartbeat."""
        if not self.is_online:
            return False
        if self.last_heartbeat == 0.0:
            return True  # No heartbeat tracking yet, assume responsive
        return (time.time() - self.last_heartbeat) < 120.0  # 2 min threshold

    @is_responsive.setter
    def is_responsive(self, value: bool) -> None:
        """Setting responsive to False marks node offline."""
        if not value:
            self.is_online = False

    @property
    def failure_count(self) -> int:
        """Alias for consecutive_failures (backward compat)."""
        return self.consecutive_failures

    @failure_count.setter
    def failure_count(self, value: int) -> None:
        """Set consecutive_failures via failure_count alias."""
        self.consecutive_failures = value


# Backward-compat alias (deprecated - use NodeRecoveryState)
NodeHealthState = NodeRecoveryState


@dataclass
class JobHealthState:
    """Track health state for a job."""

    work_id: str
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0


@dataclass
class DaemonHealthState:
    """Track health state for a daemon (December 2025).

    Used by UnifiedHealthManager to track daemon lifecycle events
    and provide visibility into daemon health across the cluster.
    """

    daemon_name: str
    hostname: str = ""
    started_at: float = 0.0
    stopped_at: float = 0.0
    is_running: bool = False
    restart_count: int = 0
    last_stop_reason: str = ""
    consecutive_failures: int = 0
    last_error: str | None = None


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    # Stuck job detection
    stuck_job_timeout_multiplier: float = 1.5

    # Recovery attempt limits
    max_recovery_attempts_per_node: int = 3
    max_recovery_attempts_per_job: int = 2
    recovery_attempt_cooldown: int = 300  # 5 min (see TIMEOUTS.RECOVERY_COOLDOWN)

    # Escalation thresholds
    consecutive_failures_for_escalation: int = 3
    escalation_cooldown: int = 3600  # 1 hour (see TIMEOUTS.ESCALATION_COOLDOWN)

    # Node health thresholds
    node_unhealthy_after_failures: int = 3
    node_recovery_timeout: int = 120

    # Circuit breaker config
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_half_open_timeout: float = 30.0

    # History limits
    max_error_history: int = 500
    max_recovery_history: int = 200

    # Enabled flag
    enabled: bool = True


@dataclass
class SystemHealthConfig:
    """Configuration for system health monitoring (from system_health_monitor.py)."""

    # Check interval
    check_interval_seconds: int = 30

    # Health score thresholds
    healthy_threshold: int = 80
    degraded_threshold: int = 60
    unhealthy_threshold: int = 40

    # Pause triggers
    pause_health_threshold: int = 40
    pause_node_offline_percent: float = 0.5  # 50%
    pause_error_burst_count: int = 10
    pause_error_burst_window: int = 300  # 5 min (see TIMEOUTS.PAUSE_ERROR_BURST_WINDOW)

    # Critical circuits that trigger immediate pause if broken
    critical_circuits: list[str] = field(
        default_factory=lambda: ["training", "evaluation", "promotion"]
    )

    # Resume thresholds (hysteresis)
    resume_health_threshold: int = 60
    resume_delay_seconds: int = 120  # 2 min (see TIMEOUTS.RESUME_DELAY_SECONDS)

    # Expected nodes (0 = auto-discover)
    expected_nodes: int = 0

    # Component weights for score calculation
    node_weight: float = 0.40
    circuit_weight: float = 0.25
    error_weight: float = 0.20
    recovery_weight: float = 0.15


@dataclass
class SystemHealthScore:
    """Aggregate system health score (from system_health_monitor.py)."""

    score: int  # 0-100
    level: SystemHealthLevel
    components: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Component scores (0-100 each)
    node_availability: float = 100.0
    circuit_health: float = 100.0
    error_rate: float = 100.0  # Inverted: 100 = no errors
    recovery_success: float = 100.0

    # Pause triggers
    pause_triggers: list[str] = field(default_factory=list)


@dataclass
class HealthStats:
    """Aggregate health statistics (consolidated)."""

    # Error stats
    total_errors: int = 0
    errors_by_severity: dict[str, int] = field(default_factory=dict)
    errors_by_component: dict[str, int] = field(default_factory=dict)
    errors_by_node: dict[str, int] = field(default_factory=dict)

    # Recovery stats
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_rate: float = 0.0
    avg_recovery_time: float = 0.0

    # Circuit breaker stats
    circuit_breakers_open: int = 0
    open_circuits: list[str] = field(default_factory=list)

    # Node health stats
    nodes_tracked: int = 0
    nodes_online: int = 0
    nodes_offline: int = 0
    escalated_nodes: list[str] = field(default_factory=list)

    # Job health stats
    jobs_tracked: int = 0


# =============================================================================
# UnifiedHealthManager
# =============================================================================


class UnifiedHealthManager(CoordinatorBase):
    """Unified health management combining error tracking and recovery operations.

    This consolidates ErrorRecoveryCoordinator and RecoveryManager into a single
    cohesive system for health monitoring and recovery.

    Key Responsibilities:
    1. Track errors and failures across the cluster
    2. Manage circuit breakers for failing components
    3. Coordinate recovery operations (job kills, service restarts)
    4. Track node and job health states
    5. Handle escalation to human operators
    """

    def __init__(
        self,
        config: RecoveryConfig | None = None,
        notifier: Any | None = None,
    ):
        """Initialize UnifiedHealthManager.

        Args:
            config: Recovery configuration
            notifier: Optional notification service for escalations
        """
        super().__init__(name="UnifiedHealthManager")
        self.config = config or RecoveryConfig()

        # Error tracking
        self._errors: list[ErrorRecord] = []
        self._errors_by_component: dict[str, list[ErrorRecord]] = defaultdict(list)
        self._error_id_counter = 0

        # Recovery tracking
        self._active_recoveries: dict[str, RecoveryAttempt] = {}
        self._recovery_history: list[RecoveryAttempt] = []
        self._recovery_events: list[RecoveryEvent] = []
        self._recovery_id_counter = 0

        # Node and job health tracking (consolidated)
        self._node_states: dict[str, NodeHealthState] = {}
        self._job_states: dict[str, JobHealthState] = {}
        self._daemon_states: dict[str, DaemonHealthState] = {}  # December 2025

        # Circuit breakers - use shared implementation from app.distributed
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Statistics
        self._total_errors = 0
        self._total_recoveries = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._total_recovery_time = 0.0

        # Callbacks
        self._error_callbacks: list[Callable[[ErrorRecord], None]] = []
        self._recovery_callbacks: list[Callable[[RecoveryAttempt], None]] = []
        self._circuit_breaker_callbacks: list[Callable[[str, bool], None]] = []
        self._escalation_callbacks: list[Callable[[str, str], None]] = []

        # Dependencies
        if notifier:
            self.set_dependency("notifier", notifier)

        # Subscription state
        self._subscribed = False

        # Mark ready
        self._status = CoordinatorStatus.READY

    def subscribe_to_events(self) -> bool:
        """Subscribe to health-related events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType  # Types still needed

            router = get_router()

            # Use enum directly (router normalizes both enum and .value)
            # Error events (from ErrorRecoveryCoordinator)
            router.subscribe(DataEventType.ERROR, self._on_error)
            router.subscribe(DataEventType.RECOVERY_INITIATED, self._on_recovery_initiated)
            router.subscribe(DataEventType.RECOVERY_COMPLETED, self._on_recovery_completed)
            router.subscribe(DataEventType.RECOVERY_FAILED, self._on_recovery_failed)
            router.subscribe(DataEventType.TRAINING_FAILED, self._on_training_failed)
            router.subscribe(DataEventType.TASK_FAILED, self._on_task_failed)
            router.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
            router.subscribe(DataEventType.REGRESSION_CRITICAL, self._on_regression_critical)

            # Node events (from RecoveryManager and P2P orchestrator)
            router.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)
            router.subscribe(DataEventType.HOST_ONLINE, self._on_host_online)
            router.subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered)

            # Parity monitoring (December 2025 - closes parity → alert loop)
            router.subscribe(DataEventType.PARITY_FAILURE_RATE_CHANGED, self._on_parity_failure_rate_changed)

            # Coordinator health monitoring (December 2025 - wires ghost event)
            router.subscribe(DataEventType.COORDINATOR_HEALTH_DEGRADED, self._on_coordinator_health_degraded)

            # Coordinator lifecycle events (December 2025 - P0 gap fix)
            router.subscribe(DataEventType.COORDINATOR_SHUTDOWN, self._on_coordinator_shutdown)
            router.subscribe(DataEventType.COORDINATOR_HEARTBEAT, self._on_coordinator_heartbeat)

            # Deadlock detection (December 2025 - critical lock contention handler)
            router.subscribe(DataEventType.DEADLOCK_DETECTED, self._on_deadlock_detected)

            # Split-brain detection (December 2025 - P2P cluster integrity)
            router.subscribe(DataEventType.SPLIT_BRAIN_DETECTED, self._on_split_brain_detected)

            # Cluster stall detection (December 2025 - stuck nodes recovery)
            router.subscribe(DataEventType.CLUSTER_STALL_DETECTED, self._on_cluster_stall_detected)

            # Daemon lifecycle events (December 2025 - wires orphaned events to health monitor)
            router.subscribe(DataEventType.DAEMON_STARTED, self._on_daemon_started)
            router.subscribe(DataEventType.DAEMON_STOPPED, self._on_daemon_stopped)

            # Daemon watchdog alerts (December 2025 - wires watchdog → health manager)
            router.subscribe(DataEventType.DAEMON_STATUS_CHANGED, self._on_daemon_status_changed)

            self._subscribed = True
            logger.info("[UnifiedHealthManager] Subscribed to health events via event router")
            return True

        except ImportError:
            logger.warning("[UnifiedHealthManager] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[UnifiedHealthManager] Failed to subscribe: {e}")
            return False

    # =========================================================================
    # ID Generators
    # =========================================================================

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        self._error_id_counter += 1
        return f"err_{int(time.time())}_{self._error_id_counter}"

    def _generate_recovery_id(self) -> str:
        """Generate a unique recovery ID."""
        self._recovery_id_counter += 1
        return f"rec_{int(time.time())}_{self._recovery_id_counter}"

    # =========================================================================
    # State Accessors
    # =========================================================================

    def _get_node_state(self, node_id: str) -> NodeHealthState:
        """Get or create node health state."""
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeHealthState(node_id=node_id)
        return self._node_states[node_id]

    def _get_job_state(self, work_id: str) -> JobHealthState:
        """Get or create job health state."""
        if work_id not in self._job_states:
            self._job_states[work_id] = JobHealthState(work_id=work_id)
        return self._job_states[work_id]

    def _get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self._circuit_breakers:
            self._circuit_breakers[component] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout,
                half_open_max_calls=2,
            )
        return self._circuit_breakers[component]

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_error(self, event) -> None:
        """Handle ERROR event."""
        payload = event.payload

        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component=payload.get("component", "unknown"),
            error_type=payload.get("error_type", "unknown"),
            message=payload.get("message", ""),
            node_id=payload.get("node_id", ""),
            severity=ErrorSeverity(payload.get("severity", "error")),
            context=payload.get("context", {}),
        )

        self._record_error(error)

    async def _on_recovery_initiated(self, event) -> None:
        """Handle RECOVERY_INITIATED event."""
        payload = event.payload

        recovery = RecoveryAttempt(
            recovery_id=self._generate_recovery_id(),
            error_id=payload.get("error_id", ""),
            component=payload.get("component", "unknown"),
            node_id=payload.get("node_id", ""),
            strategy=payload.get("strategy", "default"),
            status=RecoveryStatus.IN_PROGRESS,
            attempt_number=payload.get("attempt_number", 1),
        )

        self._active_recoveries[recovery.recovery_id] = recovery
        self._total_recoveries += 1

        logger.info(
            f"[UnifiedHealthManager] Recovery initiated: {recovery.recovery_id} "
            f"for {recovery.component} on {recovery.node_id}"
        )

    async def _on_recovery_completed(self, event) -> None:
        """Handle RECOVERY_COMPLETED event."""
        payload = event.payload
        recovery_id = payload.get("recovery_id", "")

        if recovery_id in self._active_recoveries:
            recovery = self._active_recoveries.pop(recovery_id)
        else:
            recovery = RecoveryAttempt(
                recovery_id=recovery_id or self._generate_recovery_id(),
                error_id=payload.get("error_id", ""),
                component=payload.get("component", "unknown"),
                node_id=payload.get("node_id", ""),
                strategy=payload.get("strategy", "default"),
            )

        recovery.status = RecoveryStatus.COMPLETED
        recovery.completed_at = time.time()
        recovery.success = True
        recovery.message = payload.get("message", "Recovery successful")

        self._record_recovery(recovery)
        self._successful_recoveries += 1
        self._total_recovery_time += recovery.duration

        # Reset circuit breaker on success
        self._on_component_success(recovery.component)

        logger.info(
            f"[UnifiedHealthManager] Recovery completed: {recovery.recovery_id} "
            f"in {recovery.duration:.1f}s"
        )

    async def _on_recovery_failed(self, event) -> None:
        """Handle RECOVERY_FAILED event."""
        payload = event.payload
        recovery_id = payload.get("recovery_id", "")

        if recovery_id in self._active_recoveries:
            recovery = self._active_recoveries.pop(recovery_id)
        else:
            recovery = RecoveryAttempt(
                recovery_id=recovery_id or self._generate_recovery_id(),
                error_id=payload.get("error_id", ""),
                component=payload.get("component", "unknown"),
                node_id=payload.get("node_id", ""),
                strategy=payload.get("strategy", "default"),
            )

        recovery.status = RecoveryStatus.FAILED
        recovery.completed_at = time.time()
        recovery.success = False
        recovery.message = payload.get("message", "Recovery failed")

        self._record_recovery(recovery)
        self._failed_recoveries += 1

        # Record failure for circuit breaker
        self._on_component_failure(recovery.component)

        logger.warning(
            f"[UnifiedHealthManager] Recovery failed: {recovery.recovery_id} "
            f"- {recovery.message}"
        )

    async def _on_training_failed(self, event) -> None:
        """Handle TRAINING_FAILED event."""
        payload = event.payload

        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component="training",
            error_type="training_failed",
            message=payload.get("error", "Training failed"),
            node_id=payload.get("node_id", ""),
            severity=ErrorSeverity.ERROR,
            context={
                "model_id": payload.get("model_id"),
                "iteration": payload.get("iteration"),
            },
        )

        self._record_error(error)
        self._on_component_failure("training")

    async def _on_task_failed(self, event) -> None:
        """Handle TASK_FAILED event."""
        payload = event.payload
        task_type = payload.get("task_type", "unknown")
        work_id = payload.get("work_id") or payload.get("task_id", "")

        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component=f"task:{task_type}",
            error_type="task_failed",
            message=payload.get("error", "Task failed"),
            node_id=payload.get("node_id", ""),
            severity=ErrorSeverity.ERROR,
            context={
                "task_id": payload.get("task_id"),
                "task_type": task_type,
            },
        )

        self._record_error(error)

        # Track job failure state
        if work_id:
            state = self._get_job_state(work_id)
            state.recovery_attempts += 1
            state.last_attempt_time = time.time()

    async def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED event."""
        payload = event.payload

        severity_map = {
            "minor": ErrorSeverity.WARNING,
            "moderate": ErrorSeverity.ERROR,
            "severe": ErrorSeverity.ERROR,
            "critical": ErrorSeverity.CRITICAL,
        }

        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component="model",
            error_type="regression_detected",
            message=payload.get("message", "Model regression detected"),
            severity=severity_map.get(payload.get("severity", "moderate"), ErrorSeverity.ERROR),
            context={
                "model_id": payload.get("model_id"),
                "metric": payload.get("metric"),
                "delta": payload.get("delta"),
            },
        )

        self._record_error(error)

    async def _on_regression_critical(self, event) -> None:
        """Handle REGRESSION_CRITICAL event - trigger immediate rollback.

        Added December 2025 to wire Regression → Rollback coupling.
        When a critical regression is detected, immediately trigger rollback
        to the previous stable model version.
        """
        payload = event.payload if hasattr(event, 'payload') else event

        model_id = payload.get("model_id", "")
        severity = payload.get("severity", "critical")
        win_rate = payload.get("win_rate_vs_heuristic", 0.0)
        config_key = payload.get("config_key", model_id)

        logger.warning(
            f"[UnifiedHealthManager] REGRESSION_CRITICAL received for {model_id}: "
            f"severity={severity}, win_rate={win_rate:.2%}"
        )

        # Record critical error
        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component="model",
            error_type="regression_critical",
            message=f"Critical regression detected - rollback needed: {model_id}",
            severity=ErrorSeverity.CRITICAL,
            context={
                "model_id": model_id,
                "config_key": config_key,
                "severity": severity,
                "win_rate": win_rate,
            },
        )
        self._record_error(error)

        # Trigger rollback
        try:
            from app.training.rollback_manager import RollbackManager

            # Get or create rollback manager
            if not hasattr(self, "_rollback_manager"):
                try:
                    from app.training.model_registry import ModelRegistry
                    registry = ModelRegistry()
                    self._rollback_manager = RollbackManager(registry)
                except ImportError:
                    logger.warning("ModelRegistry not available for rollback")
                    return

            result = self._rollback_manager.rollback_model(
                model_id=config_key,
                reason=f"Auto-rollback: Critical regression (win_rate={win_rate:.2%})",
                triggered_by="auto_regression_critical",
            )

            if result.get("success"):
                logger.info(
                    f"[UnifiedHealthManager] Rollback successful for {config_key}: "
                    f"v{result.get('from_version')} → v{result.get('to_version')}"
                )
                # Emit rollback event
                try:
                    from app.coordination.event_router import get_router, DataEventType

                    router = get_router()
                    await router.publish(
                        DataEventType.MODEL_PROMOTED.value,  # Re-use promotion event for rollback
                        {
                            "model_id": config_key,
                            "action": "rollback",
                            "from_version": result.get("from_version"),
                            "to_version": result.get("to_version"),
                            "reason": "auto_regression_critical",
                        }
                    )
                except (RuntimeError, OSError, ConnectionError) as e:
                    logger.warning(f"Could not emit rollback event: {e}")
            else:
                logger.error(
                    f"[UnifiedHealthManager] Rollback failed for {config_key}: "
                    f"{result.get('error')}"
                )

        except ImportError as e:
            logger.warning(f"[UnifiedHealthManager] RollbackManager not available: {e}")
        except Exception as e:
            logger.error(f"[UnifiedHealthManager] Rollback failed: {e}")

    async def _on_host_offline(self, event) -> None:
        """Handle HOST_OFFLINE event."""
        payload = event.payload
        node_id = payload.get("node_id") or payload.get("host_id", "")

        if not node_id:
            return

        state = self._get_node_state(node_id)
        state.is_online = False
        state.offline_since = time.time()
        state.consecutive_failures += 1

        logger.warning(f"[UnifiedHealthManager] Host offline: {node_id}")

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE event from P2P orchestrator.

        December 2025: Added for P2P cluster integration.
        """
        payload = event.payload if hasattr(event, "payload") else event
        node_id = payload.get("node_id") or payload.get("host_id", "")

        if not node_id:
            return

        state = self._get_node_state(node_id)
        state.is_online = True
        state.offline_since = 0.0
        state.consecutive_failures = 0

        logger.info(f"[UnifiedHealthManager] Host online: {node_id}")

    async def _on_node_recovered(self, event) -> None:
        """Handle NODE_RECOVERED event."""
        payload = event.payload
        node_id = payload.get("node_id") or payload.get("host_id", "")

        if not node_id:
            return

        if node_id in self._node_states:
            state = self._node_states[node_id]
            state.is_online = True
            state.consecutive_failures = 0
            state.offline_since = 0.0

            logger.info(f"[UnifiedHealthManager] Node recovered: {node_id}")

    async def _on_parity_failure_rate_changed(self, event) -> None:
        """Handle PARITY_FAILURE_RATE_CHANGED event - alert on parity issues.

        December 2025: Closes the parity → alert feedback loop.
        When TS/Python parity failure rate exceeds thresholds, record as an
        error and potentially trigger alerts for investigation.

        Parity failures indicate divergence between TypeScript (source of truth)
        and Python implementations, which can cause training on incorrect data.
        """
        payload = event.payload if hasattr(event, "payload") else event

        failure_rate = payload.get("failure_rate", 0.0)
        board_type = payload.get("board_type", "unknown")
        num_players = payload.get("num_players", 0)
        config_key = payload.get("config_key", f"{board_type}_{num_players}p")
        total_games = payload.get("total_games", 0)
        failed_games = payload.get("failed_games", 0)
        source = payload.get("source", "unknown")

        # Thresholds for severity levels
        CRITICAL_THRESHOLD = 0.05  # 5% failure rate is critical
        WARNING_THRESHOLD = 0.01  # 1% failure rate is concerning

        if failure_rate >= CRITICAL_THRESHOLD:
            severity = ErrorSeverity.CRITICAL
            message = (
                f"CRITICAL: Parity failure rate {failure_rate:.1%} for {config_key} "
                f"({failed_games}/{total_games} games). Training data may be corrupted."
            )
            logger.error(f"[UnifiedHealthManager] {message}")
        elif failure_rate >= WARNING_THRESHOLD:
            severity = ErrorSeverity.WARNING
            message = (
                f"WARNING: Parity failure rate {failure_rate:.1%} for {config_key} "
                f"({failed_games}/{total_games} games). Investigation recommended."
            )
            logger.warning(f"[UnifiedHealthManager] {message}")
        else:
            # Below threshold, just log for tracking
            logger.debug(
                f"[UnifiedHealthManager] Parity failure rate {failure_rate:.1%} "
                f"for {config_key} - within acceptable limits"
            )
            return  # Don't record as error

        # Record error for tracking and potential escalation
        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component="parity",
            error_type="parity_failure_rate",
            message=message,
            severity=severity,
            context={
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "failure_rate": failure_rate,
                "failed_games": failed_games,
                "total_games": total_games,
                "source": source,
            },
        )
        self._record_error(error)

        # Trigger circuit breaker for parity component if critical
        if severity == ErrorSeverity.CRITICAL:
            self._on_component_failure(f"parity:{config_key}")

            # Escalate critical parity failures
            await self._escalate_to_human(
                config_key,
                f"Critical parity failure rate: {failure_rate:.1%} "
                f"({failed_games}/{total_games} games)",
            )

    async def _on_coordinator_health_degraded(self, event) -> None:
        """Handle COORDINATOR_HEALTH_DEGRADED event - track and respond to coordinator issues.

        December 2025: Wired handler for COORDINATOR_HEALTH_DEGRADED events.
        When a coordinator reports degraded health (e.g., from consecutive handler failures),
        this handler:
        1. Records the health issue as an error
        2. Updates circuit breaker for the affected component
        3. May trigger recovery actions if severity warrants
        """
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "unknown")
        reason = payload.get("reason", "")
        health_score = payload.get("health_score", 0.5)
        issues = payload.get("issues", [])
        node_id = payload.get("node_id", "")

        # Determine severity based on health score
        if health_score < 0.3:
            severity = ErrorSeverity.CRITICAL
        elif health_score < 0.5:
            severity = ErrorSeverity.ERROR
        elif health_score < 0.7:
            severity = ErrorSeverity.WARNING
        else:
            severity = ErrorSeverity.INFO

        logger.warning(
            f"[UnifiedHealthManager] COORDINATOR_HEALTH_DEGRADED: {coordinator_name} "
            f"(health={health_score:.2f}, reason={reason})"
        )

        # Record as error for tracking
        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component=f"coordinator:{coordinator_name}",
            error_type="health_degraded",
            message=f"Coordinator health degraded: {reason}",
            node_id=node_id,
            severity=severity,
            context={
                "coordinator_name": coordinator_name,
                "health_score": health_score,
                "issues": issues[:5] if issues else [],  # Limit to first 5 issues
            },
        )
        self._record_error(error)

        # Update circuit breaker for the coordinator
        component_key = f"coordinator:{coordinator_name}"
        if health_score < 0.5:
            # Multiple failures implied by low health score
            for _ in range(2):  # Record multiple failures to potentially trip breaker
                self._on_component_failure(component_key)

        # For critical health issues, take additional action
        if severity == ErrorSeverity.CRITICAL:
            # Emit recovery event to attempt coordinator restart
            try:
                from app.coordination.event_router import get_router

                router = get_router()

                # Emit recovery initiation
                await router.publish(
                    "recovery_initiated",
                    {
                        "error_id": error.error_id,
                        "component": coordinator_name,
                        "node_id": node_id,
                        "strategy": "coordinator_restart",
                        "health_score": health_score,
                    },
                )
                logger.info(
                    f"[UnifiedHealthManager] Initiated recovery for degraded "
                    f"coordinator: {coordinator_name}"
                )

            except Exception as e:
                logger.error(
                    f"[UnifiedHealthManager] Failed to initiate coordinator recovery: {e}"
                )

            # Escalate critical coordinator failures
            await self._escalate_to_human(
                coordinator_name,
                f"Coordinator health critical (score={health_score:.2f}): {reason}",
            )

    async def _on_coordinator_shutdown(self, event) -> None:
        """Handle COORDINATOR_SHUTDOWN event - mark coordinator as offline.

        Dec 2025: P0 gap fix - wires coordinator lifecycle to health monitoring.
        When a coordinator gracefully shuts down, we:
        1. Mark the coordinator/node as offline in our health state
        2. Trip circuit breaker for the component
        3. Log for cluster visibility
        """
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "unknown")
        node_id = payload.get("node_id", coordinator_name)
        reason = payload.get("reason", "graceful_shutdown")
        timestamp = payload.get("timestamp", time.time())

        logger.info(
            f"[UnifiedHealthManager] COORDINATOR_SHUTDOWN: {coordinator_name} "
            f"(node={node_id}, reason={reason})"
        )

        # Update node health state
        node_state = self._get_node_state(node_id)
        node_state.is_healthy = False
        node_state.is_responsive = False
        node_state.last_health_update = timestamp
        node_state.failure_count += 1

        # Trip circuit breaker for this coordinator
        component_key = f"coordinator:{coordinator_name}"
        self._on_component_failure(component_key)

        # Record as informational error (shutdown is expected behavior)
        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component=f"coordinator:{coordinator_name}",
            error_type="coordinator_shutdown",
            message=f"Coordinator shutdown: {reason}",
            node_id=node_id,
            severity=ErrorSeverity.INFO,
            context={
                "coordinator_name": coordinator_name,
                "reason": reason,
                "shutdown_timestamp": timestamp,
            },
        )
        self._record_error(error)

    async def _on_coordinator_heartbeat(self, event) -> None:
        """Handle COORDINATOR_HEARTBEAT event - update liveness timestamp.

        Dec 2025: P0 gap fix - wires coordinator lifecycle to health monitoring.
        When a coordinator sends a heartbeat, we:
        1. Update its last-seen timestamp
        2. Mark it as healthy/responsive if it was previously offline
        3. Reset failure count on successful heartbeats
        """
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "unknown")
        node_id = payload.get("node_id", coordinator_name)
        health_score = payload.get("health_score", 1.0)
        timestamp = payload.get("timestamp", time.time())

        # Update node health state
        node_state = self._get_node_state(node_id)
        node_state.last_health_update = timestamp
        node_state.last_heartbeat = timestamp

        # If node was previously unhealthy, mark as recovered
        if not node_state.is_healthy:
            logger.info(
                f"[UnifiedHealthManager] Coordinator recovered via heartbeat: "
                f"{coordinator_name} (node={node_id})"
            )
            node_state.is_healthy = True
            node_state.is_responsive = True
            node_state.failure_count = 0

            # Record recovery
            recovery = RecoveryAttempt(
                recovery_id=self._generate_recovery_id(),
                error_id=f"shutdown_{node_id}",
                component=f"coordinator:{coordinator_name}",
                node_id=node_id,
                strategy="heartbeat_recovery",
                started_at=timestamp,
                completed_at=timestamp,
                success=True,
            )
            self._record_recovery(recovery)

            # Reset circuit breaker on recovery
            component_key = f"coordinator:{coordinator_name}"
            if component_key in self._circuit_breakers:
                self._circuit_breakers[component_key].record_success()
        else:
            # Already healthy, just update timestamps
            node_state.is_responsive = True

    async def _on_deadlock_detected(self, event) -> None:
        """Handle DEADLOCK_DETECTED event - log and trigger recovery.

        Dec 2025: Critical handler for lock contention and deadlocks.
        When a deadlock is detected between multiple resources/processes, we:
        1. Log critical error for immediate investigation
        2. Record involved resources and holders
        3. Increment error counters for monitoring
        4. Trigger circuit breaker to prevent cascade

        Note: Actual deadlock resolution (e.g., killing processes) should be
        handled by specialized recovery mechanisms, not here.
        """
        payload = event.payload if hasattr(event, "payload") else event

        resources = payload.get("resources", [])
        holders = payload.get("holders", [])

        logger.critical(
            f"[UnifiedHealthManager] DEADLOCK DETECTED: "
            f"Resources: {resources}, Holders: {holders}"
        )

        # Create error record
        error = ErrorRecord(
            error_id=f"deadlock_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="lock_manager",
            error_type="deadlock",
            message=f"Deadlock detected involving {len(resources)} resources",
            severity=ErrorSeverity.CRITICAL,
            context={
                "resources": resources,
                "holders": holders,
            },
        )
        self._record_error(error)

        # Trigger circuit breaker for lock manager
        for _ in range(3):  # Multiple failures to trip breaker
            self._on_component_failure("lock_manager")

        # Escalate for manual intervention
        await self._escalate_to_human(
            "lock_manager",
            f"Deadlock detected: {len(resources)} resources involved",
        )

    async def _on_split_brain_detected(self, event) -> None:
        """Handle SPLIT_BRAIN_DETECTED event - log and trigger resolution.

        Dec 2025: Critical handler for P2P cluster split-brain scenarios.
        When multiple leaders are detected in the cluster:
        1. Log critical error for immediate investigation
        2. Record involved leaders and voter information
        3. Trigger circuit breaker for P2P subsystem
        4. Escalate for human intervention

        Note: Actual split-brain resolution (demoting stale leaders) is handled
        by leader_election.py's _resolve_split_brain() method.
        """
        payload = event.payload if hasattr(event, "payload") else event

        leaders_seen = payload.get("leaders_seen", [])
        voter_count = payload.get("voter_count", 0)
        severity = payload.get("severity", "warning")

        logger.critical(
            f"[UnifiedHealthManager] SPLIT-BRAIN DETECTED: "
            f"Leaders: {leaders_seen}, Voters: {voter_count}, Severity: {severity}"
        )

        # Create error record
        error = ErrorRecord(
            error_id=f"split_brain_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="p2p_cluster",
            error_type="split_brain",
            message=f"Split-brain detected: {len(leaders_seen)} leaders seen",
            severity=ErrorSeverity.CRITICAL if severity == "critical" else ErrorSeverity.ERROR,
            context={
                "leaders_seen": leaders_seen,
                "voter_count": voter_count,
                "severity": severity,
            },
        )
        self._record_error(error)

        # Trigger circuit breaker for P2P subsystem
        for _ in range(3):
            self._on_component_failure("p2p_cluster")

        # Escalate for manual intervention
        await self._escalate_to_human(
            "p2p_cluster",
            f"Split-brain detected: {len(leaders_seen)} leaders in cluster",
        )

    async def _on_cluster_stall_detected(self, event) -> None:
        """Handle CLUSTER_STALL_DETECTED event - trigger node recovery.

        Dec 2025: Handler for stuck nodes that aren't making game progress.
        When cluster stall is detected:
        1. Log warning for investigation
        2. Record stalled nodes
        3. Mark nodes as unhealthy in tracking
        4. Trigger recovery action via node_recovery_daemon

        This handler connects stall detection to the recovery pipeline.
        """
        payload = event.payload if hasattr(event, "payload") else event

        stalled_nodes = payload.get("stalled_nodes", [])
        stall_duration_seconds = payload.get("stall_duration_seconds", 0)
        last_game_progress = payload.get("last_game_progress", 0)

        logger.warning(
            f"[UnifiedHealthManager] CLUSTER STALL DETECTED: "
            f"Nodes: {stalled_nodes}, Stall duration: {stall_duration_seconds}s"
        )

        # Mark stalled nodes as unhealthy
        for node_id in stalled_nodes:
            if node_id in self._node_states:
                state = self._node_states[node_id]
                state.is_responsive = False
                state.consecutive_failures += 1

        # Create error record
        error = ErrorRecord(
            error_id=f"cluster_stall_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="cluster",
            error_type="stall_detected",
            message=f"Cluster stall: {len(stalled_nodes)} nodes stuck for {stall_duration_seconds}s",
            severity=ErrorSeverity.WARNING,
            context={
                "stalled_nodes": stalled_nodes,
                "stall_duration_seconds": stall_duration_seconds,
                "last_game_progress": last_game_progress,
            },
        )
        self._record_error(error)

        # Emit NODE_UNHEALTHY for each stalled node to trigger recovery
        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()
            for node_id in stalled_nodes:
                bus.emit(DataEventType.NODE_UNHEALTHY, {
                    "node_id": node_id,
                    "reason": "cluster_stall",
                    "stall_duration_seconds": stall_duration_seconds,
                })
        except ImportError:
            logger.debug("[UnifiedHealthManager] data_events not available for recovery trigger")

    async def _on_daemon_started(self, event) -> None:
        """Handle DAEMON_STARTED event - track daemon health (December 2025).

        Tracks daemon starts across the cluster for health visibility.
        Updates daemon state, detects restarts, and logs for monitoring.
        """
        payload = event.payload if hasattr(event, "payload") else event

        daemon_name = payload.get("daemon_name", "unknown")
        hostname = payload.get("hostname", "unknown")
        daemon_key = f"{daemon_name}@{hostname}"

        # Get or create daemon state
        if daemon_key not in self._daemon_states:
            self._daemon_states[daemon_key] = DaemonHealthState(
                daemon_name=daemon_name,
                hostname=hostname,
            )

        state = self._daemon_states[daemon_key]

        # Track restart if daemon was previously running
        if state.is_running:
            state.restart_count += 1
            logger.warning(
                f"[UnifiedHealthManager] Daemon restarted: {daemon_name} on {hostname} "
                f"(restart #{state.restart_count})"
            )
        else:
            logger.info(f"[UnifiedHealthManager] Daemon started: {daemon_name} on {hostname}")

        state.is_running = True
        state.started_at = time.time()

    async def _on_daemon_stopped(self, event) -> None:
        """Handle DAEMON_STOPPED event - track daemon health (December 2025).

        Tracks daemon stops across the cluster for health visibility.
        Records stop reason for debugging and alerts on unexpected stops.
        """
        payload = event.payload if hasattr(event, "payload") else event

        daemon_name = payload.get("daemon_name", "unknown")
        hostname = payload.get("hostname", "unknown")
        reason = payload.get("reason", "normal")
        daemon_key = f"{daemon_name}@{hostname}"

        # Get or create daemon state
        if daemon_key not in self._daemon_states:
            self._daemon_states[daemon_key] = DaemonHealthState(
                daemon_name=daemon_name,
                hostname=hostname,
            )

        state = self._daemon_states[daemon_key]
        state.is_running = False
        state.stopped_at = time.time()
        state.last_stop_reason = reason

        # Log based on stop reason
        if reason in ("error", "crash", "killed"):
            logger.warning(
                f"[UnifiedHealthManager] Daemon stopped unexpectedly: {daemon_name} on {hostname} "
                f"(reason: {reason})"
            )
            # Record as error for monitoring
            error = ErrorRecord(
                error_id=f"daemon_stop_{int(time.time() * 1000)}",
                timestamp=time.time(),
                component=f"daemon:{daemon_name}",
                error_type="daemon_stopped",
                message=f"Daemon {daemon_name} stopped unexpectedly: {reason}",
                node_id=hostname,
                severity=ErrorSeverity.WARNING,
                context={
                    "daemon_name": daemon_name,
                    "hostname": hostname,
                    "reason": reason,
                },
            )
            self._record_error(error)
        else:
            logger.info(
                f"[UnifiedHealthManager] Daemon stopped: {daemon_name} on {hostname} ({reason})"
            )

    async def _on_daemon_status_changed(self, event) -> None:
        """Handle DAEMON_STATUS_CHANGED event from DaemonWatchdog (December 2025).

        Processes watchdog alerts for daemon health issues:
        - daemon_stuck: Task done but state RUNNING
        - daemon_crashed: Unexpected failure
        - daemon_import_failed: Import error, needs manual fix
        - daemon_restart_exhausted: Max restarts exceeded
        - daemon_auto_restarted: Successfully auto-restarted
        """
        payload = event.payload if hasattr(event, "payload") else event

        alert_type = payload.get("alert_type", "unknown")
        daemon_name = payload.get("daemon_name", "unknown")
        hostname = payload.get("hostname", "unknown")
        daemon_key = f"{daemon_name}@{hostname}"

        # Get or create daemon state
        if daemon_key not in self._daemon_states:
            self._daemon_states[daemon_key] = DaemonHealthState(
                daemon_name=daemon_name,
                hostname=hostname,
            )

        state = self._daemon_states[daemon_key]

        # Handle different alert types with appropriate severity
        if alert_type in ("daemon_crashed", "daemon_restart_exhausted", "daemon_import_failed"):
            # Critical issues - record as errors
            severity = ErrorSeverity.CRITICAL if alert_type == "daemon_restart_exhausted" else ErrorSeverity.ERROR
            state.is_running = False
            state.consecutive_failures += 1
            state.last_error = f"{alert_type}: {payload.get('message', '')}"

            logger.error(
                f"[UnifiedHealthManager] Watchdog alert: {alert_type} for {daemon_name} on {hostname}"
            )

            error = ErrorRecord(
                error_id=f"watchdog_{int(time.time() * 1000)}",
                timestamp=time.time(),
                component=f"daemon:{daemon_name}",
                error_type=alert_type,
                message=f"Watchdog detected {alert_type} for {daemon_name}",
                node_id=hostname,
                severity=severity,
                context=payload,
            )
            self._record_error(error)

            # Track component failure for health scoring
            self._on_component_failure(f"daemon:{daemon_name}")

        elif alert_type == "daemon_stuck":
            # Warning - daemon may need restart
            state.last_error = f"stuck: {payload.get('message', '')}"
            logger.warning(
                f"[UnifiedHealthManager] Watchdog alert: {daemon_name} appears stuck on {hostname}"
            )

            error = ErrorRecord(
                error_id=f"watchdog_{int(time.time() * 1000)}",
                timestamp=time.time(),
                component=f"daemon:{daemon_name}",
                error_type="daemon_stuck",
                message=f"Daemon {daemon_name} appears stuck (task done but state RUNNING)",
                node_id=hostname,
                severity=ErrorSeverity.WARNING,
                context=payload,
            )
            self._record_error(error)

        elif alert_type == "daemon_auto_restarted":
            # Informational - auto-restart succeeded
            state.restart_count += 1
            state.is_running = True
            state.last_error = None
            state.consecutive_failures = 0

            logger.info(
                f"[UnifiedHealthManager] Watchdog auto-restarted {daemon_name} on {hostname} "
                f"(restarts: {state.restart_count})"
            )

            # Track component recovery for health scoring
            self._on_component_success(f"daemon:{daemon_name}")

        else:
            # Unknown alert type - log for debugging
            logger.debug(
                f"[UnifiedHealthManager] Unknown watchdog alert: {alert_type} for {daemon_name}"
            )

    def get_daemon_states(self) -> dict[str, DaemonHealthState]:
        """Get all tracked daemon states (December 2025).

        Returns:
            Dict mapping daemon_key (daemon_name@hostname) to DaemonHealthState
        """
        return dict(self._daemon_states)

    def get_running_daemons(self) -> list[str]:
        """Get list of currently running daemons (December 2025).

        Returns:
            List of daemon keys (daemon_name@hostname) for running daemons
        """
        return [key for key, state in self._daemon_states.items() if state.is_running]

    # =========================================================================
    # Error and Recovery Recording
    # =========================================================================

    def _record_error(self, error: ErrorRecord) -> None:
        """Record an error and update statistics."""
        self._errors.append(error)
        self._errors_by_component[error.component].append(error)
        self._total_errors += 1

        # Trim history
        if len(self._errors) > self.config.max_error_history:
            self._errors = self._errors[-self.config.max_error_history :]

        # Notify callbacks
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"[UnifiedHealthManager] Error callback failed: {e}")

        # Record failure for circuit breaker
        self._on_component_failure(error.component)

        logger.debug(
            f"[UnifiedHealthManager] Error recorded: {error.error_id} "
            f"({error.component}: {error.error_type})"
        )

    def _record_recovery(self, recovery: RecoveryAttempt) -> None:
        """Record a recovery attempt in history."""
        self._recovery_history.append(recovery)

        # Trim history
        if len(self._recovery_history) > self.config.max_recovery_history:
            self._recovery_history = self._recovery_history[-self.config.max_recovery_history :]

        # Notify callbacks
        for callback in self._recovery_callbacks:
            try:
                callback(recovery)
            except Exception as e:
                logger.error(f"[UnifiedHealthManager] Recovery callback failed: {e}")

    def _record_event(
        self,
        action: RecoveryAction,
        target_type: str,
        target_id: str,
        result: RecoveryResult,
        reason: str,
        error: str | None = None,
        duration: float = 0.0,
    ) -> None:
        """Record a recovery event."""
        event = RecoveryEvent(
            timestamp=time.time(),
            action=action,
            target_type=target_type,
            target_id=target_id,
            result=result,
            reason=reason,
            error=error,
            duration_seconds=duration,
        )
        self._recovery_events.append(event)

        # Keep last 500 events
        if len(self._recovery_events) > 500:
            self._recovery_events = self._recovery_events[-500:]

    # =========================================================================
    # Circuit Breaker Management
    # =========================================================================

    def _on_component_failure(self, component: str) -> None:
        """Record a failure for circuit breaker tracking."""
        cb = self._get_circuit_breaker(component)
        cb.record_failure()

        if cb.state == CircuitState.OPEN:
            # Notify callbacks
            for callback in self._circuit_breaker_callbacks:
                try:
                    callback(component, True)
                except Exception as e:
                    logger.error(f"[UnifiedHealthManager] CB callback failed: {e}")

            logger.warning(f"[UnifiedHealthManager] Circuit breaker OPEN for {component}")

    def _on_component_success(self, component: str) -> None:
        """Record a success for circuit breaker tracking."""
        if component not in self._circuit_breakers:
            return

        cb = self._circuit_breakers[component]
        was_open = cb.state == CircuitState.OPEN

        cb.record_success()

        if was_open and cb.state == CircuitState.CLOSED:
            # Notify callbacks
            for callback in self._circuit_breaker_callbacks:
                try:
                    callback(component, False)
                except Exception as e:
                    logger.error(f"[UnifiedHealthManager] CB callback failed: {e}")

            logger.info(f"[UnifiedHealthManager] Circuit breaker CLOSED for {component}")

    def is_circuit_broken(self, component: str) -> bool:
        """Check if a component's circuit breaker is open.

        Args:
            component: Component name to check

        Returns:
            True if circuit is open (component should not be used)
        """
        if component not in self._circuit_breakers:
            return False

        cb = self._circuit_breakers[component]
        return cb.state == CircuitState.OPEN

    # =========================================================================
    # Recovery Operations (from RecoveryManager)
    # =========================================================================

    def _can_attempt_node_recovery(self, node_id: str) -> bool:
        """Check if we can attempt recovery on this node."""
        state = self._get_node_state(node_id)

        # Check if already escalated
        if state.is_escalated:
            if time.time() - state.last_escalation_time < self.config.escalation_cooldown:
                return False
            state.is_escalated = False

        # Check attempt limit
        if state.recovery_attempts >= self.config.max_recovery_attempts_per_node:
            return False

        # Check cooldown
        return time.time() - state.last_attempt_time >= self.config.recovery_attempt_cooldown

    def _can_attempt_job_recovery(self, work_id: str) -> bool:
        """Check if we can attempt recovery on this job."""
        state = self._get_job_state(work_id)
        return state.recovery_attempts < self.config.max_recovery_attempts_per_job

    async def recover_stuck_job(
        self,
        work_item: "WorkItem",
        expected_timeout: float,
    ) -> RecoveryResult:
        """Attempt to recover a stuck job.

        Args:
            work_item: The stuck work item
            expected_timeout: Expected timeout in seconds

        Returns:
            RecoveryResult indicating success/failure/escalation
        """
        if not self.config.enabled:
            return RecoveryResult.SKIPPED

        work_id = work_item.work_id
        node_id = work_item.claimed_by

        logger.info(f"Attempting to recover stuck job {work_id} on node {node_id}")

        job_state = self._get_job_state(work_id)

        if not self._can_attempt_job_recovery(work_id):
            logger.warning(f"Max recovery attempts reached for job {work_id}")
            return RecoveryResult.ESCALATED

        start_time = time.time()
        job_state.recovery_attempts += 1
        job_state.last_attempt_time = start_time

        try:
            # Kill the job on the node
            kill_callback = self.get_dependency("kill_job_callback")
            if kill_callback and node_id:
                await kill_callback(node_id, work_id)

            # Mark as failed in work queue
            work_queue = self.get_dependency("work_queue")
            if work_queue:
                work_queue.fail_work(work_id, "stuck_timeout_recovery")

            duration = time.time() - start_time
            self._record_event(
                action=RecoveryAction.KILL_JOB,
                target_type="job",
                target_id=work_id,
                result=RecoveryResult.SUCCESS,
                reason=f"job_stuck_exceeded_{expected_timeout}s",
                duration=duration,
            )

            logger.info(f"Successfully recovered stuck job {work_id}")
            return RecoveryResult.SUCCESS

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to recover stuck job {work_id}: {e}")

            self._record_event(
                action=RecoveryAction.KILL_JOB,
                target_type="job",
                target_id=work_id,
                result=RecoveryResult.FAILED,
                reason=f"job_stuck_exceeded_{expected_timeout}s",
                error=str(e),
                duration=duration,
            )

            return RecoveryResult.FAILED

    async def recover_unhealthy_node(self, node_id: str, reason: str) -> RecoveryResult:
        """Attempt to recover an unhealthy node.

        Args:
            node_id: The node to recover
            reason: Reason for recovery attempt

        Returns:
            RecoveryResult indicating success/failure/escalation
        """
        if not self.config.enabled:
            return RecoveryResult.SKIPPED

        logger.info(f"Attempting to recover unhealthy node {node_id}: {reason}")

        node_state = self._get_node_state(node_id)

        if not self._can_attempt_node_recovery(node_id):
            await self._escalate_to_human(node_id, reason)
            return RecoveryResult.ESCALATED

        start_time = time.time()
        node_state.recovery_attempts += 1
        node_state.last_attempt_time = start_time

        try:
            # Try restarting services first
            restart_callback = self.get_dependency("restart_services_callback")
            if restart_callback:
                success = await asyncio.wait_for(
                    restart_callback(node_id),
                    timeout=self.config.node_recovery_timeout,
                )

                if success:
                    duration = time.time() - start_time
                    node_state.consecutive_failures = 0

                    self._record_event(
                        action=RecoveryAction.RESTART_NODE_SERVICES,
                        target_type="node",
                        target_id=node_id,
                        result=RecoveryResult.SUCCESS,
                        reason=reason,
                        duration=duration,
                    )

                    logger.info(f"Successfully recovered node {node_id} via service restart")
                    return RecoveryResult.SUCCESS

            # If service restart failed, increment failure count
            node_state.consecutive_failures += 1

            # Check if we should escalate
            if node_state.consecutive_failures >= self.config.consecutive_failures_for_escalation:
                await self._escalate_to_human(
                    node_id, f"{reason} - {node_state.consecutive_failures} consecutive failures"
                )
                return RecoveryResult.ESCALATED

            duration = time.time() - start_time
            self._record_event(
                action=RecoveryAction.RESTART_NODE_SERVICES,
                target_type="node",
                target_id=node_id,
                result=RecoveryResult.FAILED,
                reason=reason,
                duration=duration,
            )

            return RecoveryResult.FAILED

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            node_state.consecutive_failures += 1

            logger.error(f"Recovery timeout for node {node_id}")

            self._record_event(
                action=RecoveryAction.RESTART_NODE_SERVICES,
                target_type="node",
                target_id=node_id,
                result=RecoveryResult.FAILED,
                reason=reason,
                error="timeout",
                duration=duration,
            )

            return RecoveryResult.FAILED

        except Exception as e:
            duration = time.time() - start_time
            node_state.consecutive_failures += 1

            logger.error(f"Failed to recover node {node_id}: {e}")

            self._record_event(
                action=RecoveryAction.RESTART_NODE_SERVICES,
                target_type="node",
                target_id=node_id,
                result=RecoveryResult.FAILED,
                reason=reason,
                error=str(e),
                duration=duration,
            )

            return RecoveryResult.FAILED

    async def _escalate_to_human(self, target_id: str, reason: str) -> None:
        """Escalate an issue to human operators."""
        logger.warning(f"Escalating to human: {target_id} - {reason}")

        if target_id in self._node_states:
            state = self._node_states[target_id]
            state.is_escalated = True
            state.last_escalation_time = time.time()

        self._record_event(
            action=RecoveryAction.ESCALATE_HUMAN,
            target_type="node" if target_id in self._node_states else "job",
            target_id=target_id,
            result=RecoveryResult.ESCALATED,
            reason=reason,
        )

        # Emit NODE_OVERLOADED event for job redistribution (Phase 21.2 - Dec 2025)
        if HAS_NODE_EVENTS and emit_node_overloaded and target_id in self._node_states:
            try:
                state = self._node_states[target_id]
                await emit_node_overloaded(
                    host=target_id,
                    cpu_percent=100.0,  # Assume overloaded when escalated
                    gpu_percent=0.0,  # No GPU info available from health manager
                    memory_percent=0.0,
                    resource_type="consecutive_failures",
                    source="unified_health_manager.py",
                )
                logger.info(f"[UnifiedHealthManager] Emitted NODE_OVERLOADED for {target_id}")
            except Exception as e:
                logger.debug(f"[UnifiedHealthManager] Failed to emit NODE_OVERLOADED: {e}")

        # Notify escalation callbacks
        for callback in self._escalation_callbacks:
            try:
                callback(target_id, reason)
            except Exception as e:
                logger.error(f"[UnifiedHealthManager] Escalation callback failed: {e}")

        # Send notification
        notifier = self.get_dependency("notifier")
        if notifier:
            try:
                await notifier.send_escalation_alert(
                    target_id=target_id,
                    reason=reason,
                    recovery_attempts=self._node_states.get(
                        target_id, NodeHealthState(target_id)
                    ).recovery_attempts,
                )
            except Exception as e:
                logger.error(f"Failed to send escalation notification: {e}")

    def find_stuck_jobs(
        self,
        running_items: list["WorkItem"],
        timeout_multiplier: float | None = None,
    ) -> list[tuple["WorkItem", float]]:
        """Find jobs that appear to be stuck.

        Args:
            running_items: List of currently running work items
            timeout_multiplier: Override for stuck detection

        Returns:
            List of (work_item, expected_timeout) tuples for stuck jobs
        """
        multiplier = timeout_multiplier or self.config.stuck_job_timeout_multiplier
        stuck_jobs = []
        current_time = time.time()

        for item in running_items:
            expected_timeout = item.timeout_seconds
            actual_runtime = current_time - item.started_at if item.started_at else 0

            if actual_runtime > expected_timeout * multiplier:
                stuck_jobs.append((item, expected_timeout))
                logger.debug(
                    f"Detected stuck job {item.work_id}: "
                    f"runtime={actual_runtime:.0f}s > expected={expected_timeout * multiplier:.0f}s"
                )

        return stuck_jobs

    # =========================================================================
    # Public API for Error Recording
    # =========================================================================

    def record_error(
        self,
        component: str,
        error_type: str,
        message: str,
        node_id: str = "",
        severity: str = "error",
        context: dict | None = None,
    ) -> ErrorRecord:
        """Manually record an error.

        Returns:
            The created ErrorRecord
        """
        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component=component,
            error_type=error_type,
            message=message,
            node_id=node_id,
            severity=ErrorSeverity(severity),
            context=context or {},
        )

        self._record_error(error)
        return error

    def start_recovery(
        self,
        error_id: str,
        component: str,
        node_id: str,
        strategy: str = "default",
    ) -> RecoveryAttempt:
        """Start a recovery attempt.

        Returns:
            The created RecoveryAttempt
        """
        recovery = RecoveryAttempt(
            recovery_id=self._generate_recovery_id(),
            error_id=error_id,
            component=component,
            node_id=node_id,
            strategy=strategy,
            status=RecoveryStatus.IN_PROGRESS,
        )

        self._active_recoveries[recovery.recovery_id] = recovery
        self._total_recoveries += 1

        return recovery

    def complete_recovery(self, recovery_id: str, success: bool, message: str = "") -> None:
        """Complete a recovery attempt."""
        if recovery_id not in self._active_recoveries:
            return

        recovery = self._active_recoveries.pop(recovery_id)
        recovery.status = RecoveryStatus.COMPLETED if success else RecoveryStatus.FAILED
        recovery.completed_at = time.time()
        recovery.success = success
        recovery.message = message

        self._record_recovery(recovery)

        if success:
            self._successful_recoveries += 1
            self._total_recovery_time += recovery.duration
            self._on_component_success(recovery.component)
        else:
            self._failed_recoveries += 1
            self._on_component_failure(recovery.component)

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_error(self, callback: Callable[[ErrorRecord], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_recovery(self, callback: Callable[[RecoveryAttempt], None]) -> None:
        """Register callback for recovery completions."""
        self._recovery_callbacks.append(callback)

    def on_circuit_breaker_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for circuit breaker state changes."""
        self._circuit_breaker_callbacks.append(callback)

    def on_escalation(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for escalations."""
        self._escalation_callbacks.append(callback)

    # =========================================================================
    # Dependency Setters (legacy compatibility)
    # =========================================================================

    def set_work_queue(self, work_queue: "WorkItem") -> None:
        """Set the work queue reference."""
        self.set_dependency("work_queue", work_queue)

    def set_notifier(self, notifier: Any) -> None:
        """Set the notification service."""
        self.set_dependency("notifier", notifier)

    def set_kill_job_callback(self, callback: Callable) -> None:
        """Set callback for killing jobs."""
        self.set_dependency("kill_job_callback", callback)

    def set_restart_services_callback(self, callback: Callable) -> None:
        """Set callback for restarting node services."""
        self.set_dependency("restart_services_callback", callback)

    def set_reboot_node_callback(self, callback: Callable) -> None:
        """Set callback for rebooting nodes."""
        self.set_dependency("reboot_node_callback", callback)

    # =========================================================================
    # State Reset
    # =========================================================================

    def reset_node_state(self, node_id: str) -> None:
        """Reset health state for a node."""
        if node_id in self._node_states:
            self._node_states[node_id] = NodeHealthState(node_id=node_id)

    def reset_job_state(self, work_id: str) -> None:
        """Reset health state for a job."""
        if work_id in self._job_states:
            del self._job_states[work_id]

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_recent_errors(self, limit: int = 50) -> list[ErrorRecord]:
        """Get recent errors."""
        return self._errors[-limit:]

    def get_errors_by_component(self, component: str) -> list[ErrorRecord]:
        """Get errors for a specific component."""
        return list(self._errors_by_component.get(component, []))

    def get_recovery_history(self, limit: int = 50) -> list[RecoveryAttempt]:
        """Get recent recovery attempts."""
        return self._recovery_history[-limit:]

    def get_active_recoveries(self) -> list[RecoveryAttempt]:
        """Get active recovery attempts."""
        return list(self._active_recoveries.values())

    def get_circuit_breaker_states(self) -> dict[str, CircuitState]:
        """Get all circuit breaker states."""
        return {
            component: cb.state for component, cb in self._circuit_breakers.items()
        }

    def get_online_nodes(self) -> set[str]:
        """Get set of online nodes."""
        return {
            node_id
            for node_id, state in self._node_states.items()
            if state.is_online
        }

    def get_offline_nodes(self) -> dict[str, float]:
        """Get offline nodes with offline timestamp."""
        return {
            node_id: state.offline_since
            for node_id, state in self._node_states.items()
            if not state.is_online and state.offline_since > 0
        }

    def get_escalated_nodes(self) -> list[str]:
        """Get list of escalated nodes."""
        return [
            node_id
            for node_id, state in self._node_states.items()
            if state.is_escalated
        ]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_health_stats(self) -> HealthStats:
        """Get aggregate health statistics."""
        # Count by severity
        by_severity: dict[str, int] = defaultdict(int)
        for error in self._errors:
            by_severity[error.severity.value] += 1

        # Count by component
        by_component: dict[str, int] = {
            comp: len(errors) for comp, errors in self._errors_by_component.items()
        }

        # Count by node
        by_node: dict[str, int] = defaultdict(int)
        for error in self._errors:
            if error.node_id:
                by_node[error.node_id] += 1

        # Recovery rate
        recovery_rate = (
            self._successful_recoveries / self._total_recoveries
            if self._total_recoveries > 0
            else 0.0
        )

        # Average recovery time
        avg_recovery_time = (
            self._total_recovery_time / self._successful_recoveries
            if self._successful_recoveries > 0
            else 0.0
        )

        # Circuit breaker stats
        open_circuits = [
            comp for comp, cb in self._circuit_breakers.items() if cb.state == CircuitState.OPEN
        ]

        # Node stats
        online_nodes = sum(1 for s in self._node_states.values() if s.is_online)
        offline_nodes = sum(1 for s in self._node_states.values() if not s.is_online)

        return HealthStats(
            total_errors=self._total_errors,
            errors_by_severity=dict(by_severity),
            errors_by_component=by_component,
            errors_by_node=dict(by_node),
            recovery_attempts=self._total_recoveries,
            successful_recoveries=self._successful_recoveries,
            failed_recoveries=self._failed_recoveries,
            recovery_rate=recovery_rate,
            avg_recovery_time=avg_recovery_time,
            circuit_breakers_open=len(open_circuits),
            open_circuits=open_circuits,
            nodes_tracked=len(self._node_states),
            nodes_online=online_nodes,
            nodes_offline=offline_nodes,
            escalated_nodes=self.get_escalated_nodes(),
            jobs_tracked=len(self._job_states),
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get recovery statistics for monitoring.

        Implements CoordinatorBase.get_stats() interface.
        """
        base_stats = await super().get_stats()
        health_stats = self.get_health_stats()

        recent_events = [
            e for e in self._recovery_events if time.time() - e.timestamp < 3600
        ]
        success_count = sum(1 for e in recent_events if e.result == RecoveryResult.SUCCESS)
        failed_count = sum(1 for e in recent_events if e.result == RecoveryResult.FAILED)
        escalated_count = sum(1 for e in recent_events if e.result == RecoveryResult.ESCALATED)

        base_stats.update({
            "enabled": self.config.enabled,
            "total_errors": health_stats.total_errors,
            "errors_by_severity": health_stats.errors_by_severity,
            "errors_by_component": health_stats.errors_by_component,
            "recovery_attempts": health_stats.recovery_attempts,
            "successful_recoveries": health_stats.successful_recoveries,
            "failed_recoveries": health_stats.failed_recoveries,
            "recovery_rate": round(health_stats.recovery_rate * 100, 1),
            "avg_recovery_time": round(health_stats.avg_recovery_time, 1),
            "active_recoveries": len(self._active_recoveries),
            "circuit_breakers_open": health_stats.circuit_breakers_open,
            "open_circuits": health_stats.open_circuits,
            "recoveries_last_hour": {
                "success": success_count,
                "failed": failed_count,
                "escalated": escalated_count,
            },
            "nodes_tracked": health_stats.nodes_tracked,
            "nodes_online": health_stats.nodes_online,
            "nodes_offline": health_stats.nodes_offline,
            "escalated_nodes": health_stats.escalated_nodes,
            "jobs_tracked": health_stats.jobs_tracked,
            "subscribed": self._subscribed,
        })
        return base_stats

    def health_check(self) -> "HealthCheckResult":
        """Check if the health manager is healthy (CoordinatorProtocol compliance).

        Returns HealthCheckResult for consistent health monitoring interface.
        """
        from app.coordination.protocols import HealthCheckResult

        if self.status != CoordinatorStatus.READY:
            return HealthCheckResult(
                healthy=False,
                status=self.status,
                message="Health manager not ready",
            )

        # Check event subscription is active
        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="Not subscribed to events",
            )

        # Check we haven't accumulated too many unrecovered errors
        unrecovered = self._total_errors - self._successful_recoveries
        if unrecovered > 100:
            return HealthCheckResult(
                healthy=True,  # Still operational but degraded
                status=CoordinatorStatus.DEGRADED,
                message=f"{unrecovered} unrecovered errors",
                details={"unrecovered_errors": unrecovered},
            )

        # Check we don't have too many open circuits (indicates system stress)
        health_stats = self.get_health_stats()
        if health_stats.circuit_breakers_open > 5:
            return HealthCheckResult(
                healthy=True,  # Still operational but stressed
                status=CoordinatorStatus.DEGRADED,
                message=f"{health_stats.circuit_breakers_open} circuit breakers open",
                details={"circuit_breakers_open": health_stats.circuit_breakers_open},
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.READY,
            message="Health manager running",
            details={"subscribed": self._subscribed, "total_errors": self._total_errors},
        )

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring (sync version)."""
        health_stats = self.get_health_stats()

        return {
            "name": self.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "total_errors": health_stats.total_errors,
            "errors_by_severity": health_stats.errors_by_severity,
            "recovery_attempts": health_stats.recovery_attempts,
            "successful_recoveries": health_stats.successful_recoveries,
            "failed_recoveries": health_stats.failed_recoveries,
            "recovery_rate": round(health_stats.recovery_rate * 100, 1),
            "active_recoveries": len(self._active_recoveries),
            "circuit_breakers_open": health_stats.circuit_breakers_open,
            "open_circuits": health_stats.open_circuits,
            "nodes_tracked": health_stats.nodes_tracked,
            "escalated_nodes": health_stats.escalated_nodes,
            "jobs_tracked": health_stats.jobs_tracked,
            "subscribed": self._subscribed,
        }

    # =========================================================================
    # System Health Scoring (consolidated from system_health_monitor.py)
    # =========================================================================

    def calculate_system_health_score(
        self, sys_config: SystemHealthConfig | None = None
    ) -> SystemHealthScore:
        """Calculate aggregate system health score.

        Consolidated from system_health_monitor.py - now directly available
        on the health manager.

        Args:
            sys_config: Optional system health config, uses defaults if not provided

        Returns:
            SystemHealthScore with aggregate health data
        """
        cfg = sys_config or SystemHealthConfig()

        # Calculate component scores
        node_availability = self._calculate_node_availability(cfg)
        circuit_health = self._calculate_circuit_health(cfg)
        error_rate = self._calculate_error_rate_score(cfg)
        recovery_success = self._calculate_recovery_success_score()

        # Weighted aggregate
        score = (
            node_availability * cfg.node_weight
            + circuit_health * cfg.circuit_weight
            + error_rate * cfg.error_weight
            + recovery_success * cfg.recovery_weight
        )

        score = int(max(0, min(100, score)))

        # Determine level
        if score >= cfg.healthy_threshold:
            level = SystemHealthLevel.HEALTHY
        elif score >= cfg.degraded_threshold:
            level = SystemHealthLevel.DEGRADED
        elif score >= cfg.unhealthy_threshold:
            level = SystemHealthLevel.UNHEALTHY
        else:
            level = SystemHealthLevel.CRITICAL

        # Check pause triggers
        pause_triggers = self._check_pause_triggers(
            cfg, score, node_availability, circuit_health, error_rate
        )

        return SystemHealthScore(
            score=score,
            level=level,
            components={
                "node_availability": round(node_availability, 1),
                "circuit_health": round(circuit_health, 1),
                "error_rate": round(error_rate, 1),
                "recovery_success": round(recovery_success, 1),
            },
            node_availability=node_availability,
            circuit_health=circuit_health,
            error_rate=error_rate,
            recovery_success=recovery_success,
            pause_triggers=pause_triggers,
        )

    def _calculate_node_availability(self, cfg: SystemHealthConfig) -> float:
        """Calculate node availability score (0-100)."""
        nodes_tracked = len(self._node_states)
        nodes_online = sum(1 for s in self._node_states.values() if s.is_online)

        # Determine expected nodes
        expected = cfg.expected_nodes
        if expected == 0:
            expected = max(nodes_tracked, 1)

        # Calculate availability
        availability = (nodes_online / expected) * 100 if expected > 0 else 100.0
        return min(100.0, availability)

    def _calculate_circuit_health(self, cfg: SystemHealthConfig) -> float:
        """Calculate circuit breaker health score (0-100)."""
        total_circuits = len(self._circuit_breakers)
        if total_circuits == 0:
            return 100.0

        open_circuits = sum(
            1
            for cb in self._circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )

        # Circuits closed percentage
        closed_percent = ((total_circuits - open_circuits) / total_circuits) * 100

        # Extra penalty for critical circuits
        critical_open = [
            c
            for c in cfg.critical_circuits
            if c in self._circuit_breakers
            and self._circuit_breakers[c].state == CircuitState.OPEN
        ]

        if critical_open:
            # Heavy penalty for critical circuits
            penalty = len(critical_open) * 20
            closed_percent = max(0, closed_percent - penalty)

        return closed_percent

    def _calculate_error_rate_score(self, cfg: SystemHealthConfig) -> float:
        """Calculate error rate score (0-100, inverted: higher = fewer errors)."""
        # Check recent errors
        now = time.time()
        window = cfg.pause_error_burst_window
        recent_errors = [e for e in self._errors if now - e.timestamp < window]

        error_count = len(recent_errors)

        # Score based on error count
        threshold = cfg.pause_error_burst_count
        if error_count >= threshold:
            return 0.0

        score = ((threshold - error_count) / threshold) * 100
        return max(0.0, min(100.0, score))

    def _calculate_recovery_success_score(self) -> float:
        """Calculate recovery success rate (0-100)."""
        total = self._total_recoveries
        successful = self._successful_recoveries

        if total == 0:
            return 100.0  # No recoveries needed = healthy

        return (successful / total) * 100

    def _check_pause_triggers(
        self,
        cfg: SystemHealthConfig,
        score: int,
        node_availability: float,
        circuit_health: float,
        error_rate: float,
    ) -> list[str]:
        """Check for conditions that should trigger pipeline pause."""
        triggers = []

        # Health score threshold
        if score < cfg.pause_health_threshold:
            triggers.append(f"health_score_critical:{score}")

        # Node offline threshold
        offline_percent = (100 - node_availability) / 100
        if offline_percent >= cfg.pause_node_offline_percent:
            triggers.append(f"nodes_offline:{offline_percent:.0%}")

        # Critical circuit broken
        for circuit_name in cfg.critical_circuits:
            if circuit_name in self._circuit_breakers:
                cb = self._circuit_breakers[circuit_name]
                if cb.state == CircuitState.OPEN:
                    triggers.append(f"critical_circuit_open:{circuit_name}")

        # Error burst
        if error_rate == 0:
            triggers.append("error_burst_detected")

        return triggers


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_health_manager: UnifiedHealthManager | None = None


def get_health_manager() -> UnifiedHealthManager:
    """Get the global UnifiedHealthManager singleton."""
    global _health_manager
    if _health_manager is None:
        _health_manager = UnifiedHealthManager()
    return _health_manager


def wire_health_events(
    config: RecoveryConfig | None = None,
) -> UnifiedHealthManager:
    """Wire health events to the manager.

    Returns:
        The wired UnifiedHealthManager instance
    """
    global _health_manager
    _health_manager = UnifiedHealthManager(config=config)
    _health_manager.subscribe_to_events()
    return _health_manager


def reset_health_manager() -> None:
    """Reset the global health manager (for testing)."""
    global _health_manager
    _health_manager = None


def is_component_healthy(component: str) -> bool:
    """Check if a component is healthy (circuit not broken)."""
    return not get_health_manager().is_circuit_broken(component)


# =============================================================================
# Backward Compatibility Layer
# =============================================================================


def get_error_coordinator() -> "UnifiedHealthManager":
    """DEPRECATED: Use get_health_manager() instead.

    Returns the UnifiedHealthManager for backward compatibility.

    Returns:
        UnifiedHealthManager instance
    """
    warnings.warn(
        "get_error_coordinator() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


def wire_error_events() -> "UnifiedHealthManager":
    """DEPRECATED: Use wire_health_events() instead.

    Returns the UnifiedHealthManager for backward compatibility.

    Returns:
        UnifiedHealthManager instance
    """
    warnings.warn(
        "wire_error_events() is deprecated. Use wire_health_events() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return wire_health_events()


def get_recovery_manager() -> "UnifiedHealthManager":
    """DEPRECATED: Use get_health_manager() instead.

    Returns the UnifiedHealthManager for backward compatibility.

    Returns:
        UnifiedHealthManager instance
    """
    warnings.warn(
        "get_recovery_manager() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


def wire_recovery_events() -> "UnifiedHealthManager":
    """DEPRECATED: Use wire_health_events() instead.

    Returns the UnifiedHealthManager for backward compatibility.

    Returns:
        UnifiedHealthManager instance
    """
    warnings.warn(
        "wire_recovery_events() is deprecated. Use wire_health_events() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return wire_health_events()


def get_system_health_score() -> int:
    """Get current system health score (0-100).

    Convenience function for system health checks.
    Consolidated from system_health_monitor.py.
    """
    return get_health_manager().calculate_system_health_score().score


def get_system_health_level() -> SystemHealthLevel:
    """Get current system health level.

    Convenience function for system health checks.
    Consolidated from system_health_monitor.py.
    """
    return get_health_manager().calculate_system_health_score().level


def should_pause_pipeline(
    sys_config: SystemHealthConfig | None = None,
) -> tuple[bool, list[str]]:
    """Check if pipeline should be paused based on system health.

    Convenience function for pipeline control.
    Consolidated from system_health_monitor.py.

    Returns:
        Tuple of (should_pause, list of trigger reasons)
    """
    score = get_health_manager().calculate_system_health_score(sys_config)
    return len(score.pause_triggers) > 0, score.pause_triggers


# Backward compatibility - import aliases from system_health_monitor.py
def get_system_health():
    """DEPRECATED: Use get_health_manager() and calculate_system_health_score().

    Returns the UnifiedHealthManager for backward compatibility.
    """
    warnings.warn(
        "get_system_health() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


def is_pipeline_paused() -> bool:
    """DEPRECATED: Use should_pause_pipeline() instead.

    Check if pipeline should be paused.
    Consolidated from system_health_monitor.py.
    """
    should_pause, _ = should_pause_pipeline()
    return should_pause


__all__ = [
    # Enums
    "ErrorSeverity",
    "JobRecoveryAction",
    "PipelineState",
    "RecoveryAction",
    "RecoveryResult",
    "RecoveryStatus",
    "SystemHealthLevel",
    # Data classes
    "DaemonHealthState",
    "ErrorRecord",
    "HealthStats",
    "JobHealthState",
    "NodeHealthState",
    "RecoveryAttempt",
    "RecoveryConfig",
    "RecoveryEvent",
    "SystemHealthConfig",
    "SystemHealthScore",
    # Main class
    "UnifiedHealthManager",
    # Functions
    "get_health_manager",
    "get_system_health_level",
    "get_system_health_score",
    "is_component_healthy",
    "reset_health_manager",
    "should_pause_pipeline",
    "wire_health_events",
    # Deprecated (backward compatibility)
    "get_error_coordinator",
    "get_recovery_manager",
    "get_system_health",
    "is_pipeline_paused",
    "wire_error_events",
    "wire_recovery_events",
]

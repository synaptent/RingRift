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
- Subscribes to DAEMON_PERMANENTLY_FAILED: Handle daemons that exceeded restart limit (Dec 2025)
- Subscribes to DLQ_STALE_EVENTS: Track stale events in dead letter queue (Dec 29, 2025)
- Subscribes to DLQ_EVENTS_REPLAYED: Track successful event replays from DLQ (Dec 29, 2025)
- Subscribes to DLQ_EVENTS_PURGED: Track purged events for data loss visibility (Dec 29, 2025)
- Subscribes to CAPACITY_LOW: Track when cluster GPU capacity drops (Dec 29, 2025)
- Subscribes to CAPACITY_RESTORED: Track when capacity recovers (Dec 29, 2025)
- Subscribes to BUDGET_EXCEEDED: Track when spending exceeds limits (Dec 29, 2025)
- Subscribes to BUDGET_ALERT: Track approaching budget thresholds (Dec 29, 2025)

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

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_utils import make_config_key
from app.coordination.handler_base import HandlerBase
from app.coordination.unified_health_shared import (
    CircuitState,
    ErrorRecord,
    ErrorSeverity,
    HAS_NODE_EVENTS,
    HealthStats,
    JobRecoveryAction,
    NodeHealthState,
    NodeRecoveryState,
    RecoveryAction,
    RecoveryAttempt,
    RecoveryResult,
    RecoveryStatus,
    SystemHealthConfig,
    SystemHealthLevel,
    SystemHealthScore,
    emit_node_overloaded,
)
from app.distributed.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Enums (consolidated from both modules)
# =============================================================================

class PipelineState(Enum):
    """Pipeline operational state."""

    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"



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
# =============================================================================
# UnifiedHealthManager
# =============================================================================


from app.coordination.unified_health_recovery_mixin import UnifiedHealthRecoveryMixin

class UnifiedHealthManager(UnifiedHealthRecoveryMixin, HandlerBase):
    """Unified health management combining error tracking and recovery operations.

    This consolidates ErrorRecoveryCoordinator and RecoveryManager into a single
    cohesive system for health monitoring and recovery.

    Key Responsibilities:
    1. Track errors and failures across the cluster
    2. Manage circuit breakers for failing components
    3. Coordinate recovery operations (job kills, service restarts)
    4. Track node and job health states
    5. Handle escalation to human operators

    Session 17.17 (Jan 4, 2026): Migrated to HandlerBase for unified lifecycle
    and singleton management. This is a pure event handler with no background
    loop - _run_cycle is a no-op, all work is driven by event subscriptions.
    """

    # Hourly cycle since this is event-driven (cycle not used for actual work)
    HEALTH_CYCLE_INTERVAL = 3600.0

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
        resolved_config = config or RecoveryConfig()
        super().__init__(
            name="unified_health_manager",
            config=resolved_config,
            cycle_interval=self.HEALTH_CYCLE_INTERVAL,
        )
        # Use _health_config to avoid conflict with HandlerBase's config
        self._health_config = resolved_config

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

        # Dependencies - store for recovery escalation
        self._notifier = notifier
        self._dependencies: dict[str, Any] = {}
        if notifier is not None:
            self.set_dependency("notifier", notifier)

    def set_dependency(self, name: str, value: Any) -> None:
        """Set a named dependency for recovery operations.

        This provides compatibility with legacy code that expects
        CoordinatorBase.set_dependency() behavior.

        Args:
            name: Dependency name (e.g., 'work_queue', 'notifier')
            value: The dependency instance
        """
        self._dependencies[name] = value
        # Also set as attribute for direct access
        setattr(self, f"_{name}", value)

    def get_dependency(self, name: str) -> Any | None:
        """Get a named dependency.

        Args:
            name: Dependency name

        Returns:
            The dependency instance or None if not set
        """
        return self._dependencies.get(name)

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        This is the primary method for event subscription in HandlerBase.
        All health events are mapped to their respective handlers.
        """
        from app.distributed.data_events import DataEventType

        return {
            # Error events (from ErrorRecoveryCoordinator)
            DataEventType.ERROR.value: self._on_error,
            DataEventType.RECOVERY_INITIATED.value: self._on_recovery_initiated,
            DataEventType.RECOVERY_COMPLETED.value: self._on_recovery_completed,
            DataEventType.RECOVERY_FAILED.value: self._on_recovery_failed,
            DataEventType.TRAINING_FAILED.value: self._on_training_failed,
            DataEventType.TASK_FAILED.value: self._on_task_failed,
            DataEventType.REGRESSION_DETECTED.value: self._on_regression_detected,
            DataEventType.REGRESSION_CRITICAL.value: self._on_regression_critical,
            # Node events (from RecoveryManager and P2P orchestrator)
            DataEventType.HOST_OFFLINE.value: self._on_host_offline,
            DataEventType.HOST_ONLINE.value: self._on_host_online,
            DataEventType.NODE_RECOVERED.value: self._on_node_recovered,
            # Parity monitoring (December 2025)
            DataEventType.PARITY_FAILURE_RATE_CHANGED.value: self._on_parity_failure_rate_changed,
            # Coordinator health monitoring (December 2025)
            DataEventType.COORDINATOR_HEALTH_DEGRADED.value: self._on_coordinator_health_degraded,
            DataEventType.COORDINATOR_SHUTDOWN.value: self._on_coordinator_shutdown,
            DataEventType.COORDINATOR_HEARTBEAT.value: self._on_coordinator_heartbeat,
            # Deadlock detection (December 2025)
            DataEventType.DEADLOCK_DETECTED.value: self._on_deadlock_detected,
            # Split-brain detection (December 2025)
            DataEventType.SPLIT_BRAIN_DETECTED.value: self._on_split_brain_detected,
            # Cluster stall detection (December 2025)
            DataEventType.CLUSTER_STALL_DETECTED.value: self._on_cluster_stall_detected,
            # Daemon lifecycle events (December 2025)
            DataEventType.DAEMON_STARTED.value: self._on_daemon_started,
            DataEventType.DAEMON_STOPPED.value: self._on_daemon_stopped,
            DataEventType.DAEMON_STATUS_CHANGED.value: self._on_daemon_status_changed,
            DataEventType.DAEMON_PERMANENTLY_FAILED.value: self._on_daemon_permanently_failed,
            # DLQ events (December 29, 2025)
            DataEventType.DLQ_STALE_EVENTS.value: self._on_dlq_stale_events,
            DataEventType.DLQ_EVENTS_REPLAYED.value: self._on_dlq_events_replayed,
            DataEventType.DLQ_EVENTS_PURGED.value: self._on_dlq_events_purged,
            # Budget/Capacity events (December 29, 2025)
            DataEventType.CAPACITY_LOW.value: self._on_capacity_low,
            DataEventType.CAPACITY_RESTORED.value: self._on_capacity_restored,
            DataEventType.BUDGET_EXCEEDED.value: self._on_budget_exceeded,
            DataEventType.BUDGET_ALERT.value: self._on_budget_alert,
        }

    async def _run_cycle(self) -> None:
        """No-op cycle - UnifiedHealthManager is purely event-driven.

        All work is done through event handlers registered in _get_event_subscriptions().
        This method exists only to satisfy HandlerBase's abstract method requirement.
        The high cycle_interval (1 hour) means this rarely runs.
        """
        pass

    async def _on_start(self) -> None:
        """Log startup and verify subscriptions."""
        subscriptions = self._get_event_subscriptions()
        logger.info(
            f"[UnifiedHealthManager] Started with {len(subscriptions)} event subscriptions"
        )

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
                failure_threshold=self._health_config.circuit_breaker_threshold,
                recovery_timeout=self._health_config.circuit_breaker_timeout,
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
                    from app.coordination.event_router import DataEventType, publish

                    await publish(
                        event_type=DataEventType.MODEL_PROMOTED,  # Re-use promotion event for rollback
                        payload={
                            "model_id": config_key,
                            "action": "rollback",
                            "from_version": result.get("from_version"),
                            "to_version": result.get("to_version"),
                            "reason": "auto_regression_critical",
                        },
                        source="UnifiedHealthManager",
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
        config_key = payload.get("config_key", make_config_key(board_type, num_players))
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
                from app.coordination.event_router import publish

                # Emit recovery initiation
                await publish(
                    event_type="recovery_initiated",
                    payload={
                        "error_id": error.error_id,
                        "component": coordinator_name,
                        "node_id": node_id,
                        "strategy": "coordinator_restart",
                        "health_score": health_score,
                    },
                    source="UnifiedHealthManager",
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
        # Jan 2026: Migrated to event_router (app.coordination.data_events deprecated Q2 2026)
        try:
            from app.coordination.event_router import DataEventType, publish_sync
            for node_id in stalled_nodes:
                publish_sync(
                    event_type=DataEventType.NODE_UNHEALTHY,
                    payload={
                        "node_id": node_id,
                        "reason": "cluster_stall",
                        "stall_duration_seconds": stall_duration_seconds,
                    },
                    source="UnifiedHealthManager",
                )
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

    async def _on_daemon_permanently_failed(self, event) -> None:
        """Handle DAEMON_PERMANENTLY_FAILED event - daemon exceeded restart limit (December 2025).

        When a daemon exceeds its hourly restart limit (typically 5 restarts),
        the DaemonManager emits this event. This indicates:
        1. A persistent failure that auto-restart cannot solve
        2. Likely configuration, import, or dependency issue
        3. Manual intervention is required

        This handler:
        1. Records a CRITICAL error
        2. Updates daemon state as permanently failed
        3. Trips circuit breaker for the daemon
        4. Escalates for human intervention
        """
        payload = event.payload if hasattr(event, "payload") else event

        # Support both daemon_name (from emitter) and daemon_type (legacy) field names
        daemon_name = payload.get("daemon_name") or payload.get("daemon_type", "unknown")
        restart_count = payload.get("restart_count", 0)
        error_message = payload.get("error", "")
        hostname = payload.get("hostname", "local")
        daemon_key = f"{daemon_name}@{hostname}"

        logger.critical(
            f"[UnifiedHealthManager] DAEMON_PERMANENTLY_FAILED: {daemon_name} "
            f"(restarts: {restart_count}, host: {hostname})"
        )

        # Get or create daemon state
        if daemon_key not in self._daemon_states:
            self._daemon_states[daemon_key] = DaemonHealthState(
                daemon_name=daemon_name,
                hostname=hostname,
            )

        state = self._daemon_states[daemon_key]
        state.is_running = False
        state.restart_count = restart_count
        state.consecutive_failures = restart_count
        state.last_error = f"permanently_failed: {error_message}"

        # Record as CRITICAL error
        error = ErrorRecord(
            error_id=f"daemon_perm_fail_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component=f"daemon:{daemon_name}",
            error_type="daemon_permanently_failed",
            message=f"Daemon {daemon_name} exceeded restart limit ({restart_count} restarts)",
            node_id=hostname,
            severity=ErrorSeverity.CRITICAL,
            context={
                "daemon_name": daemon_name,
                "restart_count": restart_count,
                "error": error_message,
                "hostname": hostname,
            },
        )
        self._record_error(error)

        # Trip circuit breaker multiple times to ensure it's open
        component_key = f"daemon:{daemon_name}"
        for _ in range(5):  # Multiple failures to ensure breaker trips
            self._on_component_failure(component_key)

        # Escalate for human intervention
        await self._escalate_to_human(
            f"daemon:{daemon_name}",
            f"Daemon permanently failed after {restart_count} restarts: {error_message}",
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
    # DLQ Event Handlers (December 29, 2025)
    # =========================================================================

    async def _on_dlq_stale_events(self, event) -> None:
        """Handle DLQ_STALE_EVENTS event - track stale events in dead letter queue.

        Monitors when events are detected as stale in the DLQ, indicating
        potential issues with event handlers or backpressure in the system.

        December 29, 2025: Added to close the DLQ feedback loop and provide
        visibility into failed event processing across the cluster.
        """
        payload = event.payload if hasattr(event, "payload") else event

        stale_count = payload.get("count", 0)
        oldest_age_hours = payload.get("oldest_age_hours", 0)
        event_types = payload.get("event_types", [])

        logger.warning(
            f"[UnifiedHealthManager] DLQ has {stale_count} stale events "
            f"(oldest: {oldest_age_hours:.1f}h, types: {event_types})"
        )

        # Record as warning-level error for visibility
        error = ErrorRecord(
            error_id=f"dlq_stale_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="dead_letter_queue",
            error_type="stale_events",
            message=f"DLQ has {stale_count} stale events (oldest: {oldest_age_hours:.1f}h)",
            node_id="",
            severity=ErrorSeverity.WARNING,
            context={
                "stale_count": stale_count,
                "oldest_age_hours": oldest_age_hours,
                "event_types": event_types,
            },
        )
        self._record_error(error)

        # Increment DLQ metric
        self._dlq_stale_events_count = getattr(self, "_dlq_stale_events_count", 0) + stale_count

    async def _on_dlq_events_replayed(self, event) -> None:
        """Handle DLQ_EVENTS_REPLAYED event - track successful event replays.

        Monitors when failed events are successfully replayed from the DLQ,
        indicating the system is recovering from transient failures.

        December 29, 2025: Added to track DLQ recovery and system resilience.
        """
        payload = event.payload if hasattr(event, "payload") else event

        replay_count = payload.get("count", 0)
        event_types = payload.get("event_types", [])
        source = payload.get("source", "unknown")

        logger.info(
            f"[UnifiedHealthManager] DLQ replayed {replay_count} events successfully "
            f"(types: {event_types}, source: {source})"
        )

        # Track successful replays for health scoring
        self._dlq_replayed_count = getattr(self, "_dlq_replayed_count", 0) + replay_count

        # Record component success for DLQ circuit breaker
        self._on_component_success("dead_letter_queue")

    async def _on_dlq_events_purged(self, event) -> None:
        """Handle DLQ_EVENTS_PURGED event - track purged (data loss) events.

        Monitors when events are purged from the DLQ (typically due to age
        or being unrecoverable), indicating potential data loss.

        December 29, 2025: Added to track DLQ data loss and alert on
        significant purges that may require investigation.
        """
        payload = event.payload if hasattr(event, "payload") else event

        purge_count = payload.get("count", 0)
        reason = payload.get("reason", "unknown")
        timestamp = payload.get("timestamp", "")

        logger.warning(
            f"[UnifiedHealthManager] DLQ purged {purge_count} events "
            f"(reason: {reason}, at: {timestamp})"
        )

        # Record as error since purge means data loss
        error = ErrorRecord(
            error_id=f"dlq_purge_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="dead_letter_queue",
            error_type="events_purged",
            message=f"DLQ purged {purge_count} events: {reason}",
            node_id="",
            severity=ErrorSeverity.WARNING if purge_count < 100 else ErrorSeverity.ERROR,
            context={
                "purge_count": purge_count,
                "reason": reason,
                "timestamp": timestamp,
            },
        )
        self._record_error(error)

        # Track total purged for metrics
        self._dlq_purged_count = getattr(self, "_dlq_purged_count", 0) + purge_count

    def get_dlq_metrics(self) -> dict[str, int]:
        """Get DLQ-related metrics (December 29, 2025).

        Returns:
            Dict with stale_events, replayed, purged counts
        """
        return {
            "stale_events": getattr(self, "_dlq_stale_events_count", 0),
            "replayed": getattr(self, "_dlq_replayed_count", 0),
            "purged": getattr(self, "_dlq_purged_count", 0),
        }

    # =========================================================================
    # Budget/Capacity Event Handlers (December 29, 2025)
    # =========================================================================

    async def _on_capacity_low(self, event) -> None:
        """Handle CAPACITY_LOW event - track when cluster GPU capacity drops.

        Monitors when GPU capacity falls below threshold, indicating potential
        issues with cluster health or unexpected node terminations.

        December 29, 2025: Added to close the capacity feedback loop and provide
        visibility into cluster capacity issues.
        """
        payload = event.payload if hasattr(event, "payload") else event

        current_gpus = payload.get("current_gpus", 0)
        threshold = payload.get("threshold", 0)
        needed_gpus = payload.get("needed_gpus", 0)

        logger.warning(
            f"[UnifiedHealthManager] Cluster capacity low: {current_gpus} GPUs "
            f"(threshold: {threshold}, need: {needed_gpus})"
        )

        # Record as warning for visibility
        error = ErrorRecord(
            error_id=f"capacity_low_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="cluster_capacity",
            error_type="capacity_low",
            message=f"Cluster capacity low: {current_gpus}/{threshold} GPUs",
            node_id="",
            severity=ErrorSeverity.WARNING,
            context={
                "current_gpus": current_gpus,
                "threshold": threshold,
                "needed_gpus": needed_gpus,
            },
        )
        self._record_error(error)

        # Track capacity state
        self._capacity_low = True
        self._capacity_low_since = time.time()

    async def _on_capacity_restored(self, event) -> None:
        """Handle CAPACITY_RESTORED event - track when capacity recovers.

        Monitors when GPU capacity returns above threshold after being low.

        December 29, 2025: Added to track capacity recovery events.
        """
        payload = event.payload if hasattr(event, "payload") else event

        current_gpus = payload.get("current_gpus", 0)
        threshold = payload.get("threshold", 0)

        # Calculate how long capacity was low
        low_since = getattr(self, "_capacity_low_since", 0)
        duration = time.time() - low_since if low_since else 0

        logger.info(
            f"[UnifiedHealthManager] Cluster capacity restored: {current_gpus} GPUs "
            f"(threshold: {threshold}, was low for {duration:.0f}s)"
        )

        # Record component success
        self._on_component_success("cluster_capacity")

        # Clear capacity state
        self._capacity_low = False
        self._capacity_low_since = 0

    async def _on_budget_exceeded(self, event) -> None:
        """Handle BUDGET_EXCEEDED event - track when spending exceeds budget.

        Monitors when hourly or daily cloud spending exceeds configured limits,
        requiring potential scaling down or alerting.

        December 29, 2025: Added to close the budget feedback loop and provide
        visibility into cost management issues.
        """
        payload = event.payload if hasattr(event, "payload") else event

        period = payload.get("period", "unknown")  # hourly, daily
        spent = payload.get("spent", 0)
        limit = payload.get("limit", 0)
        overage_pct = ((spent / limit) - 1) * 100 if limit > 0 else 0

        logger.error(
            f"[UnifiedHealthManager] Budget exceeded: ${spent:.2f} spent "
            f"(limit: ${limit:.2f}, {period}, {overage_pct:.1f}% over)"
        )

        # Record as error - this is a critical cost issue
        error = ErrorRecord(
            error_id=f"budget_exceeded_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="budget",
            error_type="budget_exceeded",
            message=f"Budget exceeded: ${spent:.2f}/${limit:.2f} ({period})",
            node_id="",
            severity=ErrorSeverity.ERROR,
            context={
                "period": period,
                "spent": spent,
                "limit": limit,
                "overage_pct": overage_pct,
            },
        )
        self._record_error(error)

        # Trip budget circuit breaker
        self._on_component_failure("budget")

        # Track budget state
        self._budget_exceeded = True
        self._budget_exceeded_at = time.time()

    async def _on_budget_alert(self, event) -> None:
        """Handle BUDGET_ALERT event - track approaching budget threshold.

        Monitors when spending approaches budget limits (e.g., 80% of limit),
        providing early warning before exceeding.

        December 29, 2025: Added to provide early warning on budget issues.
        """
        payload = event.payload if hasattr(event, "payload") else event

        period = payload.get("period", "unknown")
        spent = payload.get("spent", 0)
        limit = payload.get("limit", 0)
        threshold_pct = payload.get("threshold_pct", 80)
        current_pct = (spent / limit) * 100 if limit > 0 else 0

        logger.warning(
            f"[UnifiedHealthManager] Budget alert: ${spent:.2f} spent "
            f"(limit: ${limit:.2f}, {period}, {current_pct:.1f}% of limit)"
        )

        # Record as warning
        error = ErrorRecord(
            error_id=f"budget_alert_{int(time.time() * 1000)}",
            timestamp=time.time(),
            component="budget",
            error_type="budget_alert",
            message=f"Budget alert: ${spent:.2f}/${limit:.2f} ({period}, {current_pct:.1f}%)",
            node_id="",
            severity=ErrorSeverity.WARNING,
            context={
                "period": period,
                "spent": spent,
                "limit": limit,
                "threshold_pct": threshold_pct,
                "current_pct": current_pct,
            },
        )
        self._record_error(error)

    def get_budget_capacity_metrics(self) -> dict[str, Any]:
        """Get budget and capacity metrics (December 29, 2025).

        Returns:
            Dict with capacity_low, budget_exceeded, and related timestamps
        """
        return {
            "capacity_low": getattr(self, "_capacity_low", False),
            "capacity_low_since": getattr(self, "_capacity_low_since", 0),
            "budget_exceeded": getattr(self, "_budget_exceeded", False),
            "budget_exceeded_at": getattr(self, "_budget_exceeded_at", 0),
        }

    # =========================================================================
    # Error and Recovery Recording
    # =========================================================================

    def _record_error(self, error: ErrorRecord) -> None:
        """Record an error and update statistics."""
        self._errors.append(error)
        self._errors_by_component[error.component].append(error)
        self._total_errors += 1

        # Trim history
        if len(self._errors) > self._health_config.max_error_history:
            self._errors = self._errors[-self._health_config.max_error_history :]

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
        if len(self._recovery_history) > self._health_config.max_recovery_history:
            self._recovery_history = self._recovery_history[-self._health_config.max_recovery_history :]

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
        cb.record_failure(component)  # Per-target CB requires target arg

        if cb.get_state(component) == CircuitState.OPEN:
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
        was_open = cb.get_state(component) == CircuitState.OPEN

        cb.record_success(component)  # Per-target CB requires target arg

        if was_open and cb.get_state(component) == CircuitState.CLOSED:
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
        return cb.get_state(component) == CircuitState.OPEN

    # =========================================================================
    # Recovery Operations (from RecoveryManager)
    # =========================================================================

    def _can_attempt_node_recovery(self, node_id: str) -> bool:
        """Check if we can attempt recovery on this node."""
        state = self._get_node_state(node_id)

        # Check if already escalated
        if state.is_escalated:
            if time.time() - state.last_escalation_time < self._health_config.escalation_cooldown:
                return False
            state.is_escalated = False

        # Check attempt limit
        if state.recovery_attempts >= self._health_config.max_recovery_attempts_per_node:
            return False

        # Check cooldown
        return time.time() - state.last_attempt_time >= self._health_config.recovery_attempt_cooldown

    def _can_attempt_job_recovery(self, work_id: str) -> bool:
        """Check if we can attempt recovery on this job."""
        state = self._get_job_state(work_id)
        return state.recovery_attempts < self._health_config.max_recovery_attempts_per_job





    # =========================================================================
    # Public API for Error Recording
    # =========================================================================




    # =========================================================================
    # Callback Registration
    # =========================================================================





    # =========================================================================
    # Dependency Setters (legacy compatibility)
    # =========================================================================






    # =========================================================================
    # State Reset
    # =========================================================================



    # =========================================================================
    # Query Methods
    # =========================================================================









    # =========================================================================
    # Statistics
    # =========================================================================



    def health_check(self) -> HealthCheckResult:
        """Check if the health manager is healthy (CoordinatorProtocol compliance).

        Returns HealthCheckResult for consistent health monitoring interface.
        """
        # Check running state (HandlerBase manages this)
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Health manager not running",
            )

        # Check event subscription is active (HandlerBase tracks via _stats)
        if not self._stats.subscribed:
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
            status=CoordinatorStatus.RUNNING,
            message="Health manager running",
            details={"subscribed": self._stats.subscribed, "total_errors": self._total_errors},
        )

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring (sync version)."""
        health_stats = self.get_health_stats()

        return {
            "name": self.name,
            "status": self.status.value,
            "enabled": self._health_config.enabled,
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
            "subscribed": self._stats.subscribed,
        }

    # =========================================================================
    # System Health Scoring (consolidated from system_health_monitor.py)
    # =========================================================================








# =============================================================================
# Singleton and convenience functions
# =============================================================================


def get_health_manager() -> UnifiedHealthManager:
    """Get the global UnifiedHealthManager singleton.

    Uses HandlerBase's singleton management.
    """
    return UnifiedHealthManager.get_instance()


def wire_health_events(
    config: RecoveryConfig | None = None,
) -> UnifiedHealthManager:
    """Wire health events to the manager and start it.

    This resets any existing instance and creates a new one with the
    provided config. The manager is started, which subscribes to events.

    Returns:
        The wired UnifiedHealthManager instance
    """
    # Reset any existing instance
    UnifiedHealthManager.reset_instance()

    # Create new instance with config
    manager = UnifiedHealthManager(config=config)

    # Start via asyncio if loop available, otherwise just get instance
    try:
        asyncio.get_running_loop()
        # We're in async context - schedule start as task
        asyncio.create_task(manager.start())
    except RuntimeError:
        # No running loop - try running synchronously
        try:
            asyncio.run(manager.start())
        except RuntimeError:
            logger.warning("[UnifiedHealthManager] No event loop available for wire_health_events")

    return manager


def reset_health_manager() -> None:
    """Reset the global health manager (for testing).

    Uses HandlerBase's singleton reset.
    """
    UnifiedHealthManager.reset_instance()


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
def get_system_health() -> "UnifiedHealthManager":
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

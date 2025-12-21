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
- Subscribes to NODE_RECOVERED: Update recovery state

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

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkItem

logger = logging.getLogger(__name__)


# =============================================================================
# Enums (consolidated from both modules)
# =============================================================================


class ErrorSeverity(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryStatus(Enum):
    """Recovery attempt status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class RecoveryAction(str, Enum):
    """Types of recovery actions."""

    RESTART_JOB = "restart_job"
    KILL_JOB = "kill_job"
    RESTART_NODE_SERVICES = "restart_node_services"
    REBOOT_NODE = "reboot_node"
    REMOVE_NODE = "remove_node"
    ESCALATE_HUMAN = "escalate_human"
    NONE = "none"


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
class NodeHealthState:
    """Track health state for a node (consolidated)."""

    node_id: str
    is_online: bool = True
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0
    consecutive_failures: int = 0
    is_escalated: bool = False
    last_escalation_time: float = 0.0
    offline_since: float = 0.0


@dataclass
class JobHealthState:
    """Track health state for a job."""

    work_id: str
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    # Stuck job detection
    stuck_job_timeout_multiplier: float = 1.5

    # Recovery attempt limits
    max_recovery_attempts_per_node: int = 3
    max_recovery_attempts_per_job: int = 2
    recovery_attempt_cooldown: int = 300  # 5 min between attempts

    # Escalation thresholds
    consecutive_failures_for_escalation: int = 3
    escalation_cooldown: int = 3600  # 1 hour

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
            from app.distributed.data_events import DataEventType  # Types still needed

            router = get_router()

            # Error events (from ErrorRecoveryCoordinator)
            router.subscribe(DataEventType.ERROR.value, self._on_error)
            router.subscribe(DataEventType.RECOVERY_INITIATED.value, self._on_recovery_initiated)
            router.subscribe(DataEventType.RECOVERY_COMPLETED.value, self._on_recovery_completed)
            router.subscribe(DataEventType.RECOVERY_FAILED.value, self._on_recovery_failed)
            router.subscribe(DataEventType.TRAINING_FAILED.value, self._on_training_failed)
            router.subscribe(DataEventType.TASK_FAILED.value, self._on_task_failed)
            router.subscribe(DataEventType.REGRESSION_DETECTED.value, self._on_regression_detected)

            # Node events (from RecoveryManager)
            router.subscribe(DataEventType.HOST_OFFLINE.value, self._on_host_offline)
            router.subscribe(DataEventType.NODE_RECOVERED.value, self._on_node_recovered)

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


def get_error_coordinator():
    """DEPRECATED: Use get_health_manager() instead.

    Returns the UnifiedHealthManager for backward compatibility.
    """
    warnings.warn(
        "get_error_coordinator() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


def wire_error_events():
    """DEPRECATED: Use wire_health_events() instead.

    Returns the UnifiedHealthManager for backward compatibility.
    """
    warnings.warn(
        "wire_error_events() is deprecated. Use wire_health_events() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return wire_health_events()


def get_recovery_manager():
    """DEPRECATED: Use get_health_manager() instead.

    Returns the UnifiedHealthManager for backward compatibility.
    """
    warnings.warn(
        "get_recovery_manager() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


def wire_recovery_events():
    """DEPRECATED: Use wire_health_events() instead.

    Returns the UnifiedHealthManager for backward compatibility.
    """
    warnings.warn(
        "wire_recovery_events() is deprecated. Use wire_health_events() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return wire_health_events()


__all__ = [
    # Enums
    "ErrorSeverity",
    "RecoveryAction",
    "RecoveryResult",
    "RecoveryStatus",
    # Data classes
    "ErrorRecord",
    "HealthStats",
    "JobHealthState",
    "NodeHealthState",
    "RecoveryAttempt",
    "RecoveryConfig",
    "RecoveryEvent",
    # Main class
    "UnifiedHealthManager",
    # Functions
    "get_health_manager",
    "is_component_healthy",
    "reset_health_manager",
    "wire_health_events",
    # Deprecated (backward compatibility)
    "get_error_coordinator",
    "get_recovery_manager",
    "wire_error_events",
    "wire_recovery_events",
]

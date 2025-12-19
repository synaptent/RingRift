"""UnifiedErrorRecoveryCoordinator - Centralized failure handling (December 2025).

This module provides centralized coordination of error recovery across the cluster.
It tracks failures, coordinates recovery attempts, and provides failure pattern
analysis for proactive issue resolution.

Event Integration:
- Subscribes to ERROR: Track error occurrences
- Subscribes to RECOVERY_INITIATED: Track recovery start
- Subscribes to RECOVERY_COMPLETED: Track successful recoveries
- Subscribes to RECOVERY_FAILED: Track failed recovery attempts
- Subscribes to TRAINING_FAILED: Track training failures
- Subscribes to TASK_FAILED: Track task failures
- Subscribes to REGRESSION_DETECTED: Track model regressions

Key Responsibilities:
1. Track all errors across the cluster
2. Coordinate recovery attempts
3. Detect failure patterns
4. Circuit breaker for failing components
5. Provide error statistics and trends

Usage:
    from app.coordination.error_recovery_coordinator import (
        ErrorRecoveryCoordinator,
        wire_error_events,
        get_error_coordinator,
    )

    # Wire error events
    coordinator = wire_error_events()

    # Check if component should be circuit-broken
    if coordinator.is_circuit_broken("training"):
        print("Training circuit breaker open, waiting...")

    # Get error statistics
    stats = coordinator.get_stats()
    print(f"Total errors: {stats['total_errors']}")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


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
    context: Dict[str, Any] = field(default_factory=dict)
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
class CircuitBreakerState:
    """State for a circuit breaker."""

    component: str
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    opened_at: float = 0.0
    half_open_at: float = 0.0
    success_count_since_open: int = 0


@dataclass
class ErrorStats:
    """Aggregate error statistics."""

    total_errors: int = 0
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    errors_by_component: Dict[str, int] = field(default_factory=dict)
    errors_by_node: Dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_rate: float = 0.0
    circuit_breakers_open: int = 0
    avg_recovery_time: float = 0.0


class ErrorRecoveryCoordinator:
    """Coordinates error recovery across the cluster.

    Tracks errors, coordinates recovery attempts, implements circuit breakers,
    and provides failure pattern analysis.
    """

    def __init__(
        self,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        circuit_breaker_half_open_timeout: float = 30.0,
        max_error_history: int = 500,
        max_recovery_history: int = 200,
    ):
        """Initialize ErrorRecoveryCoordinator.

        Args:
            circuit_breaker_threshold: Failures to open circuit
            circuit_breaker_timeout: Time circuit stays open (seconds)
            circuit_breaker_half_open_timeout: Time in half-open state
            max_error_history: Maximum errors to retain
            max_recovery_history: Maximum recovery attempts to retain
        """
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.circuit_breaker_half_open_timeout = circuit_breaker_half_open_timeout
        self.max_error_history = max_error_history
        self.max_recovery_history = max_recovery_history

        # Error tracking
        self._errors: List[ErrorRecord] = []
        self._errors_by_component: Dict[str, List[ErrorRecord]] = defaultdict(list)
        self._error_id_counter = 0

        # Recovery tracking
        self._active_recoveries: Dict[str, RecoveryAttempt] = {}
        self._recovery_history: List[RecoveryAttempt] = []
        self._recovery_id_counter = 0

        # Circuit breakers by component
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Statistics
        self._total_errors = 0
        self._total_recoveries = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._total_recovery_time = 0.0

        # Callbacks
        self._error_callbacks: List[Callable[[ErrorRecord], None]] = []
        self._recovery_callbacks: List[Callable[[RecoveryAttempt], None]] = []
        self._circuit_breaker_callbacks: List[Callable[[str, bool], None]] = []

        # Subscription state
        self._subscribed = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to error-related events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()

            bus.subscribe(DataEventType.ERROR, self._on_error)
            bus.subscribe(DataEventType.RECOVERY_INITIATED, self._on_recovery_initiated)
            bus.subscribe(DataEventType.RECOVERY_COMPLETED, self._on_recovery_completed)
            bus.subscribe(DataEventType.RECOVERY_FAILED, self._on_recovery_failed)
            bus.subscribe(DataEventType.TRAINING_FAILED, self._on_training_failed)
            bus.subscribe(DataEventType.TASK_FAILED, self._on_task_failed)
            bus.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)

            self._subscribed = True
            logger.info("[ErrorRecoveryCoordinator] Subscribed to error events")
            return True

        except ImportError:
            logger.warning("[ErrorRecoveryCoordinator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[ErrorRecoveryCoordinator] Failed to subscribe: {e}")
            return False

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        self._error_id_counter += 1
        return f"err_{int(time.time())}_{self._error_id_counter}"

    def _generate_recovery_id(self) -> str:
        """Generate a unique recovery ID."""
        self._recovery_id_counter += 1
        return f"rec_{int(time.time())}_{self._recovery_id_counter}"

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
            f"[ErrorRecoveryCoordinator] Recovery initiated: {recovery.recovery_id} "
            f"for {recovery.component} on {recovery.node_id}"
        )

    async def _on_recovery_completed(self, event) -> None:
        """Handle RECOVERY_COMPLETED event."""
        payload = event.payload
        recovery_id = payload.get("recovery_id", "")

        if recovery_id in self._active_recoveries:
            recovery = self._active_recoveries.pop(recovery_id)
        else:
            # Create record for recovery we didn't see start
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
            f"[ErrorRecoveryCoordinator] Recovery completed: {recovery.recovery_id} "
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
            f"[ErrorRecoveryCoordinator] Recovery failed: {recovery.recovery_id} "
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

    def _record_error(self, error: ErrorRecord) -> None:
        """Record an error and update statistics."""
        self._errors.append(error)
        self._errors_by_component[error.component].append(error)
        self._total_errors += 1

        # Trim history
        if len(self._errors) > self.max_error_history:
            self._errors = self._errors[-self.max_error_history:]

        # Notify callbacks
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"[ErrorRecoveryCoordinator] Error callback failed: {e}")

        # Record failure for circuit breaker
        self._on_component_failure(error.component)

        logger.debug(
            f"[ErrorRecoveryCoordinator] Error recorded: {error.error_id} "
            f"({error.component}: {error.error_type})"
        )

    def _record_recovery(self, recovery: RecoveryAttempt) -> None:
        """Record a recovery attempt in history."""
        self._recovery_history.append(recovery)

        # Trim history
        if len(self._recovery_history) > self.max_recovery_history:
            self._recovery_history = self._recovery_history[-self.max_recovery_history:]

        # Notify callbacks
        for callback in self._recovery_callbacks:
            try:
                callback(recovery)
            except Exception as e:
                logger.error(f"[ErrorRecoveryCoordinator] Recovery callback failed: {e}")

    def _on_component_failure(self, component: str) -> None:
        """Record a failure for circuit breaker tracking."""
        if component not in self._circuit_breakers:
            self._circuit_breakers[component] = CircuitBreakerState(component=component)

        cb = self._circuit_breakers[component]
        cb.failure_count += 1
        cb.last_failure_time = time.time()

        # Check if we should open the circuit
        if not cb.is_open and cb.failure_count >= self.circuit_breaker_threshold:
            cb.is_open = True
            cb.opened_at = time.time()

            # Notify callbacks
            for callback in self._circuit_breaker_callbacks:
                try:
                    callback(component, True)
                except Exception as e:
                    logger.error(f"[ErrorRecoveryCoordinator] CB callback failed: {e}")

            logger.warning(
                f"[ErrorRecoveryCoordinator] Circuit breaker OPEN for {component} "
                f"after {cb.failure_count} failures"
            )

    def _on_component_success(self, component: str) -> None:
        """Record a success for circuit breaker tracking."""
        if component not in self._circuit_breakers:
            return

        cb = self._circuit_breakers[component]

        if cb.is_open:
            # In half-open state, success closes the circuit
            elapsed = time.time() - cb.opened_at
            if elapsed >= self.circuit_breaker_timeout:
                cb.success_count_since_open += 1
                if cb.success_count_since_open >= 2:
                    cb.is_open = False
                    cb.failure_count = 0
                    cb.success_count_since_open = 0

                    # Notify callbacks
                    for callback in self._circuit_breaker_callbacks:
                        try:
                            callback(component, False)
                        except Exception as e:
                            logger.error(f"[ErrorRecoveryCoordinator] CB callback failed: {e}")

                    logger.info(
                        f"[ErrorRecoveryCoordinator] Circuit breaker CLOSED for {component}"
                    )
        else:
            # Reset failure count on success
            cb.failure_count = max(0, cb.failure_count - 1)

    def record_error(
        self,
        component: str,
        error_type: str,
        message: str,
        node_id: str = "",
        severity: str = "error",
        context: Optional[Dict] = None,
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

        if not cb.is_open:
            return False

        # Check if timeout has elapsed (half-open state)
        elapsed = time.time() - cb.opened_at
        if elapsed >= self.circuit_breaker_timeout:
            # Allow one request through in half-open state
            return False

        return True

    def on_error(self, callback: Callable[[ErrorRecord], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_recovery(self, callback: Callable[[RecoveryAttempt], None]) -> None:
        """Register callback for recovery completions."""
        self._recovery_callbacks.append(callback)

    def on_circuit_breaker_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for circuit breaker state changes.

        Args:
            callback: Function(component, is_open)
        """
        self._circuit_breaker_callbacks.append(callback)

    def get_recent_errors(self, limit: int = 50) -> List[ErrorRecord]:
        """Get recent errors."""
        return self._errors[-limit:]

    def get_errors_by_component(self, component: str) -> List[ErrorRecord]:
        """Get errors for a specific component."""
        return list(self._errors_by_component.get(component, []))

    def get_recovery_history(self, limit: int = 50) -> List[RecoveryAttempt]:
        """Get recent recovery attempts."""
        return self._recovery_history[-limit:]

    def get_active_recoveries(self) -> List[RecoveryAttempt]:
        """Get active recovery attempts."""
        return list(self._active_recoveries.values())

    def get_circuit_breaker_states(self) -> Dict[str, CircuitBreakerState]:
        """Get all circuit breaker states."""
        return dict(self._circuit_breakers)

    def get_stats(self) -> ErrorStats:
        """Get aggregate error statistics."""
        # Count by severity
        by_severity: Dict[str, int] = defaultdict(int)
        for error in self._errors:
            by_severity[error.severity.value] += 1

        # Count by component
        by_component: Dict[str, int] = {
            comp: len(errors) for comp, errors in self._errors_by_component.items()
        }

        # Count by node
        by_node: Dict[str, int] = defaultdict(int)
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

        # Open circuit breakers
        open_cbs = sum(1 for cb in self._circuit_breakers.values() if cb.is_open)

        return ErrorStats(
            total_errors=self._total_errors,
            errors_by_severity=dict(by_severity),
            errors_by_component=by_component,
            errors_by_node=dict(by_node),
            recovery_attempts=self._total_recoveries,
            successful_recoveries=self._successful_recoveries,
            failed_recoveries=self._failed_recoveries,
            recovery_rate=recovery_rate,
            circuit_breakers_open=open_cbs,
            avg_recovery_time=avg_recovery_time,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring."""
        stats = self.get_stats()

        # Get open circuit breakers
        open_circuits = [
            comp for comp, cb in self._circuit_breakers.items() if cb.is_open
        ]

        return {
            "total_errors": stats.total_errors,
            "errors_by_severity": stats.errors_by_severity,
            "errors_by_component": stats.errors_by_component,
            "recovery_attempts": stats.recovery_attempts,
            "successful_recoveries": stats.successful_recoveries,
            "failed_recoveries": stats.failed_recoveries,
            "recovery_rate": round(stats.recovery_rate * 100, 1),
            "avg_recovery_time": round(stats.avg_recovery_time, 1),
            "active_recoveries": len(self._active_recoveries),
            "circuit_breakers_open": stats.circuit_breakers_open,
            "open_circuits": open_circuits,
            "subscribed": self._subscribed,
        }


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_error_coordinator: Optional[ErrorRecoveryCoordinator] = None


def get_error_coordinator() -> ErrorRecoveryCoordinator:
    """Get the global ErrorRecoveryCoordinator singleton."""
    global _error_coordinator
    if _error_coordinator is None:
        _error_coordinator = ErrorRecoveryCoordinator()
    return _error_coordinator


def wire_error_events() -> ErrorRecoveryCoordinator:
    """Wire error events to the coordinator.

    Returns:
        The wired ErrorRecoveryCoordinator instance
    """
    coordinator = get_error_coordinator()
    coordinator.subscribe_to_events()
    return coordinator


def is_component_healthy(component: str) -> bool:
    """Check if a component is healthy (circuit not broken)."""
    return not get_error_coordinator().is_circuit_broken(component)


def get_error_stats() -> ErrorStats:
    """Convenience function to get error statistics."""
    return get_error_coordinator().get_stats()


__all__ = [
    "ErrorRecoveryCoordinator",
    "ErrorSeverity",
    "RecoveryStatus",
    "ErrorRecord",
    "RecoveryAttempt",
    "CircuitBreakerState",
    "ErrorStats",
    "get_error_coordinator",
    "wire_error_events",
    "is_component_healthy",
    "get_error_stats",
]

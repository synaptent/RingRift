#!/usr/bin/env python3
"""Circuit breaker pattern for fault-tolerant distributed operations.

A circuit breaker prevents cascading failures by:
1. Tracking failures for each target (host, service, etc.)
2. Opening the circuit after too many failures (blocking requests)
3. After a cooldown, allowing a test request to check recovery
4. Closing the circuit if the test succeeds

Usage:
    from app.distributed.circuit_breaker import CircuitBreaker, CircuitState

    # Create a breaker for remote hosts
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60.0,
        half_open_max_calls=1,
    )

    # Check before calling a host
    if breaker.can_execute("host1"):
        try:
            result = call_remote_host("host1")
            breaker.record_success("host1")
        except (ConnectionError, TimeoutError, OSError, RuntimeError):
            breaker.record_failure("host1")
    else:
        print("Circuit open for host1, skipping")

    # Or use as a context manager
    async with breaker.protected("host1"):
        await call_remote_host("host1")

    # Get status
    states = breaker.get_all_states()
    for target, state in states.items():
        print(f"{target}: {state.state} (failures: {state.failure_count})")
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, TypeVar

__all__ = [
    "CircuitBreaker",
    # Registry and singletons
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    # Core classes
    "CircuitState",
    "CircuitStatus",
    "FallbackChain",
    # Utilities
    "format_circuit_status",
    "get_adaptive_timeout",
    "get_circuit_registry",
    "get_host_breaker",
    "get_operation_breaker",
    "get_training_breaker",
    "set_host_breaker_callback",
    "with_circuit_breaker",
]

T = TypeVar("T")

# Import centralized circuit breaker configurations (December 2025)
try:
    from app.config.coordination_defaults import (
        CircuitBreakerDefaults,
        get_circuit_breaker_configs,
    )
    CIRCUIT_BREAKER_CONFIGS = get_circuit_breaker_configs()
    DEFAULT_FAILURE_THRESHOLD = CircuitBreakerDefaults.FAILURE_THRESHOLD
    DEFAULT_RECOVERY_TIMEOUT = CircuitBreakerDefaults.RECOVERY_TIMEOUT
    DEFAULT_MAX_BACKOFF = CircuitBreakerDefaults.MAX_BACKOFF
    DEFAULT_HALF_OPEN_MAX_CALLS = CircuitBreakerDefaults.HALF_OPEN_MAX_CALLS
except ImportError:
    # Fallback defaults if central config not available
    CIRCUIT_BREAKER_CONFIGS = {
        "ssh": {"failure_threshold": 3, "recovery_timeout": 60.0},
        "http": {"failure_threshold": 5, "recovery_timeout": 30.0},
        "p2p": {"failure_threshold": 3, "recovery_timeout": 45.0},
        "aria2": {"failure_threshold": 2, "recovery_timeout": 120.0},
        "rsync": {"failure_threshold": 2, "recovery_timeout": 90.0},
    }
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RECOVERY_TIMEOUT = 60.0
    # Dec 28, 2025: Reduced from 600 to 180 to prevent long stalls in training pipelines
    DEFAULT_MAX_BACKOFF = 180.0
    DEFAULT_HALF_OPEN_MAX_CALLS = 1

# ============================================
# Prometheus Metrics (optional)
# ============================================

try:
    from prometheus_client import REGISTRY, Counter, Gauge
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    REGISTRY = None

# Helper to safely create metrics (handle re-registration on module reload)
def _get_or_create_gauge(name: str, description: str, labels: list):
    """Get existing gauge or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        return Gauge(name, description, labels)
    except ValueError:
        # Already registered, get from registry
        return REGISTRY._names_to_collectors.get(name)

def _get_or_create_counter(name: str, description: str, labels: list):
    """Get existing counter or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        return Counter(name, description, labels)
    except ValueError:
        # Already registered, get from registry
        return REGISTRY._names_to_collectors.get(name)

if HAS_PROMETHEUS:
    PROM_CIRCUIT_STATE = _get_or_create_gauge(
        'ringrift_circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half_open)',
        ['operation_type', 'target']
    )
    PROM_CIRCUIT_FAILURES = _get_or_create_counter(
        'ringrift_circuit_breaker_failures_total',
        'Total circuit breaker failures recorded',
        ['operation_type', 'target']
    )
    PROM_CIRCUIT_SUCCESSES = _get_or_create_counter(
        'ringrift_circuit_breaker_successes_total',
        'Total circuit breaker successes recorded',
        ['operation_type', 'target']
    )
    PROM_CIRCUIT_OPENS = _get_or_create_counter(
        'ringrift_circuit_breaker_opens_total',
        'Total circuit breaker open events',
        ['operation_type', 'target']
    )
    PROM_CIRCUIT_BLOCKED_REQUESTS = _get_or_create_counter(
        'ringrift_circuit_breaker_blocked_total',
        'Total requests blocked by open circuit',
        ['operation_type', 'target']
    )


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation - requests allowed
    OPEN = "open"          # Failures exceeded - requests blocked
    HALF_OPEN = "half_open"  # Testing recovery - limited requests


@dataclass
class CircuitStatus:
    """Status of a circuit for a specific target."""
    target: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    opened_at: float | None
    half_open_at: float | None
    consecutive_opens: int = 0  # For exponential backoff tracking

    @property
    def time_since_open(self) -> float | None:
        """Seconds since circuit opened."""
        if self.opened_at:
            return time.time() - self.opened_at
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "opened_at": self.opened_at,
            "half_open_at": self.half_open_at,
            "time_since_open": self.time_since_open,
            "consecutive_opens": self.consecutive_opens,
        }


@dataclass
class _CircuitData:
    """Internal circuit data per target."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    opened_at: float | None = None
    half_open_at: float | None = None
    half_open_calls: int = 0
    # Exponential backoff tracking
    consecutive_opens: int = 0  # How many times circuit opened without full recovery


class CircuitBreaker:
    """Circuit breaker for fault-tolerant distributed operations.

    This implementation supports:
    - Per-target circuit tracking (e.g., different hosts)
    - Configurable failure thresholds and recovery timeouts
    - Half-open state for gradual recovery testing
    - Exponential backoff with jitter for repeated failures
    - Thread-safe operation
    - Async context manager support

    Args:
        failure_threshold: Number of consecutive failures to open circuit
        recovery_timeout: Base seconds to wait before testing recovery (half-open)
        half_open_max_calls: Max test calls in half-open state
        success_threshold: Successes needed in half-open to close circuit
        backoff_multiplier: Multiplier for exponential backoff (default 2.0)
        max_backoff: Maximum recovery timeout in seconds (default 600 = 10 min)
        jitter_factor: Random jitter factor 0-1 (default 0.1 = 10%)
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
        half_open_max_calls: int = DEFAULT_HALF_OPEN_MAX_CALLS,
        success_threshold: int = 1,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
        operation_type: str = "default",  # For Prometheus metrics labeling
        backoff_multiplier: float = 2.0,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        jitter_factor: float = 0.1,
        # Active recovery parameters (December 2025)
        active_recovery_probe: Callable[[str], bool] | None = None,
        max_consecutive_opens: int = 5,  # After this many, require manual reset
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self._on_state_change = on_state_change
        self.operation_type = operation_type
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff
        self.jitter_factor = jitter_factor
        self._active_recovery_probe = active_recovery_probe
        self.max_consecutive_opens = max_consecutive_opens

        self._circuits: dict[str, _CircuitData] = {}
        self._lock = RLock()

    def _notify_state_change(self, target: str, old_state: CircuitState, new_state: CircuitState) -> None:
        """Notify callback of a state change."""
        if self._on_state_change and old_state != new_state:
            try:
                self._on_state_change(target, old_state, new_state)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                # Catch callback errors - don't let user callback bugs affect circuit operation
                pass

    def _get_or_create_circuit(self, target: str) -> _CircuitData:
        """Get or create circuit data for a target."""
        if target not in self._circuits:
            self._circuits[target] = _CircuitData()
        return self._circuits[target]

    def _compute_backoff_timeout(self, circuit: _CircuitData) -> float:
        """Compute recovery timeout with exponential backoff and jitter.

        Returns:
            Recovery timeout in seconds, capped at max_backoff.
        """
        # Exponential backoff: base * (multiplier ^ consecutive_opens)
        backoff = self.recovery_timeout * (
            self.backoff_multiplier ** circuit.consecutive_opens
        )
        # Cap at max_backoff
        backoff = min(backoff, self.max_backoff)
        # Add jitter: random value in range [-jitter_factor, +jitter_factor] * backoff
        if self.jitter_factor > 0:
            jitter = backoff * self.jitter_factor * (2 * random.random() - 1)
            backoff = max(0.1, backoff + jitter)  # Ensure positive
        return backoff

    def _check_recovery(self, circuit: _CircuitData) -> None:
        """Check if circuit should transition to half-open."""
        if circuit.state == CircuitState.OPEN:
            # Use exponential backoff timeout based on consecutive opens
            timeout = self._compute_backoff_timeout(circuit)
            if circuit.opened_at and (time.time() - circuit.opened_at) >= timeout:
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_at = time.time()
                circuit.half_open_calls = 0

    def can_execute(self, target: str) -> bool:
        """Check if a request to target is allowed.

        Returns True if:
        - Circuit is CLOSED (normal operation)
        - Circuit is HALF_OPEN and under max test calls

        Returns False if:
        - Circuit is OPEN (blocking requests)
        - Circuit is HALF_OPEN and max test calls reached
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            self._check_recovery(circuit)

            if circuit.state == CircuitState.CLOSED:
                return True
            elif circuit.state == CircuitState.HALF_OPEN:
                if circuit.half_open_calls < self.half_open_max_calls:
                    circuit.half_open_calls += 1
                    return True
                # Blocked in half-open
                if HAS_PROMETHEUS:
                    PROM_CIRCUIT_BLOCKED_REQUESTS.labels(
                        operation_type=self.operation_type, target=target
                    ).inc()
                return False
            else:  # OPEN
                # Record blocked request
                if HAS_PROMETHEUS:
                    PROM_CIRCUIT_BLOCKED_REQUESTS.labels(
                        operation_type=self.operation_type, target=target
                    ).inc()
                return False

    def record_success(self, target: str) -> None:
        """Record a successful operation for target."""
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.success_count += 1
            circuit.last_success_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Check if we've had enough successes to close
                if circuit.success_count >= self.success_threshold:
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    circuit.opened_at = None
                    circuit.half_open_at = None
                    # Reset backoff on successful recovery
                    circuit.consecutive_opens = 0
            elif circuit.state == CircuitState.CLOSED:
                # Reset failure count on success
                circuit.failure_count = 0

            new_state = circuit.state

        # Emit Prometheus metrics (wrapped in try/except to not crash training)
        if HAS_PROMETHEUS:
            try:
                PROM_CIRCUIT_SUCCESSES.labels(
                    operation_type=self.operation_type, target=target
                ).inc()
                # Update state gauge
                state_value = {"closed": 0, "open": 1, "half_open": 2}.get(new_state.value, 0)
                PROM_CIRCUIT_STATE.labels(
                    operation_type=self.operation_type, target=target
                ).set(state_value)
            except (ValueError, TypeError):
                pass  # Label mismatch - metric was registered with different labels

        # Notify state change outside lock
        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def record_failure(self, target: str, error: Exception | None = None) -> None:
        """Record a failed operation for target."""
        old_state = None
        new_state = None
        opened_circuit = False

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.failure_count += 1
            circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Failure in half-open: go back to open with increased backoff
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.time()
                circuit.success_count = 0
                circuit.consecutive_opens += 1  # Increase backoff for next recovery
                opened_circuit = True
            elif circuit.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if circuit.failure_count >= self.failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = time.time()
                    circuit.consecutive_opens += 1  # Track consecutive opens
                    opened_circuit = True

            new_state = circuit.state

        # Emit Prometheus metrics (wrapped in try/except to not crash training)
        if HAS_PROMETHEUS:
            try:
                PROM_CIRCUIT_FAILURES.labels(
                    operation_type=self.operation_type, target=target
                ).inc()
                if opened_circuit:
                    PROM_CIRCUIT_OPENS.labels(
                        operation_type=self.operation_type, target=target
                    ).inc()
                # Update state gauge
                state_value = {"closed": 0, "open": 1, "half_open": 2}.get(new_state.value, 0)
                PROM_CIRCUIT_STATE.labels(
                    operation_type=self.operation_type, target=target
                ).set(state_value)
            except (ValueError, TypeError):
                pass  # Label mismatch - metric was registered with different labels

        # Notify state change outside lock
        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def try_active_recovery(self, target: str) -> bool:
        """Actively probe target for recovery instead of waiting for timeout.

        Uses the configured active_recovery_probe callback to check if the
        target is healthy. If probe succeeds, transitions to HALF_OPEN or
        directly to CLOSED.

        Returns:
            True if recovery probe succeeded, False otherwise.
        """
        if not self._active_recovery_probe:
            return False

        with self._lock:
            circuit = self._get_or_create_circuit(target)

            # Only try recovery on OPEN circuits
            if circuit.state != CircuitState.OPEN:
                return circuit.state == CircuitState.CLOSED

            # Check if we've exceeded max consecutive opens
            if circuit.consecutive_opens >= self.max_consecutive_opens:
                # Requires manual reset
                return False

        # Run probe outside lock
        try:
            probe_success = self._active_recovery_probe(target)
        except (ConnectionError, TimeoutError, OSError, RuntimeError):
            # Network/system failures during health probe - treat as probe failure
            probe_success = False

        if probe_success:
            with self._lock:
                old_state = circuit.state
                # Transition to half-open for gradual recovery
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_at = time.time()
                circuit.half_open_calls = 0
                self._notify_state_change(target, old_state, CircuitState.HALF_OPEN)
            return True

        return False

    def force_reset(self, target: str) -> None:
        """Force reset a circuit to CLOSED state.

        Use this for manual intervention when a circuit is stuck open
        (e.g., after max_consecutive_opens exceeded).

        This resets all counters including consecutive_opens.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state

            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.success_count = 0
            circuit.consecutive_opens = 0
            circuit.opened_at = None
            circuit.half_open_at = None
            circuit.half_open_calls = 0

            self._notify_state_change(target, old_state, CircuitState.CLOSED)

    def is_permanently_open(self, target: str) -> bool:
        """Check if circuit has exceeded max consecutive opens.

        When a circuit exceeds max_consecutive_opens, it won't auto-recover
        and requires force_reset() for manual intervention.

        Returns:
            True if circuit is open and at max consecutive opens.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            return (
                circuit.state == CircuitState.OPEN
                and circuit.consecutive_opens >= self.max_consecutive_opens
            )

    def get_permanently_open_circuits(self) -> list[str]:
        """Get all targets with permanently open circuits.

        Returns:
            List of target names that require manual force_reset().
        """
        result = []
        with self._lock:
            for target, circuit in self._circuits.items():
                if (
                    circuit.state == CircuitState.OPEN
                    and circuit.consecutive_opens >= self.max_consecutive_opens
                ):
                    result.append(target)
        return result

    def get_state(self, target: str) -> CircuitState:
        """Get current state for a target."""
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            self._check_recovery(circuit)
            return circuit.state

    def get_status(self, target: str) -> CircuitStatus:
        """Get detailed status for a target."""
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            self._check_recovery(circuit)
            return CircuitStatus(
                target=target,
                state=circuit.state,
                failure_count=circuit.failure_count,
                success_count=circuit.success_count,
                last_failure_time=circuit.last_failure_time,
                last_success_time=circuit.last_success_time,
                opened_at=circuit.opened_at,
                half_open_at=circuit.half_open_at,
                consecutive_opens=circuit.consecutive_opens,
            )

    def get_all_states(self) -> dict[str, CircuitStatus]:
        """Get status for all tracked targets."""
        with self._lock:
            return {
                target: CircuitStatus(
                    target=target,
                    state=circuit.state,
                    failure_count=circuit.failure_count,
                    success_count=circuit.success_count,
                    last_failure_time=circuit.last_failure_time,
                    last_success_time=circuit.last_success_time,
                    opened_at=circuit.opened_at,
                    half_open_at=circuit.half_open_at,
                    consecutive_opens=circuit.consecutive_opens,
                )
                for target, circuit in self._circuits.items()
            }

    def reset(self, target: str) -> None:
        """Reset circuit for a target to CLOSED state."""
        with self._lock:
            if target in self._circuits:
                self._circuits[target] = _CircuitData()

    def reset_all(self) -> None:
        """Reset all circuits to CLOSED state."""
        with self._lock:
            self._circuits.clear()

    def force_open(self, target: str) -> None:
        """Force circuit open for a target (for testing/manual intervention)."""
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            circuit.state = CircuitState.OPEN
            circuit.opened_at = time.time()

    def force_close(self, target: str) -> None:
        """Force circuit closed for a target."""
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.opened_at = None
            circuit.half_open_at = None

    @contextmanager
    def protected_sync(self, target: str):
        """Synchronous context manager for protected execution.

        Usage:
            with breaker.protected_sync("host1"):
                call_host("host1")

        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self.can_execute(target):
            raise CircuitOpenError(f"Circuit open for {target}")

        try:
            yield
            self.record_success(target)
        except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as e:
            # Network/system failures, runtime errors - record as circuit failure
            self.record_failure(target, e)
            raise

    @asynccontextmanager
    async def protected(self, target: str):
        """Async context manager for protected execution.

        Usage:
            async with breaker.protected("host1"):
                await call_host("host1")

        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self.can_execute(target):
            raise CircuitOpenError(f"Circuit open for {target}")

        try:
            yield
            self.record_success(target)
        except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError, asyncio.CancelledError, RuntimeError, ValueError) as e:
            # Network/async failures, runtime errors - record as circuit failure
            self.record_failure(target, e)
            raise

    def execute(
        self,
        target: str,
        func: Callable[[], T],
        fallback: Callable[[], T] | None = None,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            target: Target identifier
            func: Function to execute
            fallback: Optional fallback function if circuit is open

        Returns:
            Result from func or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        if not self.can_execute(target):
            if fallback:
                return fallback()
            raise CircuitOpenError(f"Circuit open for {target}")

        try:
            result = func()
            self.record_success(target)
            return result
        except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as e:
            # Network/system failures, runtime errors - record as circuit failure
            self.record_failure(target, e)
            raise

    async def execute_async(
        self,
        target: str,
        func: Callable[[], T],
        fallback: Callable[[], T] | None = None,
    ) -> T:
        """Execute async function with circuit breaker protection."""
        if not self.can_execute(target):
            if fallback:
                return await fallback() if asyncio.iscoroutinefunction(fallback) else fallback()
            raise CircuitOpenError(f"Circuit open for {target}")

        try:
            result = await func() if asyncio.iscoroutinefunction(func) else func()
            self.record_success(target)
            return result
        except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError, asyncio.CancelledError, RuntimeError, ValueError) as e:
            # Network/async failures, runtime errors - record as circuit failure
            self.record_failure(target, e)
            raise


class CircuitOpenError(Exception):
    """Raised when attempting to execute through an open circuit."""


# Global circuit breaker instance for hosts
_host_breaker: CircuitBreaker | None = None
_host_breaker_callback: Callable[[str, CircuitState, CircuitState], None] | None = None


def set_host_breaker_callback(
    callback: Callable[[str, CircuitState, CircuitState], None]
) -> None:
    """Set a callback for host circuit breaker state changes.

    The callback is called with (target, old_state, new_state) whenever
    a circuit transitions between states. This allows other components
    (like the event bus) to react to circuit state changes.

    Must be called before get_host_breaker() is first invoked, or
    the callback will not be registered.
    """
    global _host_breaker_callback
    _host_breaker_callback = callback

    # If breaker already exists, update its callback
    if _host_breaker is not None:
        _host_breaker._on_state_change = callback


def get_host_breaker() -> CircuitBreaker:
    """Get the global host circuit breaker instance."""
    global _host_breaker
    if _host_breaker is None:
        _host_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=120.0,  # 2 minutes
            half_open_max_calls=1,
            success_threshold=1,
            on_state_change=_host_breaker_callback,
        )
    return _host_breaker


# Global circuit breaker for training operations
_training_breaker: CircuitBreaker | None = None


def get_training_breaker() -> CircuitBreaker:
    """Get the global training circuit breaker instance."""
    global _training_breaker
    if _training_breaker is None:
        _training_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=600.0,  # 10 minutes
            half_open_max_calls=1,
            success_threshold=1,
        )
    return _training_breaker


def format_circuit_status(status: CircuitStatus) -> str:
    """Format circuit status as human-readable string."""
    state_icon = {
        CircuitState.CLOSED: "✓",
        CircuitState.OPEN: "✗",
        CircuitState.HALF_OPEN: "◐",
    }.get(status.state, "?")

    parts = [f"{state_icon} {status.target}: {status.state.value}"]

    if status.failure_count > 0:
        parts.append(f"failures={status.failure_count}")

    if status.time_since_open:
        parts.append(f"open_for={status.time_since_open:.0f}s")

    if status.consecutive_opens > 0:
        parts.append(f"backoff_level={status.consecutive_opens}")

    return " ".join(parts)


# =============================================================================
# Operation-Type Registry for Fine-Grained Circuit Control
# =============================================================================

class CircuitBreakerRegistry:
    """Registry for operation-type specific circuit breakers.

    Provides separate circuit breakers for different operation types
    (SSH, HTTP, P2P, aria2) with appropriate configurations.

    Usage:
        registry = get_circuit_registry()
        breaker = registry.get_breaker("runpod-h100", "ssh")

        if breaker.can_execute("runpod-h100"):
            ...
    """

    _instance: CircuitBreakerRegistry | None = None
    _lock = RLock()

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        # Use centralized config from app/config/thresholds.py (December 2025)
        self._configs = CIRCUIT_BREAKER_CONFIGS.copy()

    @classmethod
    def get_instance(cls) -> CircuitBreakerRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_breaker(self, operation_type: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation type."""
        with self._lock:
            if operation_type not in self._breakers:
                config = self._configs.get(operation_type, {})
                self._breakers[operation_type] = CircuitBreaker(
                    failure_threshold=config.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD),
                    recovery_timeout=config.get("recovery_timeout", DEFAULT_RECOVERY_TIMEOUT),
                    half_open_max_calls=DEFAULT_HALF_OPEN_MAX_CALLS,
                    success_threshold=1,
                )
            return self._breakers[operation_type]

    def get_timeout(self, operation_type: str, host: str, default: float) -> float:
        """Get appropriate timeout based on circuit state.

        Returns shorter timeout when in HALF_OPEN for faster probing.
        """
        breaker = self.get_breaker(operation_type)
        state = breaker.get_state(host)

        if state == CircuitState.HALF_OPEN:
            # Use shorter timeout for probing
            return min(default * 0.3, 15.0)
        return default

    def get_all_open_circuits(self) -> dict[str, dict[str, CircuitStatus]]:
        """Get all circuits that are currently OPEN or HALF_OPEN."""
        result = {}
        with self._lock:
            for op_type, breaker in self._breakers.items():
                states = breaker.get_all_states()
                open_states = {
                    target: status
                    for target, status in states.items()
                    if status.state != CircuitState.CLOSED
                }
                if open_states:
                    result[op_type] = open_states
        return result


_circuit_registry: CircuitBreakerRegistry | None = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _circuit_registry
    if _circuit_registry is None:
        _circuit_registry = CircuitBreakerRegistry.get_instance()
    return _circuit_registry


def get_operation_breaker(operation_type: str) -> CircuitBreaker:
    """Get circuit breaker for a specific operation type."""
    return get_circuit_registry().get_breaker(operation_type)


def get_adaptive_timeout(operation_type: str, host: str, default: float) -> float:
    """Get timeout adjusted for circuit state."""
    return get_circuit_registry().get_timeout(operation_type, host, default)


# =============================================================================
# Fallback Chain with Timeout Budget
# =============================================================================

class FallbackChain:
    """Coordinated fallback chain with timeout budget management.

    Ensures fallback operations don't exceed total timeout budget,
    and skips operations whose circuits are open.

    Usage:
        chain = FallbackChain(total_timeout=300.0)
        chain.add_operation("ssh", ssh_sync, timeout=120.0)
        chain.add_operation("p2p", p2p_sync, timeout=90.0)
        chain.add_operation("aria2", aria2_sync, timeout=90.0)

        result = await chain.execute(host="runpod-h100")
    """

    def __init__(self, total_timeout: float = 300.0):
        self.total_timeout = total_timeout
        self._operations: list = []
        self._start_time: float | None = None

    def add_operation(
        self,
        operation_type: str,
        func: Callable,
        timeout: float,
        name: str | None = None,
    ) -> FallbackChain:
        """Add an operation to the fallback chain."""
        self._operations.append({
            "operation_type": operation_type,
            "func": func,
            "timeout": timeout,
            "name": name or operation_type,
        })
        return self

    @property
    def remaining_budget(self) -> float:
        """Get remaining timeout budget."""
        if self._start_time is None:
            return self.total_timeout
        elapsed = time.time() - self._start_time
        return max(0, self.total_timeout - elapsed)

    async def execute(self, host: str, **kwargs) -> Any:
        """Execute the fallback chain.

        Returns the result from the first successful operation.
        Raises the last exception if all operations fail.
        """
        self._start_time = time.time()
        last_error: Exception | None = None
        registry = get_circuit_registry()

        for op in self._operations:
            op_type = op["operation_type"]
            func = op["func"]
            base_timeout = op["timeout"]
            # name = op["name"]  # Available but not currently used

            # Check remaining budget
            remaining = self.remaining_budget
            if remaining <= 0:
                break

            # Check circuit state
            breaker = registry.get_breaker(op_type)
            if not breaker.can_execute(host):
                continue  # Skip this operation

            # Calculate effective timeout
            timeout = min(
                base_timeout,
                remaining,
                registry.get_timeout(op_type, host, base_timeout),
            )

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(host=host, timeout=timeout, **kwargs),
                        timeout=timeout + 5,  # Small buffer for cleanup
                    )
                else:
                    result = func(host=host, timeout=timeout, **kwargs)

                breaker.record_success(host)
                return result

            except asyncio.TimeoutError as e:
                breaker.record_failure(host, e)
                last_error = e
            except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as e:
                # Network/system failures during fallback operation
                breaker.record_failure(host, e)
                last_error = e

        if last_error:
            raise last_error
        raise CircuitOpenError(f"All circuits open for {host}")


# =============================================================================
# Circuit Breaker Decorator (December 2025)
# =============================================================================

def with_circuit_breaker(
    operation_type: str,
    host_param: str = "host",
):
    """Decorator to wrap functions with circuit breaker protection.

    Automatically records success/failure and blocks calls when circuit is open.

    Args:
        operation_type: Circuit breaker type (e.g., "ssh", "http", "p2p")
        host_param: Name of the host parameter in the function signature

    Usage:
        @with_circuit_breaker("ssh", host_param="hostname")
        async def ssh_execute(hostname: str, cmd: str):
            # Implementation
            pass

        @with_circuit_breaker("http")
        async def http_request(host: str, url: str):
            # Implementation
            pass

    The decorator will:
    1. Check if circuit is open before calling
    2. Raise CircuitOpenError if circuit is open
    3. Record success on normal return
    4. Record failure on exception
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get host from kwargs or positional args
            host = kwargs.get(host_param)
            if host is None and args:
                # Try to get from first positional arg
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if host_param in params:
                    idx = params.index(host_param)
                    if idx < len(args):
                        host = args[idx]

            host = host or "default"
            breaker = get_operation_breaker(operation_type)

            if not breaker.can_execute(host):
                raise CircuitOpenError(
                    f"Circuit '{operation_type}' is open for host '{host}'"
                )

            try:
                result = await func(*args, **kwargs)
                breaker.record_success(host)
                return result
            except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError, asyncio.CancelledError, RuntimeError, ValueError):
                # Network/async failures - record and propagate
                breaker.record_failure(host)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            host = kwargs.get(host_param)
            if host is None and args:
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if host_param in params:
                    idx = params.index(host_param)
                    if idx < len(args):
                        host = args[idx]

            host = host or "default"
            breaker = get_operation_breaker(operation_type)

            if not breaker.can_execute(host):
                raise CircuitOpenError(
                    f"Circuit '{operation_type}' is open for host '{host}'"
                )

            try:
                result = func(*args, **kwargs)
                breaker.record_success(host)
                return result
            except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError):
                # Network/system failures - record and propagate
                breaker.record_failure(host)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


if __name__ == "__main__":
    # Demo
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    print("=== Circuit Breaker Demo ===\n")

    # Simulate failures
    print("Simulating 3 failures...")
    for i in range(3):
        if breaker.can_execute("host1"):
            breaker.record_failure("host1")
            status = breaker.get_status("host1")
            print(f"  After failure {i+1}: {format_circuit_status(status)}")

    # Check blocked
    print(f"\nCan execute now? {breaker.can_execute('host1')}")

    # Wait for recovery
    print("\nWaiting 6 seconds for recovery timeout...")
    time.sleep(6)

    print("\nAfter timeout:")
    status = breaker.get_status("host1")
    print(f"  {format_circuit_status(status)}")
    print(f"  Can execute? {breaker.can_execute('host1')}")

    # Successful recovery
    if breaker.can_execute("host1"):
        breaker.record_success("host1")
        status = breaker.get_status("host1")
        print(f"\nAfter success: {format_circuit_status(status)}")

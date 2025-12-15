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
        except Exception:
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
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


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
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    opened_at: Optional[float]
    half_open_at: Optional[float]

    @property
    def time_since_open(self) -> Optional[float]:
        """Seconds since circuit opened."""
        if self.opened_at:
            return time.time() - self.opened_at
        return None

    def to_dict(self) -> Dict[str, Any]:
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
        }


@dataclass
class _CircuitData:
    """Internal circuit data per target."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    half_open_at: Optional[float] = None
    half_open_calls: int = 0


class CircuitBreaker:
    """Circuit breaker for fault-tolerant distributed operations.

    This implementation supports:
    - Per-target circuit tracking (e.g., different hosts)
    - Configurable failure thresholds and recovery timeouts
    - Half-open state for gradual recovery testing
    - Thread-safe operation
    - Async context manager support

    Args:
        failure_threshold: Number of consecutive failures to open circuit
        recovery_timeout: Seconds to wait before testing recovery (half-open)
        half_open_max_calls: Max test calls in half-open state
        success_threshold: Successes needed in half-open to close circuit
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 1,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self._on_state_change = on_state_change

        self._circuits: Dict[str, _CircuitData] = {}
        self._lock = RLock()

    def _notify_state_change(self, target: str, old_state: CircuitState, new_state: CircuitState) -> None:
        """Notify callback of a state change."""
        if self._on_state_change and old_state != new_state:
            try:
                self._on_state_change(target, old_state, new_state)
            except Exception:
                pass  # Don't let callback errors affect circuit operation

    def _get_or_create_circuit(self, target: str) -> _CircuitData:
        """Get or create circuit data for a target."""
        if target not in self._circuits:
            self._circuits[target] = _CircuitData()
        return self._circuits[target]

    def _check_recovery(self, circuit: _CircuitData) -> None:
        """Check if circuit should transition to half-open."""
        if circuit.state == CircuitState.OPEN:
            if circuit.opened_at and (time.time() - circuit.opened_at) >= self.recovery_timeout:
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
                return False
            else:  # OPEN
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
            elif circuit.state == CircuitState.CLOSED:
                # Reset failure count on success
                circuit.failure_count = 0

            new_state = circuit.state

        # Notify state change outside lock
        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def record_failure(self, target: str, error: Optional[Exception] = None) -> None:
        """Record a failed operation for target."""
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.failure_count += 1
            circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Failure in half-open: go back to open
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.time()
                circuit.success_count = 0
            elif circuit.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if circuit.failure_count >= self.failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = time.time()

            new_state = circuit.state

        # Notify state change outside lock
        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

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
            )

    def get_all_states(self) -> Dict[str, CircuitStatus]:
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
        except Exception as e:
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
        except Exception as e:
            self.record_failure(target, e)
            raise

    def execute(
        self,
        target: str,
        func: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
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
        except Exception as e:
            self.record_failure(target, e)
            raise

    async def execute_async(
        self,
        target: str,
        func: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
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
        except Exception as e:
            self.record_failure(target, e)
            raise


class CircuitOpenError(Exception):
    """Raised when attempting to execute through an open circuit."""
    pass


# Global circuit breaker instance for hosts
_host_breaker: Optional[CircuitBreaker] = None
_host_breaker_callback: Optional[Callable[[str, CircuitState, CircuitState], None]] = None


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
_training_breaker: Optional[CircuitBreaker] = None


def get_training_breaker() -> CircuitBreaker:
    """Get the global training circuit breaker instance."""
    global _training_breaker
    if _training_breaker is None:
        _training_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=300.0,  # 5 minutes
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

    return " ".join(parts)


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

    print(f"\nAfter timeout:")
    status = breaker.get_status("host1")
    print(f"  {format_circuit_status(status)}")
    print(f"  Can execute? {breaker.can_execute('host1')}")

    # Successful recovery
    if breaker.can_execute("host1"):
        breaker.record_success("host1")
        status = breaker.get_status("host1")
        print(f"\nAfter success: {format_circuit_status(status)}")

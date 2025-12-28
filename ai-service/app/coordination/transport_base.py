#!/usr/bin/env python3
"""Transport Base Class for Unified Transport Operations.

Provides a unified base class for all cluster transport operations,
consolidating common patterns:
- Circuit breaker management (can_attempt, record_success, record_failure)
- Timeout handling (execute_with_timeout)
- State persistence (_load_state, _save_state)
- Health check interface

Usage:
    from app.coordination.transport_base import TransportBase, TransportResult

    class MyTransport(TransportBase):
        async def transfer(self, source, destination, target):
            if not self.can_attempt(target):
                return TransportResult(success=False, error="Circuit open")

            try:
                result = await self.execute_with_timeout(
                    self._do_transfer(source, destination),
                    timeout=self.operation_timeout,
                )
                self.record_success(target)
                return result
            except Exception as e:
                self.record_failure(target, e)
                return TransportResult(success=False, error=str(e))

December 2025: Consolidates patterns from:
- cluster_transport.py (794 LOC)
- sync_base.py (438 LOC)
- handler_resilience.py (448 LOC)
- database_sync_manager.py (669 LOC)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

# Import HealthCheckResult for runtime use (not just type hints)
from app.coordination.contracts import HealthCheckResult

logger = logging.getLogger(__name__)

# Type variable for generic transfer result
T = TypeVar("T")


# =============================================================================
# Data Classes
# =============================================================================


class TransportState(Enum):
    """State of a transport target's circuit breaker."""

    CLOSED = "closed"  # Operating normally
    OPEN = "open"  # Blocked due to failures
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class TransportResult:
    """Result of a transport operation.

    Unified result type for all transport operations to provide
    consistent return values and error handling.
    """

    success: bool
    transport_used: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    bytes_transferred: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure success=False has an error message."""
        if not self.success and not self.error:
            self.error = "Unknown error"


@dataclass
class TransportError(Exception):
    """Exception raised by transport operations."""

    message: str
    transport: str = ""
    target: str = ""
    cause: Exception | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.transport:
            parts.append(f"transport={self.transport}")
        if self.target:
            parts.append(f"target={self.target}")
        return " | ".join(parts)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 3  # Failures before opening circuit
    recovery_timeout: float = 300.0  # Seconds before trying again
    half_open_max_calls: int = 1  # Calls allowed in half-open state

    @classmethod
    def aggressive(cls) -> CircuitBreakerConfig:
        """Quick-failing configuration for unreliable targets."""
        return cls(failure_threshold=2, recovery_timeout=60.0)

    @classmethod
    def patient(cls) -> CircuitBreakerConfig:
        """Tolerant configuration for occasionally-failing targets."""
        return cls(failure_threshold=5, recovery_timeout=600.0)


@dataclass
class TimeoutConfig:
    """Configuration for transport timeouts."""

    connect_timeout: int = 30  # Seconds to establish connection
    operation_timeout: int = 180  # Seconds for operation to complete
    http_timeout: int = 30  # Seconds for HTTP requests

    @classmethod
    def fast(cls) -> TimeoutConfig:
        """Quick timeout for responsive targets."""
        return cls(connect_timeout=10, operation_timeout=60, http_timeout=15)

    @classmethod
    def slow(cls) -> TimeoutConfig:
        """Extended timeout for slow networks/large transfers."""
        return cls(connect_timeout=60, operation_timeout=600, http_timeout=120)


@dataclass
class TargetStatus:
    """Status of a transport target."""

    state: TransportState = TransportState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_error: str | None = None


# =============================================================================
# Transport Base Class
# =============================================================================


class TransportBase(ABC):
    """Base class for all cluster transport operations.

    Provides unified circuit breaker, timeout, state persistence,
    and health check patterns for transport implementations.

    Attributes:
        name: Human-readable transport name
        connect_timeout: Seconds to establish connection
        operation_timeout: Seconds for operations to complete
        circuit_breaker_config: Circuit breaker settings
    """

    # Default configurations (subclasses can override)
    DEFAULT_CONNECT_TIMEOUT: int = 30
    DEFAULT_OPERATION_TIMEOUT: int = 180
    DEFAULT_FAILURE_THRESHOLD: int = 3
    DEFAULT_RECOVERY_TIMEOUT: float = 300.0

    def __init__(
        self,
        name: str | None = None,
        timeout_config: TimeoutConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        state_path: Path | None = None,
    ):
        """Initialize the transport base.

        Args:
            name: Transport name (defaults to class name)
            timeout_config: Timeout configuration
            circuit_breaker_config: Circuit breaker configuration
            state_path: Path for state persistence (optional)
        """
        self._name = name or self.__class__.__name__

        # Timeout configuration
        timeout_cfg = timeout_config or TimeoutConfig()
        self.connect_timeout = timeout_cfg.connect_timeout
        self.operation_timeout = timeout_cfg.operation_timeout
        self.http_timeout = timeout_cfg.http_timeout

        # Circuit breaker configuration
        cb_cfg = circuit_breaker_config or CircuitBreakerConfig()
        self._failure_threshold = cb_cfg.failure_threshold
        self._recovery_timeout = cb_cfg.recovery_timeout
        self._half_open_max_calls = cb_cfg.half_open_max_calls

        # Per-target circuit breaker state
        self._target_status: dict[str, TargetStatus] = {}

        # State persistence
        self._state_path = state_path

        # Metrics
        self._total_operations = 0
        self._total_successes = 0
        self._total_failures = 0

    @property
    def name(self) -> str:
        """Transport name for logging and metrics."""
        return self._name

    # =========================================================================
    # Circuit Breaker Methods
    # =========================================================================

    def can_attempt(self, target: str) -> bool:
        """Check if operation can be attempted for target.

        Args:
            target: Target identifier (hostname, URL, etc.)

        Returns:
            True if circuit allows operation, False if blocked
        """
        status = self._get_or_create_status(target)

        if status.state == TransportState.CLOSED:
            return True

        if status.state == TransportState.OPEN:
            # Check if recovery timeout has passed
            import time

            elapsed = time.time() - status.last_failure_time
            if elapsed >= self._recovery_timeout:
                # Transition to half-open
                status.state = TransportState.HALF_OPEN
                logger.debug(f"[{self._name}] Circuit half-open for {target}")
                return True
            return False

        if status.state == TransportState.HALF_OPEN:
            # Allow limited calls in half-open state
            return True

        return False

    def record_success(self, target: str, latency_ms: float = 0.0) -> None:
        """Record successful operation for a target.

        Args:
            target: Target identifier
            latency_ms: Operation latency in milliseconds
        """
        import time

        status = self._get_or_create_status(target)
        status.success_count += 1
        status.last_success_time = time.time()
        status.failure_count = 0  # Reset on success
        status.state = TransportState.CLOSED
        status.last_error = None

        self._total_operations += 1
        self._total_successes += 1

        logger.debug(f"[{self._name}] Success for {target} ({latency_ms:.1f}ms)")

    def record_failure(self, target: str, error: Exception | str | None = None) -> None:
        """Record failed operation for a target.

        Args:
            target: Target identifier
            error: Error that caused the failure
        """
        import time

        status = self._get_or_create_status(target)
        status.failure_count += 1
        status.last_failure_time = time.time()

        if error:
            status.last_error = str(error)

        self._total_operations += 1
        self._total_failures += 1

        # Check if circuit should open
        if status.failure_count >= self._failure_threshold:
            if status.state != TransportState.OPEN:
                status.state = TransportState.OPEN
                logger.warning(
                    f"[{self._name}] Circuit opened for {target} "
                    f"(failures={status.failure_count})"
                )
        elif status.state == TransportState.HALF_OPEN:
            # Failed during half-open, go back to open
            status.state = TransportState.OPEN
            logger.debug(f"[{self._name}] Half-open test failed for {target}")

    def reset_circuit(self, target: str) -> None:
        """Reset circuit breaker for a target."""
        if target in self._target_status:
            self._target_status[target] = TargetStatus()
            logger.debug(f"[{self._name}] Circuit reset for {target}")

    def reset_all_circuits(self) -> None:
        """Reset all circuit breakers."""
        self._target_status.clear()
        logger.debug(f"[{self._name}] All circuits reset")

    def get_circuit_state(self, target: str) -> TransportState:
        """Get current circuit state for a target."""
        status = self._target_status.get(target)
        return status.state if status else TransportState.CLOSED

    def get_all_circuit_states(self) -> dict[str, TargetStatus]:
        """Get status of all known targets."""
        return dict(self._target_status)

    def _get_or_create_status(self, target: str) -> TargetStatus:
        """Get or create status for a target."""
        if target not in self._target_status:
            self._target_status[target] = TargetStatus()
        return self._target_status[target]

    # =========================================================================
    # Timeout Helpers
    # =========================================================================

    async def execute_with_timeout(
        self,
        coro,
        timeout: float | None = None,
        timeout_error_msg: str = "Operation timed out",
    ) -> Any:
        """Execute async operation with timeout handling.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds (defaults to operation_timeout)
            timeout_error_msg: Error message on timeout

        Returns:
            Result of the coroutine

        Raises:
            TransportError: On timeout
        """
        timeout_val = timeout or self.operation_timeout

        try:
            return await asyncio.wait_for(coro, timeout=timeout_val)
        except asyncio.TimeoutError:
            raise TransportError(
                message=timeout_error_msg,
                transport=self._name,
            ) from None

    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        target: str,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        timeout: float | None = None,
    ) -> Any:
        """Execute operation with retry logic.

        Args:
            operation: Async callable to execute
            target: Target identifier for circuit breaker
            max_retries: Maximum retry attempts
            backoff_base: Base delay for exponential backoff
            timeout: Timeout per attempt

        Returns:
            Result of successful operation

        Raises:
            TransportError: If all retries fail
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            if not self.can_attempt(target):
                raise TransportError(
                    message="Circuit open",
                    transport=self._name,
                    target=target,
                )

            try:
                result = await self.execute_with_timeout(
                    operation(),
                    timeout=timeout,
                )
                self.record_success(target)
                return result

            except Exception as e:
                last_error = e
                self.record_failure(target, e)

                if attempt < max_retries:
                    delay = backoff_base * (2**attempt)
                    logger.debug(
                        f"[{self._name}] Retry {attempt + 1}/{max_retries} "
                        f"for {target} in {delay}s"
                    )
                    await asyncio.sleep(delay)

        raise TransportError(
            message=f"All {max_retries + 1} attempts failed",
            transport=self._name,
            target=target,
            cause=last_error,
        )

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _load_state(self) -> dict[str, Any]:
        """Load persisted state from JSON.

        Returns:
            Loaded state dict, or empty dict on error
        """
        if not self._state_path:
            return {}

        try:
            if self._state_path.exists():
                with open(self._state_path) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"[{self._name}] Failed to load state: {e}")

        return {}

    def _save_state(self, data: dict[str, Any]) -> bool:
        """Save state to JSON file.

        Args:
            data: State data to persist

        Returns:
            True if saved successfully
        """
        if not self._state_path:
            return False

        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except (OSError, TypeError) as e:
            logger.warning(f"[{self._name}] Failed to save state: {e}")
            return False

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> "HealthCheckResult":
        """Check transport health for daemon monitoring.

        Returns:
            HealthCheckResult with transport health status.

        December 2025: Standardized to return HealthCheckResult instead of dict.
        """
        from app.coordination.contracts import HealthCheckResult

        all_states = self._target_status
        total_targets = len(all_states)
        open_circuits = sum(
            1 for s in all_states.values() if s.state == TransportState.OPEN
        )
        half_open = sum(
            1 for s in all_states.values() if s.state == TransportState.HALF_OPEN
        )

        # Calculate success rate
        success_rate = (
            self._total_successes / self._total_operations
            if self._total_operations > 0
            else 1.0
        )

        details = {
            "transport_name": self._name,
            "total_targets": total_targets,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open,
            "closed_circuits": total_targets - open_circuits - half_open,
            "total_operations": self._total_operations,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": round(success_rate, 3),
        }

        # Determine health status
        if total_targets == 0:
            return HealthCheckResult.healthy("No targets tracked yet", **details)
        elif open_circuits == total_targets:
            return HealthCheckResult.unhealthy(f"All {total_targets} circuits open", **details)
        elif open_circuits > 0:
            return HealthCheckResult.degraded(f"{open_circuits}/{total_targets} circuits open", **details)
        else:
            return HealthCheckResult.healthy(f"All {total_targets} targets reachable", **details)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get transport statistics.

        Returns:
            Dict with operation counts and rates
        """
        return {
            "name": self._name,
            "total_operations": self._total_operations,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": (
                self._total_successes / self._total_operations
                if self._total_operations > 0
                else 0.0
            ),
            "targets_tracked": len(self._target_status),
            "circuits_open": sum(
                1
                for s in self._target_status.values()
                if s.state == TransportState.OPEN
            ),
        }

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def transfer(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> TransportResult:
        """Perform transport operation.

        Must be implemented by subclasses.

        Args:
            source: Source path or URL
            destination: Destination path or URL
            target: Target identifier for circuit breaker
            **kwargs: Transport-specific options

        Returns:
            TransportResult with success status and metadata
        """
        pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base class
    "TransportBase",
    # Data classes
    "TransportResult",
    "TransportError",
    "TransportState",
    "TargetStatus",
    # Configuration
    "TimeoutConfig",
    "CircuitBreakerConfig",
]

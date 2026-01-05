"""Abstract base class for circuit breakers.

January 3, 2026 - Sprint 13.2: Consolidates 10+ circuit breaker implementations
into a unified base class hierarchy.

Circuit Breaker Pattern Overview:
1. CLOSED: Normal operation - requests allowed
2. OPEN: Too many failures - requests blocked for recovery_timeout
3. HALF_OPEN: Testing recovery - limited requests allowed

Target Architecture:
    CircuitBreakerBase (abstract)
    ├── OperationCircuitBreaker  # Per-(target, operation) - replaces inline CBs
    ├── NodeCircuitBreaker       # Per-node health (keep as-is in node_circuit_breaker.py)
    └── ClusterCircuitBreaker    # Cluster-wide (keep as-is in node_circuit_breaker.py)

Usage:
    from app.coordination.circuit_breaker_base import (
        CircuitBreakerBase,
        CircuitConfig,
        CircuitState,
        CircuitStatusBase,
    )

    class MyCircuitBreaker(CircuitBreakerBase):
        def _on_state_change(self, target, old_state, new_state):
            # Custom handling (optional)
            pass
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)

__all__ = [
    "CircuitBreakerBase",
    "CircuitConfig",
    "CircuitState",
    "CircuitStatusBase",
    "CircuitDataBase",
    # Concrete implementations
    "OperationCircuitBreaker",
    "OperationCircuitStatus",
    # Registry and utilities
    "get_operation_circuit_breaker",
    "get_transport_circuit_breaker",
    "OperationCircuitBreakerRegistry",
]


class CircuitState(Enum):
    """Universal circuit breaker states."""

    CLOSED = "closed"  # Normal operation - requests allowed
    OPEN = "open"  # Failures exceeded - requests blocked
    HALF_OPEN = "half_open"  # Testing recovery - limited requests


@dataclass
class CircuitConfig:
    """Configuration for circuit breakers.

    Sprint 13.2: Unified configuration dataclass that can be customized
    per-use case while providing sensible defaults.

    Attributes:
        failure_threshold: Consecutive failures to open circuit (default: 5)
        recovery_timeout: Base seconds before testing recovery (default: 60.0)
        success_threshold: Successes in half-open to close (default: 1)
        half_open_max_calls: Max calls allowed in half-open state (default: 1)
        backoff_multiplier: Exponential backoff multiplier (default: 2.0)
        max_backoff: Maximum recovery timeout seconds (default: 300.0)
        jitter_factor: Random jitter factor 0-1 (default: 0.1)
        emit_events: Whether to emit events on state changes (default: True)
        operation_type: Label for metrics/logging (default: "default")
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 1
    half_open_max_calls: int = 1
    backoff_multiplier: float = 2.0
    max_backoff: float = 300.0
    jitter_factor: float = 0.1
    emit_events: bool = True
    operation_type: str = "default"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be > 0")
        if self.half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be > 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        if self.max_backoff <= 0:
            raise ValueError("max_backoff must be > 0")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")

    @classmethod
    def for_transport(cls, transport: str) -> CircuitConfig:
        """Get transport-specific configuration.

        Args:
            transport: Transport type (http, ssh, rsync, p2p, aria2, tailscale)

        Returns:
            CircuitConfig with transport-appropriate thresholds
        """
        configs = {
            "http": cls(failure_threshold=5, recovery_timeout=30.0, operation_type="http"),
            "ssh": cls(failure_threshold=3, recovery_timeout=60.0, operation_type="ssh"),
            "rsync": cls(failure_threshold=2, recovery_timeout=90.0, operation_type="rsync"),
            "p2p": cls(failure_threshold=3, recovery_timeout=45.0, operation_type="p2p"),
            "aria2": cls(failure_threshold=2, recovery_timeout=120.0, operation_type="aria2"),
            "tailscale": cls(failure_threshold=3, recovery_timeout=60.0, operation_type="tailscale"),
            "base64": cls(failure_threshold=2, recovery_timeout=30.0, operation_type="base64"),
        }
        return configs.get(transport, cls(operation_type=transport))


@dataclass
class CircuitDataBase:
    """Internal circuit data per target.

    Subclasses may extend this with additional fields.
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    opened_at: float | None = None
    half_open_at: float | None = None
    half_open_calls: int = 0
    consecutive_opens: int = 0  # For exponential backoff tracking
    jitter_offset: float = 0.0  # Per-circuit jitter (set when circuit opens)


@dataclass
class CircuitStatusBase:
    """Status of a circuit for a specific target.

    Base class with common fields. Subclasses may extend with additional fields.
    """

    target: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    opened_at: float | None
    half_open_at: float | None
    recovery_timeout: float
    consecutive_opens: int = 0

    @property
    def time_since_open(self) -> float | None:
        """Seconds since circuit opened."""
        if self.opened_at:
            return time.time() - self.opened_at
        return None

    @property
    def time_until_recovery(self) -> float:
        """Seconds until circuit transitions to half-open."""
        if self.state != CircuitState.OPEN or self.opened_at is None:
            return 0.0
        elapsed = time.time() - self.opened_at
        return max(0.0, self.recovery_timeout - elapsed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target": self.target,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "opened_at": self.opened_at,
            "half_open_at": self.half_open_at,
            "recovery_timeout": self.recovery_timeout,
            "consecutive_opens": self.consecutive_opens,
            "time_since_open": self.time_since_open,
            "time_until_recovery": self.time_until_recovery,
        }


class CircuitBreakerBase(ABC):
    """Abstract base class for circuit breakers.

    Sprint 13.2: Provides the common implementation for all circuit breaker
    types. Subclasses only need to implement:
    - _create_circuit_data() - Create circuit data for a new target
    - _create_status() - Create status object from circuit data
    - Optionally: _on_state_change_hook() - Custom state change handling

    Thread-safe via RLock. Supports exponential backoff with jitter.
    """

    def __init__(
        self,
        config: CircuitConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ):
        """Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration (uses defaults if None)
            on_state_change: Optional callback for state transitions.
                            Called with (target, old_state, new_state).
        """
        self.config = config or CircuitConfig()
        self._on_state_change_callback = on_state_change
        self._circuits: dict[str, CircuitDataBase] = {}
        self._lock = RLock()

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _create_circuit_data(self) -> CircuitDataBase:
        """Create a new circuit data instance for a target.

        Override to use custom CircuitData subclass with additional fields.

        Returns:
            New CircuitDataBase instance (or subclass)
        """
        pass

    @abstractmethod
    def _create_status(self, target: str, circuit: CircuitDataBase) -> CircuitStatusBase:
        """Create a status object from circuit data.

        Override to use custom CircuitStatus subclass with additional fields.

        Args:
            target: Target identifier
            circuit: Circuit data

        Returns:
            CircuitStatusBase instance (or subclass)
        """
        pass

    # =========================================================================
    # Optional hooks - subclasses may override
    # =========================================================================

    def _on_state_change_hook(
        self, target: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Hook called on state changes (after callback, before return).

        Override to emit events, update metrics, etc.

        Args:
            target: Target identifier
            old_state: Previous state
            new_state: New state
        """
        pass

    # =========================================================================
    # Core implementation
    # =========================================================================

    def _get_or_create_circuit(self, target: str) -> CircuitDataBase:
        """Get or create circuit data for a target."""
        if target not in self._circuits:
            self._circuits[target] = self._create_circuit_data()
        return self._circuits[target]

    def _compute_jittered_timeout(self, circuit: CircuitDataBase) -> float:
        """Compute recovery timeout with jitter and exponential backoff.

        Returns:
            Recovery timeout in seconds with jitter applied.
        """
        # Exponential backoff: base * (multiplier ^ consecutive_opens)
        backoff = self.config.recovery_timeout * (
            self.config.backoff_multiplier ** circuit.consecutive_opens
        )
        # Cap at max_backoff
        backoff = min(backoff, self.config.max_backoff)
        # Apply jitter: backoff * (1 + jitter_offset)
        if self.config.jitter_factor > 0 and circuit.jitter_offset != 0:
            backoff = backoff * (1.0 + circuit.jitter_offset)
        return max(0.1, backoff)

    def _check_recovery(self, circuit: CircuitDataBase) -> None:
        """Check if circuit should transition to half-open."""
        if circuit.state == CircuitState.OPEN:
            timeout = self._compute_jittered_timeout(circuit)
            if circuit.opened_at and (time.time() - circuit.opened_at) >= timeout:
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_at = time.time()
                circuit.half_open_calls = 0

    def _notify_state_change(
        self, target: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Notify callback and hook on state change."""
        if old_state == new_state:
            return

        # Call optional callback
        if self._on_state_change_callback:
            try:
                self._on_state_change_callback(target, old_state, new_state)
            except Exception as e:
                logger.warning(f"[CircuitBreakerBase] State change callback error: {e}")

        # Call hook for subclass customization
        try:
            self._on_state_change_hook(target, old_state, new_state)
        except Exception as e:
            logger.warning(f"[CircuitBreakerBase] State change hook error: {e}")

    def _set_jitter_offset(self, circuit: CircuitDataBase) -> None:
        """Set random jitter offset to prevent thundering herd."""
        if self.config.jitter_factor > 0:
            # Range: [-jitter_factor, +jitter_factor]
            circuit.jitter_offset = self.config.jitter_factor * (2 * random.random() - 1)

    # =========================================================================
    # Public API - core circuit breaker operations
    # =========================================================================

    def can_execute(self, target: str) -> bool:
        """Check if a request to target is allowed.

        Returns True if:
        - Circuit is CLOSED (normal operation)
        - Circuit is HALF_OPEN and under max test calls

        Returns False if:
        - Circuit is OPEN (blocking requests)
        - Circuit is HALF_OPEN and max test calls reached

        Args:
            target: Target identifier

        Returns:
            True if request is allowed
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            self._check_recovery(circuit)

            if circuit.state == CircuitState.CLOSED:
                return True
            elif circuit.state == CircuitState.HALF_OPEN:
                if circuit.half_open_calls < self.config.half_open_max_calls:
                    circuit.half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self, target: str) -> None:
        """Record a successful operation for target.

        Args:
            target: Target identifier
        """
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.success_count += 1
            circuit.last_success_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.success_count >= self.config.success_threshold:
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    circuit.opened_at = None
                    circuit.half_open_at = None
                    circuit.consecutive_opens = 0
                    circuit.jitter_offset = 0.0
                    logger.info(f"[CircuitBreakerBase] Circuit CLOSED for {target}")
            elif circuit.state == CircuitState.CLOSED:
                circuit.failure_count = 0

            new_state = circuit.state

        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def record_failure(
        self,
        target: str,
        error: Exception | None = None,
        preemptive: bool = False,
    ) -> None:
        """Record a failed operation for target.

        Args:
            target: Target identifier
            error: Optional exception that caused the failure
            preemptive: If True, this is a preemptive failure from gossip replication.
                       Preemptive failures increment failure_count but don't update
                       last_failure_time or trigger immediate state changes.
        """
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.failure_count += 1

            # Preemptive failures from gossip don't update last_failure_time
            if not preemptive:
                circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Failure in half-open: go back to open with increased backoff
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.time()
                circuit.success_count = 0
                circuit.consecutive_opens += 1
                self._set_jitter_offset(circuit)
                logger.warning(
                    f"[CircuitBreakerBase] Circuit OPEN for {target} (half-open test failed)"
                )
            elif circuit.state == CircuitState.CLOSED:
                if circuit.failure_count >= self.config.failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = time.time()
                    circuit.consecutive_opens += 1
                    self._set_jitter_offset(circuit)
                    logger.warning(
                        f"[CircuitBreakerBase] Circuit OPEN for {target} "
                        f"({circuit.failure_count} failures)"
                    )

            new_state = circuit.state

        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def get_state(self, target: str) -> CircuitState:
        """Get current circuit state for a target.

        Args:
            target: Target identifier

        Returns:
            Current CircuitState
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            self._check_recovery(circuit)
            return circuit.state

    def get_status(self, target: str) -> CircuitStatusBase:
        """Get detailed circuit status for a target.

        Args:
            target: Target identifier

        Returns:
            CircuitStatusBase (or subclass) with detailed status
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            self._check_recovery(circuit)
            return self._create_status(target, circuit)

    def get_all_states(self) -> dict[str, CircuitStatusBase]:
        """Get status for all tracked targets.

        Returns:
            Dict mapping target -> CircuitStatusBase
        """
        with self._lock:
            result = {}
            for target in self._circuits:
                result[target] = self.get_status(target)
            return result

    def get_open_circuits(self) -> list[str]:
        """Get list of targets with open circuits.

        Returns:
            List of target identifiers with OPEN circuits
        """
        with self._lock:
            result = []
            for target, circuit in self._circuits.items():
                self._check_recovery(circuit)
                if circuit.state == CircuitState.OPEN:
                    result.append(target)
            return result

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all circuit states.

        Returns:
            Dict with counts and open circuit list
        """
        with self._lock:
            closed_count = 0
            open_count = 0
            half_open_count = 0

            for circuit in self._circuits.values():
                self._check_recovery(circuit)
                if circuit.state == CircuitState.CLOSED:
                    closed_count += 1
                elif circuit.state == CircuitState.OPEN:
                    open_count += 1
                else:
                    half_open_count += 1

            return {
                "total_targets": len(self._circuits),
                "closed": closed_count,
                "open": open_count,
                "half_open": half_open_count,
                "open_targets": self.get_open_circuits(),
                "operation_type": self.config.operation_type,
            }

    # =========================================================================
    # Reset and force operations
    # =========================================================================

    def reset(self, target: str) -> None:
        """Reset circuit for a target to CLOSED state.

        Args:
            target: Target identifier
        """
        with self._lock:
            if target in self._circuits:
                old_state = self._circuits[target].state
                self._circuits[target] = self._create_circuit_data()
                self._notify_state_change(target, old_state, CircuitState.CLOSED)
                logger.info(f"[CircuitBreakerBase] Reset circuit for {target}")

    def reset_all(self) -> None:
        """Reset all circuits to CLOSED state."""
        with self._lock:
            for target in list(self._circuits.keys()):
                old_state = self._circuits[target].state
                self._circuits[target] = self._create_circuit_data()
                self._notify_state_change(target, old_state, CircuitState.CLOSED)
            logger.info(
                f"[CircuitBreakerBase] Reset all circuits ({len(self._circuits)} targets)"
            )

    def force_open(self, target: str) -> None:
        """Force circuit open for a target (manual intervention).

        Args:
            target: Target identifier
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.state = CircuitState.OPEN
            circuit.opened_at = time.time()
            self._set_jitter_offset(circuit)
            self._notify_state_change(target, old_state, CircuitState.OPEN)

    def force_close(self, target: str) -> None:
        """Force circuit closed for a target (manual intervention).

        Args:
            target: Target identifier
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.opened_at = None
            circuit.half_open_at = None
            circuit.jitter_offset = 0.0
            self._notify_state_change(target, old_state, CircuitState.CLOSED)

    def force_reset(self, target: str) -> None:
        """Force reset a circuit to CLOSED state with all counters cleared.

        Use this for manual intervention when a circuit is stuck.

        Args:
            target: Target identifier
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
            circuit.jitter_offset = 0.0

            self._notify_state_change(target, old_state, CircuitState.CLOSED)
            logger.info(f"[CircuitBreakerBase] Force reset circuit for {target}")

    def decay_old_circuits(self, ttl_seconds: float = 21600.0) -> dict[str, list[str]]:
        """Automatically reset circuits that have been open for too long.

        This prevents circuits from being stuck open indefinitely after transient
        failures. After ttl_seconds, circuits are automatically reset to CLOSED,
        allowing operations to resume even if failures previously exceeded threshold.

        Args:
            ttl_seconds: Max time to keep circuit OPEN (default: 21600 = 6 hours)

        Returns:
            Dict with "decayed" list of reset circuit IDs and "checked" count
        """
        now = time.time()
        decayed: list[str] = []
        checked = 0

        with self._lock:
            for target, circuit in self._circuits.items():
                checked += 1
                if circuit.state == CircuitState.OPEN and circuit.opened_at:
                    elapsed = now - circuit.opened_at
                    if elapsed > ttl_seconds:
                        old_state = circuit.state
                        circuit.state = CircuitState.CLOSED
                        circuit.failure_count = 0
                        circuit.consecutive_opens = 0
                        circuit.opened_at = None
                        circuit.half_open_at = None
                        circuit.jitter_offset = 0.0
                        decayed.append(target)
                        self._notify_state_change(target, old_state, CircuitState.CLOSED)
                        logger.info(
                            f"[CircuitBreakerBase] TTL decay reset circuit for {target} "
                            f"(was open for {elapsed:.0f}s, threshold: {ttl_seconds:.0f}s)"
                        )

        if decayed:
            logger.info(
                f"[CircuitBreakerBase] TTL decay: reset {len(decayed)}/{checked} circuits"
            )

        return {"decayed": decayed, "checked": checked}

    # =========================================================================
    # Health check support
    # =========================================================================

    def health_check(self) -> "HealthCheckResult":
        """Health check for monitoring integration.

        Returns:
            HealthCheckResult with circuit breaker health status
        """
        # Lazy import to avoid circular dependency
        from app.coordination.protocols import HealthCheckResult

        summary = self.get_summary()
        total = summary["total_targets"]
        open_count = summary["open"]

        # Unhealthy if more than 50% of circuits are open
        open_ratio = open_count / total if total > 0 else 0.0
        is_healthy = open_ratio < 0.5

        if is_healthy:
            message = "OK"
        else:
            message = f"High open circuit ratio: {open_ratio:.1%}"

        return HealthCheckResult(
            healthy=is_healthy,
            message=message,
            details={
                "operation_type": self.config.operation_type,
                "total_targets": total,
                "open_circuits": open_count,
                "half_open_circuits": summary["half_open"],
                "closed_circuits": summary["closed"],
                "open_ratio": round(open_ratio, 3),
                "open_targets": summary["open_targets"],
            },
        )


# =============================================================================
# OperationCircuitBreaker - Concrete implementation for per-operation CB
# =============================================================================


@dataclass
class OperationCircuitData(CircuitDataBase):
    """Extended circuit data with escalation tracking.

    Adds escalation tier support for multi-level recovery.
    Session 17.25: Added consecutive_successful_probes for tier decay.
    """

    escalation_tier: int = 0
    escalation_entered_at: float | None = None
    last_probe_at: float | None = None
    # Session 17.25: Track consecutive successes for tier decay
    consecutive_successful_probes: int = 0


@dataclass
class OperationCircuitStatus(CircuitStatusBase):
    """Extended status with escalation information."""

    escalation_tier: int = 0
    escalation_entered_at: float | None = None
    time_until_next_probe: float | None = None
    is_escalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "escalation_tier": self.escalation_tier,
            "escalation_entered_at": self.escalation_entered_at,
            "time_until_next_probe": self.time_until_next_probe,
            "is_escalated": self.is_escalated,
        })
        return result


# Escalation tiers for automatic recovery (from circuit_breaker.py)
ESCALATION_TIERS = [
    {"wait": 60, "probe_interval": 10},      # Tier 0: 1 min wait, probe every 10s
    {"wait": 300, "probe_interval": 30},     # Tier 1: 5 min wait, probe every 30s
    {"wait": 900, "probe_interval": 60},     # Tier 2: 15 min wait, probe every 1 min
    {"wait": 3600, "probe_interval": 300},   # Tier 3: 1 hour wait, probe every 5 min
]
MAX_ESCALATION_TIER = len(ESCALATION_TIERS) - 1


class OperationCircuitBreaker(CircuitBreakerBase):
    """Concrete circuit breaker for per-(target, operation) tracking.

    Sprint 13.2: Consolidates transport-specific circuit breakers into a
    unified implementation with:
    - Escalation tiers for long-running failures
    - Prometheus metrics integration
    - Gossip replication support for cluster-wide awareness
    - Event emission for monitoring

    Usage:
        from app.coordination.circuit_breaker_base import (
            OperationCircuitBreaker,
            CircuitConfig,
        )

        # Create breaker for SSH operations
        config = CircuitConfig.for_transport("ssh")
        breaker = OperationCircuitBreaker(config)

        # Check before operation
        if breaker.can_execute("host1"):
            try:
                result = ssh_connect("host1")
                breaker.record_success("host1")
            except Exception as e:
                breaker.record_failure("host1", e)
        else:
            # Circuit open - use fallback transport
            pass
    """

    # Prometheus metrics (lazy-loaded)
    _prom_state: Any = None
    _prom_failures: Any = None
    _prom_successes: Any = None
    _prom_opens: Any = None
    _prom_blocked: Any = None
    _prom_escalation: Any = None
    _prometheus_initialized: bool = False

    def __init__(
        self,
        config: CircuitConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
        max_consecutive_opens: int = 5,
    ):
        """Initialize the operation circuit breaker.

        Args:
            config: Circuit breaker configuration
            on_state_change: Optional callback for state transitions
            max_consecutive_opens: Opens before escalating (default: 5)
        """
        super().__init__(config=config, on_state_change=on_state_change)
        self.max_consecutive_opens = max_consecutive_opens
        self._init_prometheus()

    def _init_prometheus(self) -> None:
        """Initialize Prometheus metrics (once per class)."""
        if OperationCircuitBreaker._prometheus_initialized:
            return

        try:
            from prometheus_client import REGISTRY, Counter, Gauge

            def get_or_create_gauge(name: str, desc: str, labels: list):
                try:
                    return Gauge(name, desc, labels)
                except ValueError:
                    return REGISTRY._names_to_collectors.get(name)

            def get_or_create_counter(name: str, desc: str, labels: list):
                try:
                    return Counter(name, desc, labels)
                except ValueError:
                    return REGISTRY._names_to_collectors.get(name)

            OperationCircuitBreaker._prom_state = get_or_create_gauge(
                "ringrift_op_circuit_state",
                "Circuit state (0=closed, 1=open, 2=half_open)",
                ["operation_type", "target"],
            )
            OperationCircuitBreaker._prom_failures = get_or_create_counter(
                "ringrift_op_circuit_failures_total",
                "Total failures recorded",
                ["operation_type", "target"],
            )
            OperationCircuitBreaker._prom_successes = get_or_create_counter(
                "ringrift_op_circuit_successes_total",
                "Total successes recorded",
                ["operation_type", "target"],
            )
            OperationCircuitBreaker._prom_opens = get_or_create_counter(
                "ringrift_op_circuit_opens_total",
                "Total circuit open events",
                ["operation_type", "target"],
            )
            OperationCircuitBreaker._prom_blocked = get_or_create_counter(
                "ringrift_op_circuit_blocked_total",
                "Total requests blocked by open circuit",
                ["operation_type", "target"],
            )
            OperationCircuitBreaker._prom_escalation = get_or_create_gauge(
                "ringrift_op_circuit_escalation_tier",
                "Current escalation tier (0-3)",
                ["operation_type", "target"],
            )
            OperationCircuitBreaker._prometheus_initialized = True
        except ImportError:
            OperationCircuitBreaker._prometheus_initialized = True  # Skip metrics

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _create_circuit_data(self) -> OperationCircuitData:
        """Create a new circuit data instance."""
        return OperationCircuitData()

    def _create_status(
        self, target: str, circuit: CircuitDataBase
    ) -> OperationCircuitStatus:
        """Create a status object from circuit data."""
        op_circuit = circuit if isinstance(circuit, OperationCircuitData) else None
        escalation_tier = op_circuit.escalation_tier if op_circuit else 0
        escalation_entered_at = op_circuit.escalation_entered_at if op_circuit else None
        time_until_probe = self._get_time_until_next_probe(circuit) if op_circuit else None

        return OperationCircuitStatus(
            target=target,
            state=circuit.state,
            failure_count=circuit.failure_count,
            success_count=circuit.success_count,
            last_failure_time=circuit.last_failure_time,
            last_success_time=circuit.last_success_time,
            opened_at=circuit.opened_at,
            half_open_at=circuit.half_open_at,
            recovery_timeout=self._compute_jittered_timeout(circuit),
            consecutive_opens=circuit.consecutive_opens,
            escalation_tier=escalation_tier,
            escalation_entered_at=escalation_entered_at,
            time_until_next_probe=time_until_probe,
            is_escalated=escalation_tier > 0,
        )

    def _on_state_change_hook(
        self, target: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Emit events and update metrics on state change."""
        # Update Prometheus metrics
        self._update_prometheus_state(target, new_state)

        # Emit event if configured
        if self.config.emit_events:
            self._emit_state_change_event(target, old_state, new_state)

    # =========================================================================
    # Escalation tier support
    # =========================================================================

    def _get_escalation_tier(self, circuit: CircuitDataBase) -> int:
        """Get the current escalation tier for a circuit.

        Returns the tier based on consecutive_opens:
        - 0-4: Normal backoff (tier 0)
        - 5-9: Tier 1
        - 10-14: Tier 2
        - 15+: Tier 3 (max)
        """
        if circuit.consecutive_opens < self.max_consecutive_opens:
            return 0
        extra_opens = circuit.consecutive_opens - self.max_consecutive_opens
        tier = min(1 + extra_opens // 5, MAX_ESCALATION_TIER)
        return tier

    def _get_tier_config(self, tier: int) -> dict:
        """Get configuration for a specific escalation tier."""
        tier = min(tier, MAX_ESCALATION_TIER)
        return ESCALATION_TIERS[tier] if tier < len(ESCALATION_TIERS) else ESCALATION_TIERS[-1]

    def _get_time_until_next_probe(self, circuit: CircuitDataBase) -> float | None:
        """Get seconds until next probe is allowed."""
        if not isinstance(circuit, OperationCircuitData):
            return None

        tier = self._get_escalation_tier(circuit)
        config = self._get_tier_config(tier)

        # Check tier wait first
        if circuit.escalation_entered_at:
            time_in_tier = time.time() - circuit.escalation_entered_at
            if time_in_tier < config["wait"]:
                return config["wait"] - time_in_tier

        # Check probe interval
        if circuit.last_probe_at is None:
            return 0.0

        time_since_probe = time.time() - circuit.last_probe_at
        if time_since_probe >= config["probe_interval"]:
            return 0.0
        return config["probe_interval"] - time_since_probe

    def _should_probe_in_tier(self, circuit: OperationCircuitData) -> bool:
        """Check if we should attempt a probe based on tier timing."""
        tier = self._get_escalation_tier(circuit)
        config = self._get_tier_config(tier)

        # Check if we've waited long enough since entering this tier
        if circuit.escalation_entered_at:
            time_in_tier = time.time() - circuit.escalation_entered_at
            if time_in_tier < config["wait"]:
                return False

        # Check probe interval
        if circuit.last_probe_at is None:
            return True

        time_since_probe = time.time() - circuit.last_probe_at
        return time_since_probe >= config["probe_interval"]

    def _escalate(self, circuit: OperationCircuitData, target: str = "") -> None:
        """Move circuit to escalation mode if needed."""
        old_tier = circuit.escalation_tier
        new_tier = self._get_escalation_tier(circuit)
        if new_tier != old_tier:
            circuit.escalation_tier = new_tier
            circuit.escalation_entered_at = time.time()
            # Emit escalation event
            if self.config.emit_events:
                self._emit_escalation_event(target, old_tier, new_tier, circuit.consecutive_opens)

    # =========================================================================
    # Override record_failure for escalation
    # =========================================================================

    def record_failure(
        self,
        target: str,
        error: Exception | None = None,
        preemptive: bool = False,
    ) -> None:
        """Record a failed operation with escalation tracking."""
        old_state = None
        new_state = None
        opened_circuit = False

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.failure_count += 1

            # Session 17.25: Reset consecutive successful probes on any failure
            # This ensures only truly consecutive successes trigger tier decay
            if isinstance(circuit, OperationCircuitData):
                circuit.consecutive_successful_probes = 0

            if not preemptive:
                circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.time()
                circuit.success_count = 0
                circuit.consecutive_opens += 1
                self._set_jitter_offset(circuit)
                if isinstance(circuit, OperationCircuitData):
                    self._escalate(circuit, target)
                opened_circuit = True
                logger.warning(
                    f"[OperationCircuitBreaker] Circuit OPEN for {target} (half-open test failed)"
                )
            elif circuit.state == CircuitState.CLOSED:
                if circuit.failure_count >= self.config.failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = time.time()
                    circuit.consecutive_opens += 1
                    self._set_jitter_offset(circuit)
                    if isinstance(circuit, OperationCircuitData):
                        self._escalate(circuit, target)
                    opened_circuit = True
                    logger.warning(
                        f"[OperationCircuitBreaker] Circuit OPEN for {target} "
                        f"({circuit.failure_count} failures)"
                    )

            new_state = circuit.state

        # Update Prometheus metrics
        if OperationCircuitBreaker._prom_failures:
            try:
                OperationCircuitBreaker._prom_failures.labels(
                    operation_type=self.config.operation_type, target=target
                ).inc()
                if opened_circuit and OperationCircuitBreaker._prom_opens:
                    OperationCircuitBreaker._prom_opens.labels(
                        operation_type=self.config.operation_type, target=target
                    ).inc()
            except (ValueError, TypeError):
                pass

        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def record_success(self, target: str) -> None:
        """Record a successful operation with escalation tier decay.

        Session 17.25: Added gradual tier decay after 3 consecutive successes.
        Previously tier was only reset to 0 on HALF_OPEN -> CLOSED transition.
        Now tier decays by 1 after 3 consecutive successful probes, allowing
        faster recovery from elevated escalation states.
        """
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.success_count += 1
            circuit.last_success_time = time.time()

            # Session 17.25: Track consecutive successful probes for tier decay
            if isinstance(circuit, OperationCircuitData):
                circuit.consecutive_successful_probes += 1
                # Decay tier after 3 consecutive successful probes
                if circuit.escalation_tier > 0 and circuit.consecutive_successful_probes >= 3:
                    old_tier = circuit.escalation_tier
                    circuit.escalation_tier -= 1
                    circuit.consecutive_successful_probes = 0
                    logger.info(
                        f"[OperationCircuitBreaker] Escalation tier decayed "
                        f"{old_tier} -> {circuit.escalation_tier} for {target} (3 consecutive successes)"
                    )

            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.success_count >= self.config.success_threshold:
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    circuit.opened_at = None
                    circuit.half_open_at = None
                    circuit.consecutive_opens = 0
                    circuit.jitter_offset = 0.0
                    # Reset escalation state fully when circuit closes
                    if isinstance(circuit, OperationCircuitData):
                        circuit.escalation_tier = 0
                        circuit.escalation_entered_at = None
                        circuit.last_probe_at = None
                        circuit.consecutive_successful_probes = 0
                    logger.info(f"[OperationCircuitBreaker] Circuit CLOSED for {target}")
            elif circuit.state == CircuitState.CLOSED:
                circuit.failure_count = 0

            new_state = circuit.state

        # Update Prometheus metrics
        if OperationCircuitBreaker._prom_successes:
            try:
                OperationCircuitBreaker._prom_successes.labels(
                    operation_type=self.config.operation_type, target=target
                ).inc()
            except (ValueError, TypeError):
                pass

        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def can_execute(self, target: str) -> bool:
        """Check if a request to target is allowed (with metrics)."""
        result = super().can_execute(target)

        if not result and OperationCircuitBreaker._prom_blocked:
            try:
                OperationCircuitBreaker._prom_blocked.labels(
                    operation_type=self.config.operation_type, target=target
                ).inc()
            except (ValueError, TypeError):
                pass

        return result

    # =========================================================================
    # Active recovery
    # =========================================================================

    def try_active_recovery(
        self,
        target: str,
        probe_func: Callable[[str], bool] | None = None,
    ) -> bool:
        """Actively probe target for recovery.

        Args:
            target: Target identifier
            probe_func: Optional health probe function (returns True if healthy)

        Returns:
            True if recovery probe succeeded
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)

            if circuit.state != CircuitState.OPEN:
                return circuit.state == CircuitState.CLOSED

            # Check tier timing for escalated circuits
            if isinstance(circuit, OperationCircuitData):
                if not self._should_probe_in_tier(circuit):
                    return False
                circuit.last_probe_at = time.time()

        # Run probe outside lock
        if probe_func:
            try:
                probe_success = probe_func(target)
            except (ConnectionError, TimeoutError, OSError, RuntimeError):
                probe_success = False
        else:
            probe_success = False  # No probe function = no recovery

        if probe_success:
            with self._lock:
                old_state = circuit.state
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_at = time.time()
                circuit.half_open_calls = 0
                self._notify_state_change(target, old_state, CircuitState.HALF_OPEN)
            return True

        return False

    def is_permanently_open(self, target: str) -> bool:
        """Check if circuit is at maximum escalation tier.

        Returns:
            True if circuit is OPEN and at max escalation tier
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            if circuit.state != CircuitState.OPEN:
                return False
            if isinstance(circuit, OperationCircuitData):
                return circuit.escalation_tier >= MAX_ESCALATION_TIER
            return circuit.consecutive_opens >= self.max_consecutive_opens

    # =========================================================================
    # Prometheus and event helpers
    # =========================================================================

    def _update_prometheus_state(self, target: str, state: CircuitState) -> None:
        """Update Prometheus state gauge."""
        if OperationCircuitBreaker._prom_state:
            try:
                state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state.value, 0)
                OperationCircuitBreaker._prom_state.labels(
                    operation_type=self.config.operation_type, target=target
                ).set(state_value)
            except (ValueError, TypeError):
                pass

        # Update escalation tier
        if OperationCircuitBreaker._prom_escalation:
            with self._lock:
                circuit = self._circuits.get(target)
                if circuit and isinstance(circuit, OperationCircuitData):
                    try:
                        OperationCircuitBreaker._prom_escalation.labels(
                            operation_type=self.config.operation_type, target=target
                        ).set(circuit.escalation_tier)
                    except (ValueError, TypeError):
                        pass

    def _emit_state_change_event(
        self, target: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Emit event for circuit state change."""
        try:
            from app.coordination.event_router import get_event_router

            router = get_event_router()
            if router:
                from app.distributed.data_events import DataEventType

                event_type = None
                if new_state == CircuitState.OPEN:
                    event_type = DataEventType.CIRCUIT_BREAKER_OPENED
                elif new_state == CircuitState.CLOSED:
                    event_type = DataEventType.CIRCUIT_BREAKER_CLOSED
                elif new_state == CircuitState.HALF_OPEN:
                    event_type = DataEventType.CIRCUIT_BREAKER_HALF_OPEN

                if event_type:
                    router.emit(
                        event_type.value,
                        {
                            "target": target,
                            "operation_type": self.config.operation_type,
                            "old_state": old_state.value,
                            "new_state": new_state.value,
                            "timestamp": time.time(),
                        },
                    )
        except (ImportError, RuntimeError, TypeError, AttributeError):
            pass  # Graceful fallback if event system unavailable

    def _emit_escalation_event(
        self, target: str, old_tier: int, new_tier: int, consecutive_opens: int
    ) -> None:
        """Emit event for escalation tier change."""
        try:
            from app.coordination.event_router import get_event_router

            router = get_event_router()
            if router:
                from app.distributed.data_events import DataEventType

                tier_config = self._get_tier_config(new_tier)
                router.emit(
                    DataEventType.ESCALATION_TIER_CHANGED.value,
                    {
                        "target": target,
                        "operation_type": self.config.operation_type,
                        "old_tier": old_tier,
                        "new_tier": new_tier,
                        "consecutive_opens": consecutive_opens,
                        "wait_seconds": tier_config["wait"],
                        "probe_interval": tier_config["probe_interval"],
                        "timestamp": time.time(),
                    },
                )
        except (ImportError, RuntimeError, TypeError, AttributeError):
            pass


# =============================================================================
# Registry and singleton access
# =============================================================================


class OperationCircuitBreakerRegistry:
    """Registry for operation-type specific circuit breakers.

    Sprint 13.2: Provides singleton access to circuit breakers by operation type.
    Replaces multiple inline registries with a unified approach.

    Usage:
        registry = get_operation_circuit_registry()
        breaker = registry.get_breaker("ssh")

        if breaker.can_execute("host1"):
            ...
    """

    _instance: OperationCircuitBreakerRegistry | None = None
    _lock = RLock()

    def __init__(self):
        self._breakers: dict[str, OperationCircuitBreaker] = {}
        self._breaker_lock = RLock()

    @classmethod
    def get_instance(cls) -> OperationCircuitBreakerRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def get_breaker(
        self,
        operation_type: str,
        config: CircuitConfig | None = None,
    ) -> OperationCircuitBreaker:
        """Get or create a circuit breaker for an operation type.

        Args:
            operation_type: Type of operation (e.g., "ssh", "http", "p2p")
            config: Optional custom configuration

        Returns:
            OperationCircuitBreaker for the operation type
        """
        with self._breaker_lock:
            if operation_type not in self._breakers:
                cfg = config or CircuitConfig.for_transport(operation_type)
                self._breakers[operation_type] = OperationCircuitBreaker(config=cfg)
            return self._breakers[operation_type]

    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get summary for all operation types."""
        with self._breaker_lock:
            return {
                op_type: breaker.get_summary()
                for op_type, breaker in self._breakers.items()
            }

    def get_all_open_circuits(self) -> dict[str, list[str]]:
        """Get all open circuits grouped by operation type."""
        with self._breaker_lock:
            return {
                op_type: breaker.get_open_circuits()
                for op_type, breaker in self._breakers.items()
                if breaker.get_open_circuits()
            }

    def decay_all_old_circuits(self, ttl_seconds: float = 21600.0) -> dict[str, Any]:
        """Decay old circuits across all operation types.

        Call this periodically (e.g., every hour) to prevent stuck circuits.

        Args:
            ttl_seconds: Max time to keep circuit OPEN (default: 6 hours)

        Returns:
            Dict with results per operation type
        """
        results: dict[str, Any] = {}
        total_decayed = 0
        total_checked = 0

        with self._breaker_lock:
            for op_type, breaker in self._breakers.items():
                result = breaker.decay_old_circuits(ttl_seconds)
                results[op_type] = result
                total_decayed += len(result.get("decayed", []))
                total_checked += result.get("checked", 0)

        if total_decayed > 0:
            logger.info(
                f"[OperationCircuitBreakerRegistry] TTL decay: "
                f"reset {total_decayed}/{total_checked} circuits across {len(results)} breakers"
            )

        return {
            "by_operation": results,
            "total_decayed": total_decayed,
            "total_checked": total_checked,
        }

    def health_check(self) -> "HealthCheckResult":
        """Health check for monitoring integration."""
        from app.coordination.protocols import HealthCheckResult

        total_targets = 0
        open_circuits = 0
        half_open_circuits = 0

        with self._breaker_lock:
            for breaker in self._breakers.values():
                summary = breaker.get_summary()
                total_targets += summary["total_targets"]
                open_circuits += summary["open"]
                half_open_circuits += summary["half_open"]

        open_ratio = open_circuits / total_targets if total_targets > 0 else 0.0
        is_healthy = open_ratio < 0.5

        return HealthCheckResult(
            healthy=is_healthy,
            message="OK" if is_healthy else f"High open circuit ratio: {open_ratio:.1%}",
            details={
                "total_operation_types": len(self._breakers),
                "total_targets": total_targets,
                "open_circuits": open_circuits,
                "half_open_circuits": half_open_circuits,
                "open_ratio": round(open_ratio, 3),
            },
        )


# Module-level singletons
_registry: OperationCircuitBreakerRegistry | None = None
_transport_breakers: dict[str, OperationCircuitBreaker] = {}
_transport_lock = RLock()


def get_operation_circuit_registry() -> OperationCircuitBreakerRegistry:
    """Get the global operation circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = OperationCircuitBreakerRegistry.get_instance()
    return _registry


def get_operation_circuit_breaker(operation_type: str) -> OperationCircuitBreaker:
    """Get circuit breaker for a specific operation type.

    Args:
        operation_type: Type of operation (e.g., "ssh", "http")

    Returns:
        OperationCircuitBreaker for the operation type
    """
    return get_operation_circuit_registry().get_breaker(operation_type)


def get_transport_circuit_breaker(host: str, transport: str) -> OperationCircuitBreaker:
    """Get a circuit breaker for a specific (host, transport) combination.

    This enables per-transport failover: if HTTP fails for a host, SSH can
    still work because they have separate circuit breakers.

    Args:
        host: Hostname or IP of the target
        transport: Transport type ("http", "ssh", "rsync", "p2p", etc.)

    Returns:
        OperationCircuitBreaker for this (host, transport) combination
    """
    key = f"{host}:{transport}"

    with _transport_lock:
        if key not in _transport_breakers:
            config = CircuitConfig.for_transport(transport)
            # Set operation_type to include host for better metrics labeling
            config.operation_type = f"{transport}:{host}"
            _transport_breakers[key] = OperationCircuitBreaker(config=config)
        return _transport_breakers[key]


def decay_all_circuit_breakers(ttl_seconds: float = 21600.0) -> dict[str, Any]:
    """Decay old circuits across all circuit breaker registries.

    Call this periodically (e.g., every hour) from master_loop or health checks
    to prevent circuits from being stuck open indefinitely.

    This covers:
    - OperationCircuitBreakerRegistry (for transport operations)
    - Transport-level circuit breakers (per host:transport)

    Args:
        ttl_seconds: Max time to keep circuit OPEN (default: 6 hours)

    Returns:
        Dict with decay results from all registries
    """
    results: dict[str, Any] = {}

    # Decay operation registry circuits
    try:
        registry = get_operation_circuit_registry()
        results["operation_registry"] = registry.decay_all_old_circuits(ttl_seconds)
    except Exception as e:
        logger.warning(f"[decay_all_circuit_breakers] Operation registry error: {e}")
        results["operation_registry"] = {"error": str(e)}

    # Decay transport-level circuits
    transport_decayed = []
    transport_checked = 0
    now = time.time()

    with _transport_lock:
        for key, breaker in _transport_breakers.items():
            result = breaker.decay_old_circuits(ttl_seconds)
            transport_checked += result.get("checked", 0)
            transport_decayed.extend(result.get("decayed", []))

    results["transport_breakers"] = {
        "decayed": transport_decayed,
        "checked": transport_checked,
    }

    total_decayed = (
        len(results.get("operation_registry", {}).get("decayed", []))
        + len(transport_decayed)
    )

    if total_decayed > 0:
        logger.info(
            f"[decay_all_circuit_breakers] Total TTL decay: {total_decayed} circuits reset"
        )

    return results

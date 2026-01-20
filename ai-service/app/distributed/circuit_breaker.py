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
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from app.coordination.protocols import HealthCheckResult

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
    # Phase 15.1.4: Health probe utilities
    "DEFAULT_HEALTH_PROBE_PORTS",
    "DEFAULT_HEALTH_PROBE_TIMEOUT",
    "start_recovery_probing",
    "stop_recovery_probing",
    # Per-transport circuit breakers (January 2026 - Phase 1)
    "get_transport_breaker",
    "check_transport_circuit",
    "record_transport_success",
    "record_transport_failure",
    "get_open_transports_for_host",
    "get_available_transports_for_host",
    "reset_transport_breakers_for_host",
    "get_all_transport_breaker_states",
    "TRANSPORT_CONFIGS",
    # Jan 5, 2026: Transport-specific TTL decay
    "decay_transport_circuit_breakers",
    "DEFAULT_TRANSPORT_DECAY_TTL",
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

# Phase 15.1.4: Default health probe ports by target pattern (December 2025)
# Maps target name patterns to (port, endpoint) tuples for health checks
DEFAULT_HEALTH_PROBE_PORTS = {
    "p2p": (8770, "/health"),      # P2P orchestrator
    "http": (80, "/health"),        # HTTP services
    "data": (8780, "/health"),      # Data distribution
    "ssh": None,                    # SSH uses subprocess, no HTTP probe
    "rsync": None,                  # rsync uses subprocess
    "aria2": (6800, "/jsonrpc"),    # aria2 RPC
}
DEFAULT_HEALTH_PROBE_TIMEOUT = 5.0

# Phase 15.1.8: Escalation tiers for automatic recovery (December 2025)
# Instead of requiring manual force_reset() after max_consecutive_opens,
# circuits escalate through increasingly longer recovery periods.
# Each tier has:
#   - wait: Minimum seconds before attempting recovery
#   - probe_interval: Seconds between probe attempts in this tier
ESCALATION_TIERS = [
    {"wait": 60, "probe_interval": 10},      # Tier 0: 1 min wait, probe every 10s
    {"wait": 300, "probe_interval": 30},     # Tier 1: 5 min wait, probe every 30s
    {"wait": 900, "probe_interval": 60},     # Tier 2: 15 min wait, probe every 1 min
    {"wait": 3600, "probe_interval": 300},   # Tier 3: 1 hour wait, probe every 5 min
]
MAX_ESCALATION_TIER = len(ESCALATION_TIERS) - 1

# Sprint 12: Event emission for escalation tier changes
_event_router = None


def _get_event_router():
    """Lazy-load event router to avoid circular imports."""
    global _event_router
    if _event_router is None:
        try:
            from app.coordination.event_router import get_event_router
            _event_router = get_event_router()
        except ImportError:
            pass
    return _event_router


def _emit_escalation_event(target: str, old_tier: int, new_tier: int, consecutive_opens: int, operation_type: str) -> None:
    """Emit ESCALATION_TIER_CHANGED event for monitoring."""
    router = _get_event_router()
    if router is None:
        return
    try:
        from app.distributed.data_events import DataEventType
        tier_config = ESCALATION_TIERS[new_tier] if new_tier < len(ESCALATION_TIERS) else ESCALATION_TIERS[-1]
        router.emit(
            DataEventType.ESCALATION_TIER_CHANGED.value,
            {
                "target": target,
                "operation_type": operation_type,
                "old_tier": old_tier,
                "new_tier": new_tier,
                "consecutive_opens": consecutive_opens,
                "wait_seconds": tier_config["wait"],
                "probe_interval": tier_config["probe_interval"],
                "timestamp": time.time(),
            },
        )
    except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
        # Graceful fallback if event system unavailable
        pass

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
    # Sprint 10 (Jan 3, 2026): Escalation tier tracking
    PROM_ESCALATION_TIER = _get_or_create_gauge(
        'ringrift_circuit_breaker_escalation_tier',
        'Current escalation tier for circuit breaker (0-5)',
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
    # Phase 15.1.8: Escalation tracking
    escalation_tier: int = 0
    escalation_entered_at: float | None = None
    time_until_next_probe: float | None = None

    @property
    def time_since_open(self) -> float | None:
        """Seconds since circuit opened."""
        if self.opened_at:
            return time.time() - self.opened_at
        return None

    @property
    def is_escalated(self) -> bool:
        """True if circuit is in an escalated state (beyond normal backoff)."""
        return self.escalation_tier > 0

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
            "escalation_tier": self.escalation_tier,
            "escalation_entered_at": self.escalation_entered_at,
            "is_escalated": self.is_escalated,
            "time_until_next_probe": self.time_until_next_probe,
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
    # Phase 15.1.8: Escalation tier tracking
    escalation_tier: int = 0  # Current escalation tier (0 = normal)
    escalation_entered_at: float | None = None  # When we entered current tier
    last_probe_at: float | None = None  # When we last attempted a probe


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

    def _default_health_probe(self, target: str) -> bool:
        """Default HTTP health probe for a target (Phase 15.1.4).

        Attempts an HTTP GET to common health endpoints based on the
        operation_type and target. Returns True if the probe succeeds.

        This provides automatic recovery probing for circuits without
        requiring explicit probe configuration.

        Args:
            target: Target identifier (hostname or hostname:port)

        Returns:
            True if health check passed, False otherwise.
        """
        import socket
        import urllib.request
        import urllib.error

        # Determine health endpoint based on operation type
        probe_config = DEFAULT_HEALTH_PROBE_PORTS.get(self.operation_type)
        if probe_config is None:
            # No HTTP probe for this operation type (e.g., SSH, rsync)
            return False

        port, endpoint = probe_config

        # Extract hostname from target (may include :port suffix)
        hostname = target.split(":")[0] if ":" in target else target

        # Build health check URL
        url = f"http://{hostname}:{port}{endpoint}"

        try:
            # Use short timeout for health probes
            with urllib.request.urlopen(
                url,
                timeout=DEFAULT_HEALTH_PROBE_TIMEOUT
            ) as response:
                return response.status == 200
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            socket.timeout,
            OSError,
            ConnectionError,
        ):
            return False
        except Exception as e:
            # Catch-all for unexpected errors - log for visibility
            logger.debug(f"Unexpected error in health probe for {target}: {type(e).__name__}: {e}")
            return False

    def _get_escalation_tier(self, circuit: _CircuitData) -> int:
        """Get the current escalation tier for a circuit (Phase 15.1.8).

        Returns the tier based on consecutive_opens:
        - 0-4: Normal backoff (tier 0)
        - 5-9: Tier 1
        - 10-14: Tier 2
        - 15+: Tier 3 (max)
        """
        if circuit.consecutive_opens < self.max_consecutive_opens:
            return 0
        # Each 5 additional opens moves up a tier
        extra_opens = circuit.consecutive_opens - self.max_consecutive_opens
        tier = min(1 + extra_opens // 5, MAX_ESCALATION_TIER)
        return tier

    def _get_tier_config(self, tier: int) -> dict:
        """Get configuration for a specific escalation tier."""
        tier = min(tier, MAX_ESCALATION_TIER)
        return ESCALATION_TIERS[tier] if tier < len(ESCALATION_TIERS) else ESCALATION_TIERS[-1]

    def _should_probe_in_tier(self, circuit: _CircuitData) -> bool:
        """Check if we should attempt a probe based on tier timing (Phase 15.1.8).

        Returns True if enough time has passed since the last probe
        according to the current escalation tier's probe_interval.
        """
        tier = self._get_escalation_tier(circuit)
        config = self._get_tier_config(tier)

        # First, check if we've waited long enough since entering this tier
        if circuit.escalation_entered_at:
            time_in_tier = time.time() - circuit.escalation_entered_at
            if time_in_tier < config["wait"]:
                return False

        # Check probe interval
        if circuit.last_probe_at is None:
            return True

        time_since_probe = time.time() - circuit.last_probe_at
        return time_since_probe >= config["probe_interval"]

    def _get_time_until_next_probe(self, circuit: _CircuitData) -> float | None:
        """Get seconds until next probe is allowed (Phase 15.1.8)."""
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

    def _escalate(self, circuit: _CircuitData, target: str = "") -> None:
        """Move circuit to escalation mode (Phase 15.1.8).

        Called when consecutive_opens exceeds max_consecutive_opens.
        Instead of being "permanently open", the circuit enters escalation
        where it will still attempt recovery with longer intervals.
        """
        old_tier = circuit.escalation_tier
        new_tier = self._get_escalation_tier(circuit)
        if new_tier != old_tier:
            circuit.escalation_tier = new_tier
            circuit.escalation_entered_at = time.time()
            # Sprint 12: Emit event for monitoring
            _emit_escalation_event(
                target=target,
                old_tier=old_tier,
                new_tier=new_tier,
                consecutive_opens=circuit.consecutive_opens,
                operation_type=self.operation_type,
            )

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
                    # Phase 15.1.8: Reset escalation state
                    circuit.escalation_tier = 0
                    circuit.escalation_entered_at = None
                    circuit.last_probe_at = None
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
                # Sprint 10 (Jan 3, 2026): Update escalation tier gauge
                with self._lock:
                    circuit = self._get_or_create_circuit(target)
                    PROM_ESCALATION_TIER.labels(
                        operation_type=self.operation_type, target=target
                    ).set(circuit.escalation_tier)
            except (ValueError, TypeError):
                pass  # Label mismatch - metric was registered with different labels

        # Notify state change outside lock
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
            target: The target that failed (e.g., hostname, endpoint).
            error: Optional exception that caused the failure.
            preemptive: If True, this is a preemptive failure from gossip replication.
                       Preemptive failures increment failure_count but don't update
                       last_failure_time or trigger immediate state changes. This
                       biases the circuit towards opening faster if local failures occur.
        """
        old_state = None
        new_state = None
        opened_circuit = False

        with self._lock:
            circuit = self._get_or_create_circuit(target)
            old_state = circuit.state
            circuit.failure_count += 1

            # Jan 3, 2026: Preemptive failures from gossip don't update last_failure_time
            # This prevents gossip from resetting recovery timeouts
            if not preemptive:
                circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Failure in half-open: go back to open with increased backoff
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.time()
                circuit.success_count = 0
                circuit.consecutive_opens += 1  # Increase backoff for next recovery
                # Phase 15.1.8: Check for escalation
                self._escalate(circuit, target)
                opened_circuit = True
            elif circuit.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if circuit.failure_count >= self.failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = time.time()
                    circuit.consecutive_opens += 1  # Track consecutive opens
                    # Phase 15.1.8: Check for escalation
                    self._escalate(circuit, target)
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
                # Sprint 10 (Jan 3, 2026): Update escalation tier gauge
                with self._lock:
                    circuit = self._get_or_create_circuit(target)
                    PROM_ESCALATION_TIER.labels(
                        operation_type=self.operation_type, target=target
                    ).set(circuit.escalation_tier)
            except (ValueError, TypeError):
                pass  # Label mismatch - metric was registered with different labels

        # Notify state change outside lock
        if old_state is not None and new_state is not None:
            self._notify_state_change(target, old_state, new_state)

    def try_active_recovery(self, target: str) -> bool:
        """Actively probe target for recovery instead of waiting for timeout.

        Uses the configured active_recovery_probe callback to check if the
        target is healthy. If no callback is configured, uses the default
        HTTP health probe (Phase 15.1.4).

        Phase 15.1.8: Respects escalation tier timing. In escalated state,
        probing is throttled according to the tier's probe_interval.

        If probe succeeds, transitions to HALF_OPEN for gradual recovery.

        Returns:
            True if recovery probe succeeded, False otherwise.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)

            # Only try recovery on OPEN circuits
            if circuit.state != CircuitState.OPEN:
                return circuit.state == CircuitState.CLOSED

            # Phase 15.1.8: Check tier timing (replaces permanent open check)
            # Instead of blocking forever after max_consecutive_opens,
            # we use escalation tiers with increasing probe intervals
            if not self._should_probe_in_tier(circuit):
                return False

            # Update last probe time
            circuit.last_probe_at = time.time()

        # Phase 15.1.4: Use custom probe if configured, otherwise use default
        probe_func = self._active_recovery_probe or self._default_health_probe

        # Run probe outside lock
        try:
            probe_success = probe_func(target)
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

        This resets all counters including consecutive_opens and escalation.
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
            # Phase 15.1.8: Reset escalation state
            circuit.escalation_tier = 0
            circuit.escalation_entered_at = None
            circuit.last_probe_at = None

            self._notify_state_change(target, old_state, CircuitState.CLOSED)

    def is_permanently_open(self, target: str) -> bool:
        """Check if circuit is in an escalated state beyond normal backoff.

        Phase 15.1.8: With escalation tiers, circuits are never truly
        "permanently" open. They will continue attempting recovery with
        increasing intervals. This method now returns True only if the
        circuit is at the maximum escalation tier.

        Note: For backward compatibility, this still returns True when
        consecutive_opens >= max_consecutive_opens AND at max tier.
        Use force_reset() if you want to immediately reset.

        Returns:
            True if circuit is at maximum escalation tier.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(target)
            if circuit.state != CircuitState.OPEN:
                return False
            # At max tier - will still try recovery but very slowly
            return circuit.escalation_tier >= MAX_ESCALATION_TIER

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
        """Get detailed status for a target including escalation info."""
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
                # Phase 15.1.8: Escalation info
                escalation_tier=circuit.escalation_tier,
                escalation_entered_at=circuit.escalation_entered_at,
                time_until_next_probe=self._get_time_until_next_probe(circuit),
            )

    def get_all_states(self) -> dict[str, CircuitStatus]:
        """Get status for all tracked targets including escalation info."""
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
                    # Phase 15.1.8: Escalation info
                    escalation_tier=circuit.escalation_tier,
                    escalation_entered_at=circuit.escalation_entered_at,
                    time_until_next_probe=self._get_time_until_next_probe(circuit),
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


# =============================================================================
# Per-Transport Circuit Breaker (January 2026 - Phase 1 Critical Hardening)
# =============================================================================

# Registry for per-(host, transport) circuit breakers
_transport_breakers: dict[str, CircuitBreaker] = {}
_transport_breakers_lock = RLock()

# Transport-specific configurations (recovery timeouts, failure thresholds, decay TTLs)
# Jan 5, 2026: Added decay_ttl for transport-specific TTL decay
# Transports with faster recovery get shorter TTLs to avoid prolonged blocking
TRANSPORT_CONFIGS = {
    "http": {"failure_threshold": 5, "recovery_timeout": 30.0, "decay_ttl": 7200.0},      # 2 hours
    "ssh": {"failure_threshold": 3, "recovery_timeout": 60.0, "decay_ttl": 3600.0},       # 1 hour
    "rsync": {"failure_threshold": 2, "recovery_timeout": 90.0, "decay_ttl": 3600.0},     # 1 hour
    "p2p": {"failure_threshold": 3, "recovery_timeout": 45.0, "decay_ttl": 1800.0},       # 30 min
    "aria2": {"failure_threshold": 2, "recovery_timeout": 120.0, "decay_ttl": 3600.0},    # 1 hour
    "tailscale": {"failure_threshold": 3, "recovery_timeout": 60.0, "decay_ttl": 1800.0}, # 30 min (usually recovers fast)
    "base64": {"failure_threshold": 2, "recovery_timeout": 30.0, "decay_ttl": 1800.0},    # 30 min
    "relay": {"failure_threshold": 2, "recovery_timeout": 30.0, "decay_ttl": 900.0},      # 15 min (high churn)
}

# Default TTL for unknown transport types
DEFAULT_TRANSPORT_DECAY_TTL = 3600.0  # 1 hour


def get_transport_breaker(host: str, transport: str = "default") -> CircuitBreaker:
    """Get a circuit breaker for a specific (host, transport) combination.

    This enables per-transport failover: if HTTP fails for a host, SSH can
    still work because they have separate circuit breakers.

    January 2026: Created as part of P2P critical hardening (Phase 1).
    Problem: Global breaker would mark entire node as unhealthy when only
    one transport failed, defeating the 6-tier transport cascade.

    Args:
        host: Hostname or IP of the target
        transport: Transport type ("http", "ssh", "rsync", "p2p", "aria2", etc.)

    Returns:
        Circuit breaker for this (host, transport) combination

    Example:
        # HTTP fails - this breaker opens
        http_breaker = get_transport_breaker("node-1", "http")
        http_breaker.record_failure("node-1")

        # SSH still works - separate breaker
        ssh_breaker = get_transport_breaker("node-1", "ssh")
        if ssh_breaker.can_execute("node-1"):
            # Use SSH fallback
            pass
    """
    key = f"{host}:{transport}"

    with _transport_breakers_lock:
        if key not in _transport_breakers:
            config = TRANSPORT_CONFIGS.get(transport, {})
            _transport_breakers[key] = CircuitBreaker(
                failure_threshold=config.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD),
                recovery_timeout=config.get("recovery_timeout", DEFAULT_RECOVERY_TIMEOUT),
                half_open_max_calls=DEFAULT_HALF_OPEN_MAX_CALLS,
                success_threshold=1,
                operation_type=f"{transport}:{host}",
            )
        return _transport_breakers[key]


def check_transport_circuit(host: str, transport: str = "default") -> bool:
    """Check if a transport to a host is allowed (circuit not open).

    Args:
        host: Target host
        transport: Transport type

    Returns:
        True if request allowed, False if circuit is open
    """
    return get_transport_breaker(host, transport).can_execute(host)


def record_transport_success(host: str, transport: str = "default") -> None:
    """Record successful transport operation.

    Args:
        host: Target host
        transport: Transport type
    """
    get_transport_breaker(host, transport).record_success(host)


def record_transport_failure(
    host: str,
    transport: str = "default",
    error: Exception | None = None
) -> None:
    """Record failed transport operation.

    Args:
        host: Target host
        transport: Transport type
        error: Optional exception that caused the failure
    """
    get_transport_breaker(host, transport).record_failure(host, error)


def get_open_transports_for_host(host: str) -> list[str]:
    """Get list of transports with open circuits for a host.

    Args:
        host: Target host

    Returns:
        List of transport names with open circuits
    """
    open_transports = []
    with _transport_breakers_lock:
        for key, breaker in _transport_breakers.items():
            if key.startswith(f"{host}:"):
                transport = key.split(":", 1)[1]
                if breaker.get_state(host) == CircuitState.OPEN:
                    open_transports.append(transport)
    return open_transports


def get_available_transports_for_host(host: str) -> list[str]:
    """Get list of transports that can be used for a host (circuit not open).

    Args:
        host: Target host

    Returns:
        List of transport names with closed or half-open circuits
    """
    available = []
    with _transport_breakers_lock:
        for key, breaker in _transport_breakers.items():
            if key.startswith(f"{host}:"):
                transport = key.split(":", 1)[1]
                if breaker.can_execute(host):
                    available.append(transport)
    return available


def reset_transport_breakers_for_host(host: str) -> int:
    """Reset all transport circuit breakers for a host.

    Useful when a host is known to be back online after maintenance.

    Args:
        host: Target host

    Returns:
        Number of breakers reset
    """
    reset_count = 0
    with _transport_breakers_lock:
        for key, breaker in _transport_breakers.items():
            if key.startswith(f"{host}:"):
                breaker.reset(host)
                reset_count += 1
    return reset_count


def get_all_transport_breaker_states() -> dict[str, dict[str, CircuitStatus]]:
    """Get status of all transport circuit breakers grouped by host.

    Returns:
        Dict mapping host -> {transport -> CircuitStatus}
    """
    result: dict[str, dict[str, CircuitStatus]] = {}
    with _transport_breakers_lock:
        for key, breaker in _transport_breakers.items():
            host, transport = key.split(":", 1)
            if host not in result:
                result[host] = {}
            result[host][transport] = breaker.get_status(host)
    return result


def decay_transport_circuit_breakers(
    transport_ttls: dict[str, float] | None = None,
    external_alive_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Decay old transport circuit breakers with transport-specific TTLs.

    Jan 5, 2026: Added to provide faster recovery for transports that
    typically recover quickly (e.g., relay has 15 min TTL, ssh has 1 hour TTL).

    Jan 20, 2026: Added external_alive_check for gossip-integrated recovery.
    If gossip reports a node as alive while its circuit is open, we can reset
    the circuit immediately without waiting for TTL. This enables faster
    recovery (seconds vs minutes) when we have external proof the node is up.

    Args:
        transport_ttls: Optional override for per-transport TTL values.
            If not provided, uses TRANSPORT_CONFIGS decay_ttl values.
        external_alive_check: Optional callback that returns True if a host
            is known to be alive (e.g., from gossip/P2P membership). If provided
            and returns True for an open circuit's host, the circuit is reset
            immediately without waiting for TTL.

    Returns:
        Dict with "decayed" list of reset breaker keys, "checked" count,
        "external_recovered" list of circuits reset via external check,
        and per-transport breakdown.
    """
    import logging

    logger = logging.getLogger(__name__)

    now = time.time()
    decayed: list[str] = []
    external_recovered: list[str] = []
    checked = 0
    by_transport: dict[str, list[str]] = {}

    # Use provided TTLs or fall back to TRANSPORT_CONFIGS
    ttls = transport_ttls or {
        t: cfg.get("decay_ttl", DEFAULT_TRANSPORT_DECAY_TTL)
        for t, cfg in TRANSPORT_CONFIGS.items()
    }

    with _transport_breakers_lock:
        for key, breaker in _transport_breakers.items():
            checked += 1
            try:
                host, transport = key.split(":", 1)
            except ValueError:
                continue  # Invalid key format

            # Get TTL for this transport type
            ttl = ttls.get(transport, DEFAULT_TRANSPORT_DECAY_TTL)

            # Check if circuit is open
            status = breaker.get_status(host)
            if status and status.state == CircuitState.OPEN:
                # Jan 20, 2026: First check external alive status (gossip)
                # This allows immediate recovery when we know the node is alive
                if external_alive_check:
                    try:
                        if external_alive_check(host):
                            breaker.reset(host)
                            external_recovered.append(key)
                            by_transport.setdefault(transport, []).append(host)
                            logger.info(
                                f"[TransportCircuitBreaker] External recovery reset {transport}:{host} "
                                f"(gossip reports node alive)"
                            )
                            continue  # Skip TTL check, already reset
                    except Exception as e:
                        logger.debug(f"External alive check failed for {host}: {e}")

                # Fall back to TTL-based decay
                if status.opened_at:
                    elapsed = now - status.opened_at
                    if elapsed > ttl:
                        breaker.reset(host)
                        decayed.append(key)
                        by_transport.setdefault(transport, []).append(host)
                        logger.info(
                            f"[TransportCircuitBreaker] TTL decay reset {transport}:{host} "
                            f"(was open for {elapsed:.0f}s, TTL={ttl:.0f}s)"
                        )

    total_reset = len(decayed) + len(external_recovered)
    if total_reset > 0:
        logger.info(
            f"[TransportCircuitBreaker] Reset {total_reset}/{checked} circuits "
            f"(TTL decay: {len(decayed)}, external recovery: {len(external_recovered)}) "
            f"across transports: {list(by_transport.keys())}"
        )

    return {
        "decayed": decayed,
        "external_recovered": external_recovered,
        "checked": checked,
        "by_transport": by_transport,
        "ttls_used": ttls,
    }


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

    def health_check(self) -> "HealthCheckResult":
        """Health check for monitoring integration.

        Returns:
            HealthCheckResult with circuit breaker health status.
        """
        # Lazy import to avoid circular dependency
        from app.coordination.protocols import HealthCheckResult

        with self._lock:
            total_breakers = len(self._breakers)
            total_circuits = 0
            open_circuits = 0
            half_open_circuits = 0

            for breaker in self._breakers.values():
                states = breaker.get_all_states()
                for status in states.values():
                    total_circuits += 1
                    if status.state == CircuitState.OPEN:
                        open_circuits += 1
                    elif status.state == CircuitState.HALF_OPEN:
                        half_open_circuits += 1

            # Unhealthy if more than 50% of circuits are open
            open_ratio = open_circuits / total_circuits if total_circuits > 0 else 0.0
            is_healthy = open_ratio < 0.5

            if is_healthy:
                message = "OK"
            else:
                message = f"High open circuit ratio: {open_ratio:.1%}"

            return HealthCheckResult(
                healthy=is_healthy,
                message=message,
                details={
                    "total_breakers": total_breakers,
                    "total_circuits": total_circuits,
                    "open_circuits": open_circuits,
                    "half_open_circuits": half_open_circuits,
                    "closed_circuits": total_circuits - open_circuits - half_open_circuits,
                    "open_ratio": round(open_ratio, 3),
                },
            )


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


# =============================================================================
# Phase 15.1.4: Periodic Recovery Probing (December 2025)
# =============================================================================

_recovery_probe_task: asyncio.Task | None = None
_recovery_probe_interval: float = 30.0  # Probe open circuits every 30 seconds


async def _periodic_recovery_probe_loop() -> None:
    """Background loop that periodically probes open circuits for recovery.

    Phase 15.1.4: This addresses the issue where circuits that opened due to
    transient failures would never recover because try_active_recovery()
    was only called explicitly.

    The loop:
    1. Gets all open/half-open circuits from the registry
    2. Attempts recovery probes on each
    3. Logs results for monitoring

    Runs until cancelled via stop_recovery_probing().
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(
        f"[circuit_breaker] Starting periodic recovery probing "
        f"(interval={_recovery_probe_interval}s)"
    )

    while True:
        try:
            await asyncio.sleep(_recovery_probe_interval)

            registry = get_circuit_registry()
            open_circuits = registry.get_all_open_circuits()

            if not open_circuits:
                continue

            recovered = 0
            probed = 0

            for op_type, statuses in open_circuits.items():
                breaker = registry.get_breaker(op_type)
                for target, status in statuses.items():
                    probed += 1

                    # Skip permanently open circuits - they need manual reset
                    if breaker.is_permanently_open(target):
                        logger.debug(
                            f"[circuit_breaker] {op_type}/{target}: permanently open, "
                            f"skipping (needs force_reset)"
                        )
                        continue

                    if breaker.try_active_recovery(target):
                        recovered += 1
                        logger.info(
                            f"[circuit_breaker] {op_type}/{target}: "
                            f"recovered via health probe"
                        )

            if recovered > 0:
                logger.info(
                    f"[circuit_breaker] Recovery probing: {recovered}/{probed} circuits recovered"
                )

        except asyncio.CancelledError:
            logger.info("[circuit_breaker] Stopping periodic recovery probing")
            break
        except Exception as e:
            logger.warning(f"[circuit_breaker] Recovery probe error: {e}")
            # Continue probing despite errors


def start_recovery_probing(interval: float = 30.0) -> asyncio.Task | None:
    """Start the periodic recovery probing background task.

    Args:
        interval: Seconds between probe attempts (default: 30)

    Returns:
        The background task, or None if already running or no event loop.
    """
    global _recovery_probe_task, _recovery_probe_interval
    _recovery_probe_interval = interval

    if _recovery_probe_task is not None and not _recovery_probe_task.done():
        return _recovery_probe_task

    try:
        loop = asyncio.get_running_loop()
        _recovery_probe_task = loop.create_task(_periodic_recovery_probe_loop())
        return _recovery_probe_task
    except RuntimeError:
        # No event loop running
        return None


def stop_recovery_probing() -> None:
    """Stop the periodic recovery probing background task."""
    global _recovery_probe_task
    if _recovery_probe_task is not None and not _recovery_probe_task.done():
        _recovery_probe_task.cancel()
        _recovery_probe_task = None


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

"""Transport-Aware Circuit Breaker - Downgrade transports, don't exclude nodes.

January 2026: Created as part of distributed architecture Phase 3.

Problem:
    When one transport fails (e.g., Tailscale times out), the node gets excluded
    across ALL transports. Escalation tiers then lock nodes out for hours.

Solution:
    Track failures per-node, per-transport. When a transport fails:
    1. Open circuit for THAT transport only
    2. Fall back to next available transport
    3. Only exclude node when ALL transports are broken

Transport priority order:
    1. Tailscale (100.x.x.x) - Best for mesh, encrypted
    2. SSH (direct IP) - Reliable, works behind NAT
    3. HTTP (direct) - Simplest, may be blocked
    4. Relay (via leader) - Last resort for NAT-blocked nodes

Usage:
    from app.coordination.transport_circuit_breaker import (
        TransportAwareCircuitBreaker,
        get_transport_circuit_breaker,
    )

    breaker = get_transport_circuit_breaker()

    # Execute with automatic transport fallback
    result = await breaker.execute_with_fallback(
        node="lambda-gh200-1",
        operation=my_operation,
        transports=["tailscale", "ssh", "http"],
    )

    # Or check available transports before operation
    available = breaker.get_available_transports("lambda-gh200-1")
    if available:
        transport = available[0]  # Use best available
        try:
            result = await my_operation(transport)
            breaker.record_success("lambda-gh200-1", transport)
        except Exception:
            breaker.record_failure("lambda-gh200-1", transport)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TransportType(Enum):
    """Available transport types in priority order."""

    TAILSCALE = "tailscale"
    SSH = "ssh"
    HTTP = "http"
    RELAY = "relay"

    @classmethod
    def priority_order(cls) -> list["TransportType"]:
        """Get transports in priority order."""
        return [cls.TAILSCALE, cls.SSH, cls.HTTP, cls.RELAY]


class CircuitState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, skip operations
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class TransportCircuitState:
    """State for a single transport to a single node."""

    transport: TransportType
    node_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    opened_at: float = 0.0
    half_open_at: float = 0.0

    # Thresholds
    failure_threshold: int = 3  # Open after N failures
    success_threshold: int = 2  # Close after N successes in half-open
    open_duration_seconds: float = 60.0  # Stay open for this long
    half_open_timeout_seconds: float = 30.0  # Half-open probe timeout

    def should_open(self) -> bool:
        """Check if circuit should open."""
        return self.failure_count >= self.failure_threshold

    def should_close(self) -> bool:
        """Check if circuit should close (in half-open state)."""
        return (
            self.state == CircuitState.HALF_OPEN
            and self.success_count >= self.success_threshold
        )

    def should_half_open(self) -> bool:
        """Check if circuit should transition to half-open."""
        if self.state != CircuitState.OPEN:
            return False
        return time.time() - self.opened_at >= self.open_duration_seconds

    def record_failure(self) -> None:
        """Record a failure for this transport."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()

        if self.should_open() and self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
            logger.info(
                f"Circuit opened for {self.node_id}/{self.transport.value} "
                f"after {self.failure_count} failures"
            )

    def record_success(self) -> None:
        """Record a success for this transport."""
        self.success_count += 1
        self.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN and self.should_close():
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info(
                f"Circuit closed for {self.node_id}/{self.transport.value} "
                f"after {self.success_count} successes"
            )
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def can_execute(self) -> bool:
        """Check if operations can be executed on this transport."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.should_half_open():
                self.state = CircuitState.HALF_OPEN
                self.half_open_at = time.time()
                self.success_count = 0
                logger.debug(
                    f"Circuit half-open for {self.node_id}/{self.transport.value}"
                )
                return True  # Allow probe
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited probes in half-open
            return True

        return False


@dataclass
class TransportAwareCircuitBreakerConfig:
    """Configuration for transport-aware circuit breaker."""

    # Per-transport failure thresholds
    failure_threshold: int = 3
    success_threshold: int = 2
    open_duration_seconds: float = 60.0
    half_open_timeout_seconds: float = 30.0

    # Node exclusion settings
    exclude_after_all_transports_fail: bool = True
    node_exclusion_duration_seconds: float = 300.0  # 5 minutes

    # Transport-specific timeouts
    transport_timeouts: dict[str, float] = field(
        default_factory=lambda: {
            "tailscale": 15.0,
            "ssh": 30.0,
            "http": 10.0,
            "relay": 45.0,
        }
    )


class AllTransportsFailed(Exception):
    """Raised when all transports have failed for a node."""

    def __init__(self, node_id: str, errors: list[tuple[str, Exception]]):
        self.node_id = node_id
        self.errors = errors
        super().__init__(
            f"All transports failed for {node_id}: "
            + ", ".join(f"{t}: {e}" for t, e in errors)
        )


class TransportAwareCircuitBreaker:
    """Circuit breaker that downgrades transports, not nodes.

    Instead of excluding a node when connections fail, this tracks
    failures per-transport and falls back to working transports.
    """

    def __init__(self, config: TransportAwareCircuitBreakerConfig | None = None):
        """Initialize transport-aware circuit breaker."""
        self.config = config or TransportAwareCircuitBreakerConfig()
        self._lock = RLock()

        # Per-node, per-transport tracking
        # {node_id: {transport: TransportCircuitState}}
        self._transport_states: dict[str, dict[TransportType, TransportCircuitState]] = {}

        # Node-level exclusion tracking
        self._excluded_nodes: dict[str, float] = {}  # node_id -> excluded_until

    def _get_or_create_state(
        self, node_id: str, transport: TransportType
    ) -> TransportCircuitState:
        """Get or create circuit state for node/transport."""
        with self._lock:
            if node_id not in self._transport_states:
                self._transport_states[node_id] = {}

            if transport not in self._transport_states[node_id]:
                self._transport_states[node_id][transport] = TransportCircuitState(
                    transport=transport,
                    node_id=node_id,
                    failure_threshold=self.config.failure_threshold,
                    success_threshold=self.config.success_threshold,
                    open_duration_seconds=self.config.open_duration_seconds,
                    half_open_timeout_seconds=self.config.half_open_timeout_seconds,
                )

            return self._transport_states[node_id][transport]

    def can_use_transport(self, node_id: str, transport: TransportType | str) -> bool:
        """Check if a specific transport can be used for a node.

        Args:
            node_id: Target node
            transport: Transport type (string or enum)

        Returns:
            True if transport circuit is closed or half-open
        """
        if isinstance(transport, str):
            try:
                transport = TransportType(transport)
            except ValueError:
                return False

        # Check node-level exclusion first
        if self._is_node_excluded(node_id):
            return False

        state = self._get_or_create_state(node_id, transport)
        return state.can_execute()

    def get_available_transports(self, node_id: str) -> list[TransportType]:
        """Get transports that aren't circuit-broken for a node.

        Returns:
            List of available transports in priority order
        """
        if self._is_node_excluded(node_id):
            return []

        available = []
        for transport in TransportType.priority_order():
            state = self._get_or_create_state(node_id, transport)
            if state.can_execute():
                available.append(transport)

        return available

    def record_failure(self, node_id: str, transport: TransportType | str) -> None:
        """Record failure for specific transport only.

        Args:
            node_id: Target node
            transport: Transport that failed
        """
        if isinstance(transport, str):
            try:
                transport = TransportType(transport)
            except ValueError:
                return

        with self._lock:
            state = self._get_or_create_state(node_id, transport)
            state.record_failure()

            # Check if all transports are now broken
            if self._should_exclude_node(node_id):
                self._exclude_node(
                    node_id, f"All transports failed for {node_id}"
                )

    def record_success(self, node_id: str, transport: TransportType | str) -> None:
        """Record success for specific transport.

        Args:
            node_id: Target node
            transport: Transport that succeeded
        """
        if isinstance(transport, str):
            try:
                transport = TransportType(transport)
            except ValueError:
                return

        with self._lock:
            state = self._get_or_create_state(node_id, transport)
            state.record_success()

            # Clear node exclusion on any transport success
            if node_id in self._excluded_nodes:
                del self._excluded_nodes[node_id]
                logger.info(f"Node {node_id} un-excluded after transport success")

    def _should_exclude_node(self, node_id: str) -> bool:
        """Only exclude if no transports available."""
        if not self.config.exclude_after_all_transports_fail:
            return False

        available = self.get_available_transports(node_id)
        return len(available) == 0

    def _exclude_node(self, node_id: str, reason: str) -> None:
        """Exclude a node temporarily."""
        exclude_until = time.time() + self.config.node_exclusion_duration_seconds
        self._excluded_nodes[node_id] = exclude_until
        logger.warning(f"Node {node_id} excluded: {reason}")

        # Emit event
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.HOST_OFFLINE,
                {
                    "node_id": node_id,
                    "reason": reason,
                    "excluded_until": exclude_until,
                    "source": "transport_circuit_breaker",
                },
            )
        except ImportError:
            pass

    def _is_node_excluded(self, node_id: str) -> bool:
        """Check if node is currently excluded."""
        if node_id not in self._excluded_nodes:
            return False

        exclude_until = self._excluded_nodes[node_id]
        if time.time() >= exclude_until:
            # Exclusion expired
            del self._excluded_nodes[node_id]
            logger.info(f"Node {node_id} exclusion expired")
            return False

        return True

    async def execute_with_fallback(
        self,
        node_id: str,
        operation: Callable[[TransportType], Any],
        transports: list[str | TransportType] | None = None,
    ) -> Any:
        """Execute with transport fallback - only exclude if ALL fail.

        Args:
            node_id: Target node
            operation: Async callable that takes transport type
            transports: Optional list of transports to try (default: all)

        Returns:
            Result from successful operation

        Raises:
            AllTransportsFailed: If all transports fail
        """
        if transports is None:
            transport_list = TransportType.priority_order()
        else:
            transport_list = []
            for t in transports:
                if isinstance(t, str):
                    try:
                        transport_list.append(TransportType(t))
                    except ValueError:
                        continue
                else:
                    transport_list.append(t)

        errors: list[tuple[str, Exception]] = []

        for transport in transport_list:
            if not self.can_use_transport(node_id, transport):
                continue

            timeout = self.config.transport_timeouts.get(
                transport.value, 30.0
            )

            try:
                result = await asyncio.wait_for(
                    operation(transport),
                    timeout=timeout,
                )
                self.record_success(node_id, transport)
                return result

            except asyncio.TimeoutError as e:
                self.record_failure(node_id, transport)
                errors.append((transport.value, e))
                logger.debug(
                    f"Transport {transport.value} timed out for {node_id}"
                )
                continue

            except Exception as e:
                self.record_failure(node_id, transport)
                errors.append((transport.value, e))
                logger.debug(
                    f"Transport {transport.value} failed for {node_id}: {e}"
                )
                continue

        # All transports exhausted
        if errors and self.config.exclude_after_all_transports_fail:
            self._exclude_node(node_id, f"All transports failed: {errors}")

        raise AllTransportsFailed(node_id, errors)

    def get_node_status(self, node_id: str) -> dict[str, Any]:
        """Get detailed status for a node."""
        excluded = self._is_node_excluded(node_id)
        available = self.get_available_transports(node_id)

        transport_status = {}
        for transport in TransportType.priority_order():
            state = self._get_or_create_state(node_id, transport)
            transport_status[transport.value] = {
                "state": state.state.value,
                "failure_count": state.failure_count,
                "success_count": state.success_count,
                "can_execute": state.can_execute(),
            }

        return {
            "node_id": node_id,
            "excluded": excluded,
            "excluded_until": self._excluded_nodes.get(node_id),
            "available_transports": [t.value for t in available],
            "transport_status": transport_status,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all circuit breaker states."""
        with self._lock:
            nodes_tracked = len(self._transport_states)
            nodes_excluded = len(self._excluded_nodes)

            open_circuits = 0
            half_open_circuits = 0
            for node_states in self._transport_states.values():
                for state in node_states.values():
                    if state.state == CircuitState.OPEN:
                        open_circuits += 1
                    elif state.state == CircuitState.HALF_OPEN:
                        half_open_circuits += 1

            return {
                "nodes_tracked": nodes_tracked,
                "nodes_excluded": nodes_excluded,
                "open_circuits": open_circuits,
                "half_open_circuits": half_open_circuits,
                "excluded_nodes": list(self._excluded_nodes.keys()),
            }


# Singleton instance
_instance: TransportAwareCircuitBreaker | None = None


def get_transport_circuit_breaker() -> TransportAwareCircuitBreaker:
    """Get the singleton transport-aware circuit breaker."""
    global _instance
    if _instance is None:
        _instance = TransportAwareCircuitBreaker()
    return _instance


def reset_transport_circuit_breaker() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None

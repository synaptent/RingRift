"""Per-Node Circuit Breaker for Health Checks.

December 2025: Provides per-node granularity for circuit breakers in health monitoring.

Problem: The existing per-component circuit breaker causes cascading failures:
- Circuit breaker trips for "health_check" component
- ALL nodes affected when one node is slow
- False positives cascade across cluster

Solution: Per-node circuit breakers isolate failures to individual nodes.
A slow node only affects its own circuit, not the entire cluster.

Usage:
    from app.coordination.node_circuit_breaker import (
        NodeCircuitBreakerRegistry,
        get_node_circuit_breaker,
    )

    # Get circuit breaker for a specific node
    breaker = get_node_circuit_breaker()

    # Check before health check
    if breaker.can_check(node_id):
        try:
            result = await check_node_health(node_id)
            breaker.record_success(node_id)
        except (ConnectionError, TimeoutError):
            breaker.record_failure(node_id)
    else:
        # Circuit open for this node - skip check, use cached state

    # Get all node states
    summary = breaker.get_summary()
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable

from app.config.coordination_defaults import CircuitBreakerDefaults

logger = logging.getLogger(__name__)

# January 2026 Sprint 10: Import event emission for circuit breaker monitoring
try:
    from app.coordination.event_router import publish_sync
    from app.distributed.data_events import DataEventType
    _HAS_EVENTS = True
except ImportError:
    publish_sync = None  # type: ignore[assignment]
    DataEventType = None  # type: ignore[misc, assignment]
    _HAS_EVENTS = False

__all__ = [
    "NodeCircuitBreaker",
    "NodeCircuitBreakerRegistry",
    "NodeCircuitState",
    "NodeCircuitStatus",
    "get_node_circuit_breaker",
    "get_node_circuit_registry",
    # Cluster-level (January 2026)
    "ClusterCircuitBreaker",
    "ClusterCircuitConfig",
    "ClusterCircuitStatus",
    "ClusterDegradationState",
    "get_cluster_circuit_breaker",
]


class NodeCircuitState(Enum):
    """Circuit breaker states for node health checks."""

    CLOSED = "closed"  # Normal operation - checks allowed
    OPEN = "open"  # Too many failures - checks blocked
    HALF_OPEN = "half_open"  # Testing recovery - limited checks


@dataclass
class NodeCircuitStatus:
    """Status of a circuit for a specific node."""

    node_id: str
    state: NodeCircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    opened_at: float | None
    recovery_timeout: float

    @property
    def time_until_recovery(self) -> float:
        """Seconds until circuit transitions to half-open."""
        if self.state != NodeCircuitState.OPEN or self.opened_at is None:
            return 0.0
        elapsed = time.time() - self.opened_at
        return max(0.0, self.recovery_timeout - elapsed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "opened_at": self.opened_at,
            "recovery_timeout": self.recovery_timeout,
            "time_until_recovery": self.time_until_recovery,
        }


@dataclass
class _NodeCircuitData:
    """Internal circuit data per node."""

    state: NodeCircuitState = NodeCircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    opened_at: float | None = None
    half_open_at: float | None = None
    # January 2026 Sprint 10: Per-circuit jitter offset (set when circuit opens)
    # This ensures consistent jitter per-circuit until reset
    jitter_offset: float = 0.0
    # January 2026 Sprint 17.4: Track when circuit first opened for TTL decay
    # Prevents circuits from staying OPEN indefinitely after transient failures
    first_opened_at: float | None = None


@dataclass
class NodeCircuitConfig:
    """Configuration for per-node circuit breakers.

    Jan 2, 2026: Consolidated to use CircuitBreakerDefaults as fallback.
    Environment variables still take precedence for runtime tuning.
    """

    # Number of consecutive failures to open circuit
    # Uses CircuitBreakerDefaults.FAILURE_THRESHOLD as fallback
    failure_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "RINGRIFT_P2P_NODE_CIRCUIT_FAILURE_THRESHOLD",
                str(CircuitBreakerDefaults.FAILURE_THRESHOLD),
            )
        )
    )

    # Seconds to wait before testing recovery
    # Uses CircuitBreakerDefaults.RECOVERY_TIMEOUT as fallback
    recovery_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "RINGRIFT_P2P_NODE_CIRCUIT_RECOVERY_TIMEOUT",
                str(CircuitBreakerDefaults.RECOVERY_TIMEOUT),
            )
        )
    )

    # Successes needed in half-open to close circuit
    success_threshold: int = CircuitBreakerDefaults.HALF_OPEN_MAX_CALLS

    # Enable event emission on state changes
    emit_events: bool = True

    # January 2026 Sprint 10: Jitter factor to prevent thundering herd
    # When multiple nodes reset at the same time, jitter spreads them out
    # Default 15% jitter = ±7.5% variation from recovery_timeout
    jitter_factor: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "RINGRIFT_P2P_NODE_CIRCUIT_JITTER_FACTOR",
                "0.15",
            )
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")


class NodeCircuitBreaker:
    """Per-node circuit breaker for health check isolation.

    Each node has its own circuit that can trip independently.
    This prevents a slow/failing node from affecting health checks
    for other nodes in the cluster.

    Features:
    - Per-node failure tracking
    - Configurable thresholds
    - Half-open state for gradual recovery
    - Event callbacks for state changes
    """

    def __init__(
        self,
        config: NodeCircuitConfig | None = None,
        on_state_change: Callable[[str, NodeCircuitState, NodeCircuitState], None] | None = None,
    ):
        """Initialize the per-node circuit breaker.

        Args:
            config: Circuit breaker configuration
            on_state_change: Optional callback for state transitions.
                            Called with (node_id, old_state, new_state).
        """
        self.config = config or NodeCircuitConfig()
        self._on_state_change = on_state_change
        self._circuits: dict[str, _NodeCircuitData] = {}
        self._lock = RLock()

    def _get_or_create_circuit(self, node_id: str) -> _NodeCircuitData:
        """Get or create circuit data for a node."""
        if node_id not in self._circuits:
            self._circuits[node_id] = _NodeCircuitData()
        return self._circuits[node_id]

    def _compute_jittered_timeout(self, circuit: _NodeCircuitData) -> float:
        """Compute recovery timeout with jitter applied.

        January 2026 Sprint 10: Prevents thundering herd when multiple
        node circuits reset at the same time. Uses per-circuit jitter
        offset for consistency (same offset until circuit resets).

        Returns:
            Recovery timeout in seconds with jitter applied.
        """
        base = self.config.recovery_timeout
        if self.config.jitter_factor <= 0:
            return base
        # Apply jitter: base * (1 + jitter_offset), where offset is in [-jitter_factor, +jitter_factor]
        return base * (1.0 + circuit.jitter_offset)

    # January 2026 Sprint 17.4: Maximum time a circuit can stay OPEN before forced reset
    # Prevents circuits from staying OPEN indefinitely after transient failures
    MAX_CIRCUIT_OPEN_DURATION = 4 * 3600  # 4 hours

    def _check_recovery(self, circuit: _NodeCircuitData) -> None:
        """Check if circuit should transition to half-open or force reset via TTL decay."""
        now = time.time()

        # January 2026 Sprint 17.4: TTL decay - force reset after MAX_CIRCUIT_OPEN_DURATION
        # This prevents circuits from staying OPEN indefinitely after transient failures
        if circuit.state == NodeCircuitState.OPEN and circuit.first_opened_at:
            total_time_open = now - circuit.first_opened_at
            if total_time_open > self.MAX_CIRCUIT_OPEN_DURATION:
                logger.warning(
                    f"[NodeCircuitBreaker] Circuit TTL expired after {total_time_open:.0f}s, "
                    f"forcing CLOSED"
                )
                circuit.state = NodeCircuitState.CLOSED
                circuit.failure_count = 0
                circuit.first_opened_at = None
                circuit.opened_at = None
                circuit.half_open_at = None
                return

        if circuit.state == NodeCircuitState.OPEN:
            jittered_timeout = self._compute_jittered_timeout(circuit)
            if circuit.opened_at and (now - circuit.opened_at) >= jittered_timeout:
                circuit.state = NodeCircuitState.HALF_OPEN
                circuit.half_open_at = now

    def _notify_state_change(
        self, node_id: str, old_state: NodeCircuitState, new_state: NodeCircuitState
    ) -> None:
        """Notify callback and emit events on state change.

        January 2026 Sprint 10: Added event emission for circuit breaker monitoring.
        Enables alerting infrastructure to track circuit health cluster-wide.
        """
        if old_state == new_state:
            return

        # Call optional callback
        if self._on_state_change:
            try:
                self._on_state_change(node_id, old_state, new_state)
            except Exception as e:
                logger.warning(f"[NodeCircuitBreaker] State change callback error: {e}")

        # January 2026 Sprint 10: Emit circuit breaker events for monitoring
        if not self.config.emit_events or not _HAS_EVENTS:
            return

        try:
            event_type = None
            if new_state == NodeCircuitState.OPEN:
                event_type = DataEventType.CIRCUIT_BREAKER_OPENED
            elif new_state == NodeCircuitState.CLOSED:
                event_type = DataEventType.CIRCUIT_BREAKER_CLOSED
            elif new_state == NodeCircuitState.HALF_OPEN:
                event_type = DataEventType.CIRCUIT_BREAKER_HALF_OPEN

            if event_type and publish_sync:
                circuit = self._circuits.get(node_id)
                payload = {
                    "node_id": node_id,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": circuit.failure_count if circuit else 0,
                    "recovery_timeout": self.config.recovery_timeout,
                    "timestamp": time.time(),
                }
                publish_sync(event_type, payload)
                logger.debug(
                    f"[NodeCircuitBreaker] Emitted {event_type.value} for {node_id}"
                )

            # Check if too many circuits are open (threshold alert)
            self._check_threshold_alert()
        except Exception as e:
            logger.warning(f"[NodeCircuitBreaker] Event emission error: {e}")

    def _check_threshold_alert(self) -> None:
        """Check if too many circuits are open and emit threshold alert.

        January 2026 Sprint 10: Emits CIRCUIT_BREAKER_THRESHOLD when >20% of
        tracked nodes have open circuits. This indicates potential cluster-wide
        health issues requiring investigation.
        """
        if not _HAS_EVENTS or not publish_sync:
            return

        total_nodes = len(self._circuits)
        if total_nodes < 3:
            # Too few nodes to meaningfully calculate threshold
            return

        open_count = sum(
            1 for c in self._circuits.values()
            if c.state == NodeCircuitState.OPEN
        )

        open_ratio = open_count / total_nodes
        threshold_ratio = 0.20  # 20% of nodes with open circuits triggers alert

        if open_ratio >= threshold_ratio:
            try:
                payload = {
                    "open_count": open_count,
                    "total_nodes": total_nodes,
                    "open_ratio": round(open_ratio, 3),
                    "threshold_ratio": threshold_ratio,
                    "open_nodes": [
                        node_id for node_id, c in self._circuits.items()
                        if c.state == NodeCircuitState.OPEN
                    ],
                    "timestamp": time.time(),
                }
                publish_sync(DataEventType.CIRCUIT_BREAKER_THRESHOLD, payload)
                logger.warning(
                    f"[NodeCircuitBreaker] Threshold alert: {open_count}/{total_nodes} "
                    f"circuits open ({open_ratio:.1%})"
                )
            except Exception as e:
                logger.warning(f"[NodeCircuitBreaker] Threshold alert emission error: {e}")

    def can_check(self, node_id: str) -> bool:
        """Check if a health check is allowed for this node.

        Returns True if:
        - Circuit is CLOSED (normal operation)
        - Circuit is HALF_OPEN (testing recovery)

        Returns False if:
        - Circuit is OPEN (blocking checks)
        """
        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            self._check_recovery(circuit)

            if circuit.state == NodeCircuitState.CLOSED:
                return True
            elif circuit.state == NodeCircuitState.HALF_OPEN:
                return True
            else:  # OPEN
                return False

    def record_success(self, node_id: str) -> None:
        """Record a successful health check for a node."""
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            old_state = circuit.state
            circuit.success_count += 1
            circuit.last_success_time = time.time()

            if circuit.state == NodeCircuitState.HALF_OPEN:
                if circuit.success_count >= self.config.success_threshold:
                    circuit.state = NodeCircuitState.CLOSED
                    circuit.failure_count = 0
                    circuit.opened_at = None
                    circuit.half_open_at = None
                    # January 2026 Sprint 17.4: Clear TTL tracking when circuit closes
                    circuit.first_opened_at = None
                    logger.info(f"[NodeCircuitBreaker] Circuit CLOSED for {node_id}")
            elif circuit.state == NodeCircuitState.CLOSED:
                circuit.failure_count = 0

            new_state = circuit.state

        if old_state is not None and new_state is not None:
            self._notify_state_change(node_id, old_state, new_state)

    def record_failure(self, node_id: str, error: Exception | None = None) -> None:
        """Record a failed health check for a node."""
        old_state = None
        new_state = None

        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            old_state = circuit.state
            circuit.failure_count += 1
            circuit.last_failure_time = time.time()

            if circuit.state == NodeCircuitState.HALF_OPEN:
                circuit.state = NodeCircuitState.OPEN
                circuit.opened_at = time.time()
                circuit.success_count = 0
                # January 2026 Sprint 10: Set jitter offset to prevent thundering herd
                circuit.jitter_offset = self.config.jitter_factor * (2 * random.random() - 1)
                # January 2026 Sprint 17.4: Track first open time for TTL decay
                # Don't reset first_opened_at if already set (circuit was already open before half-open)
                if circuit.first_opened_at is None:
                    circuit.first_opened_at = circuit.opened_at
                logger.warning(
                    f"[NodeCircuitBreaker] Circuit OPEN for {node_id} (half-open test failed)"
                )
            elif circuit.state == NodeCircuitState.CLOSED:
                if circuit.failure_count >= self.config.failure_threshold:
                    circuit.state = NodeCircuitState.OPEN
                    circuit.opened_at = time.time()
                    # January 2026 Sprint 10: Set jitter offset to prevent thundering herd
                    circuit.jitter_offset = self.config.jitter_factor * (2 * random.random() - 1)
                    # January 2026 Sprint 17.4: Track first open time for TTL decay
                    circuit.first_opened_at = circuit.opened_at
                    logger.warning(
                        f"[NodeCircuitBreaker] Circuit OPEN for {node_id} "
                        f"({circuit.failure_count} failures)"
                    )

            new_state = circuit.state

        if old_state is not None and new_state is not None:
            self._notify_state_change(node_id, old_state, new_state)

    def get_state(self, node_id: str) -> NodeCircuitState:
        """Get current circuit state for a node."""
        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            self._check_recovery(circuit)
            return circuit.state

    def get_status(self, node_id: str) -> NodeCircuitStatus:
        """Get detailed circuit status for a node."""
        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            self._check_recovery(circuit)
            return NodeCircuitStatus(
                node_id=node_id,
                state=circuit.state,
                failure_count=circuit.failure_count,
                success_count=circuit.success_count,
                last_failure_time=circuit.last_failure_time,
                last_success_time=circuit.last_success_time,
                opened_at=circuit.opened_at,
                recovery_timeout=self.config.recovery_timeout,
            )

    def get_all_states(self) -> dict[str, NodeCircuitStatus]:
        """Get status for all tracked nodes."""
        with self._lock:
            result = {}
            for node_id in self._circuits:
                result[node_id] = self.get_status(node_id)
            return result

    def get_open_circuits(self) -> list[str]:
        """Get list of nodes with open circuits."""
        with self._lock:
            result = []
            for node_id, circuit in self._circuits.items():
                self._check_recovery(circuit)
                if circuit.state == NodeCircuitState.OPEN:
                    result.append(node_id)
            return result

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all circuit states."""
        with self._lock:
            closed_count = 0
            open_count = 0
            half_open_count = 0

            for circuit in self._circuits.values():
                self._check_recovery(circuit)
                if circuit.state == NodeCircuitState.CLOSED:
                    closed_count += 1
                elif circuit.state == NodeCircuitState.OPEN:
                    open_count += 1
                else:
                    half_open_count += 1

            return {
                "total_nodes": len(self._circuits),
                "closed": closed_count,
                "open": open_count,
                "half_open": half_open_count,
                "open_nodes": self.get_open_circuits(),
            }

    def reset(self, node_id: str) -> None:
        """Reset circuit for a node to CLOSED state."""
        with self._lock:
            if node_id in self._circuits:
                old_state = self._circuits[node_id].state
                self._circuits[node_id] = _NodeCircuitData()
                self._notify_state_change(node_id, old_state, NodeCircuitState.CLOSED)
                logger.info(f"[NodeCircuitBreaker] Reset circuit for {node_id}")

    def reset_all(self) -> None:
        """Reset all circuits to CLOSED state."""
        with self._lock:
            for node_id in list(self._circuits.keys()):
                old_state = self._circuits[node_id].state
                self._circuits[node_id] = _NodeCircuitData()
                self._notify_state_change(node_id, old_state, NodeCircuitState.CLOSED)
            logger.info(f"[NodeCircuitBreaker] Reset all circuits ({len(self._circuits)} nodes)")

    def force_open(self, node_id: str) -> None:
        """Force circuit open for a node (manual intervention)."""
        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            old_state = circuit.state
            circuit.state = NodeCircuitState.OPEN
            circuit.opened_at = time.time()
            self._notify_state_change(node_id, old_state, NodeCircuitState.OPEN)

    def force_close(self, node_id: str) -> None:
        """Force circuit closed for a node (manual intervention)."""
        with self._lock:
            circuit = self._get_or_create_circuit(node_id)
            old_state = circuit.state
            circuit.state = NodeCircuitState.CLOSED
            circuit.failure_count = 0
            circuit.opened_at = None
            circuit.half_open_at = None
            self._notify_state_change(node_id, old_state, NodeCircuitState.CLOSED)

    def decay_old_circuits(self, ttl_seconds: float = 21600.0) -> dict[str, list[str]]:
        """Automatically reset node circuits that have been open for too long.

        Prevents stuck circuits from blocking nodes indefinitely.

        Args:
            ttl_seconds: Max time to keep circuit OPEN (default: 21600 = 6 hours)

        Returns:
            Dict with "decayed" list of reset node IDs and "checked" count
        """
        now = time.time()
        decayed: list[str] = []
        checked = 0

        with self._lock:
            for node_id, circuit in self._circuits.items():
                checked += 1
                if circuit.state == NodeCircuitState.OPEN and circuit.opened_at:
                    elapsed = now - circuit.opened_at
                    if elapsed > ttl_seconds:
                        old_state = circuit.state
                        circuit.state = NodeCircuitState.CLOSED
                        circuit.failure_count = 0
                        circuit.opened_at = None
                        circuit.half_open_at = None
                        decayed.append(node_id)
                        self._notify_state_change(node_id, old_state, NodeCircuitState.CLOSED)
                        logger.info(
                            f"[NodeCircuitBreaker] TTL decay reset circuit for node {node_id} "
                            f"(was open for {elapsed:.0f}s)"
                        )

        return {"decayed": decayed, "checked": checked}


# =============================================================================
# Registry and Singleton Access
# =============================================================================


class NodeCircuitBreakerRegistry:
    """Registry for per-operation-type node circuit breakers.

    Different operation types (health_check, gossip, sync) may have
    different circuit breaker configurations.
    """

    _instance: NodeCircuitBreakerRegistry | None = None
    _lock = RLock()

    def __init__(self):
        self._breakers: dict[str, NodeCircuitBreaker] = {}
        self._default_config = NodeCircuitConfig()

    @classmethod
    def get_instance(cls) -> NodeCircuitBreakerRegistry:
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
        operation_type: str = "health_check",
        config: NodeCircuitConfig | None = None,
    ) -> NodeCircuitBreaker:
        """Get or create a node circuit breaker for an operation type."""
        with self._lock:
            if operation_type not in self._breakers:
                self._breakers[operation_type] = NodeCircuitBreaker(
                    config=config or self._default_config
                )
            return self._breakers[operation_type]

    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get summary for all operation types."""
        with self._lock:
            return {
                op_type: breaker.get_summary()
                for op_type, breaker in self._breakers.items()
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

        with self._lock:
            for op_type, breaker in self._breakers.items():
                result = breaker.decay_old_circuits(ttl_seconds)
                results[op_type] = result
                total_decayed += len(result.get("decayed", []))
                total_checked += result.get("checked", 0)

        if total_decayed > 0:
            logger.info(
                f"[NodeCircuitBreakerRegistry] TTL decay: "
                f"reset {total_decayed}/{total_checked} circuits across {len(results)} breakers"
            )

        return {
            "by_operation": results,
            "total_decayed": total_decayed,
            "total_checked": total_checked,
        }


# Module-level singleton access
_registry: NodeCircuitBreakerRegistry | None = None


def get_node_circuit_breaker(operation_type: str = "health_check") -> NodeCircuitBreaker:
    """Get the node circuit breaker for an operation type.

    This is the main entry point for per-node circuit breaking in health checks.

    Args:
        operation_type: Type of operation (default: "health_check")

    Returns:
        NodeCircuitBreaker instance
    """
    global _registry
    if _registry is None:
        _registry = NodeCircuitBreakerRegistry.get_instance()
    return _registry.get_breaker(operation_type)


def get_node_circuit_registry() -> NodeCircuitBreakerRegistry:
    """Get the global node circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = NodeCircuitBreakerRegistry.get_instance()
    return _registry


# =============================================================================
# Cluster-Level Circuit Breaker (January 2026)
# =============================================================================


class ClusterDegradationState(Enum):
    """Cluster-level health states based on failure ratio."""

    HEALTHY = "healthy"  # < 20% nodes failing
    DEGRADED = "degraded"  # 20-40% nodes failing - reduced operations
    CRITICAL = "critical"  # > 40% nodes failing - pause new work


@dataclass
class ClusterCircuitConfig:
    """Configuration for cluster-level circuit breaker.

    January 2026: Prevents cascade failures by tracking cluster-wide failure ratio.
    When too many nodes fail simultaneously, pause operations to prevent amplification.
    """

    degraded_threshold: float = 0.20  # 20% open = degraded
    critical_threshold: float = 0.40  # 40% open = critical (pause work)
    min_nodes_for_tracking: int = 5  # Don't trigger with < 5 nodes
    recovery_check_interval: float = 30.0  # Check recovery every 30s
    auto_recovery_threshold: float = 0.15  # Recover when < 15% open


@dataclass
class ClusterCircuitStatus:
    """Status of cluster-level circuit breaker."""

    state: ClusterDegradationState
    total_nodes: int
    open_nodes: int
    failure_ratio: float
    degraded_since: float | None = None
    critical_since: float | None = None
    open_node_ids: list[str] = field(default_factory=list)


class ClusterCircuitBreaker:
    """Cluster-level circuit breaker for cascade prevention.

    January 2026: Tracks overall cluster health and triggers degradation
    mode when too many nodes fail simultaneously. This prevents:
    - Retry storms against failing nodes
    - Resource exhaustion from repeated timeouts
    - Cascade failures spreading through the cluster

    Usage:
        from app.coordination.node_circuit_breaker import get_cluster_circuit_breaker

        cluster_cb = get_cluster_circuit_breaker()

        # Check before scheduling new work
        if cluster_cb.should_pause_new_work():
            logger.warning("Cluster degraded - pausing new work scheduling")
            return

        # Get current cluster health
        status = cluster_cb.get_status()
        if status.state == ClusterDegradationState.DEGRADED:
            # Reduce concurrent operations
            max_concurrent = max_concurrent // 2
    """

    _instance: ClusterCircuitBreaker | None = None
    _lock = RLock()

    def __init__(
        self,
        config: ClusterCircuitConfig | None = None,
        node_breaker: NodeCircuitBreaker | None = None,
    ):
        self.config = config or ClusterCircuitConfig()
        self._node_breaker = node_breaker
        self._state = ClusterDegradationState.HEALTHY
        self._degraded_since: float | None = None
        self._critical_since: float | None = None
        self._last_state_change: float = time.time()
        self._state_change_callbacks: list[Callable[[ClusterDegradationState, ClusterDegradationState], None]] = []

    @classmethod
    def get_instance(
        cls,
        config: ClusterCircuitConfig | None = None,
    ) -> ClusterCircuitBreaker:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _get_node_breaker(self) -> NodeCircuitBreaker:
        """Get node circuit breaker (lazy initialization)."""
        if self._node_breaker is None:
            self._node_breaker = get_node_circuit_breaker()
        return self._node_breaker

    def _calculate_failure_ratio(self) -> tuple[float, int, int, list[str]]:
        """Calculate current cluster failure ratio.

        Returns:
            (failure_ratio, total_nodes, open_nodes, open_node_ids)
        """
        summary = self._get_node_breaker().get_summary()
        total = summary["total_nodes"]
        open_count = summary["open"]
        open_nodes = summary.get("open_nodes", [])

        if total < self.config.min_nodes_for_tracking:
            return 0.0, total, open_count, open_nodes

        ratio = open_count / total if total > 0 else 0.0
        return ratio, total, open_count, open_nodes

    def update_state(self) -> ClusterDegradationState:
        """Update cluster state based on current failure ratio.

        Call this periodically (e.g., after health checks) to update state.
        """
        ratio, total, open_count, open_nodes = self._calculate_failure_ratio()
        old_state = self._state
        now = time.time()

        if total < self.config.min_nodes_for_tracking:
            # Not enough nodes to track - stay healthy
            new_state = ClusterDegradationState.HEALTHY
        elif ratio >= self.config.critical_threshold:
            new_state = ClusterDegradationState.CRITICAL
            if self._critical_since is None:
                self._critical_since = now
        elif ratio >= self.config.degraded_threshold:
            new_state = ClusterDegradationState.DEGRADED
            if self._degraded_since is None:
                self._degraded_since = now
            self._critical_since = None
        elif ratio < self.config.auto_recovery_threshold:
            new_state = ClusterDegradationState.HEALTHY
            self._degraded_since = None
            self._critical_since = None
        else:
            # Hysteresis: stay in current state if between recovery and degraded thresholds
            new_state = self._state

        if new_state != old_state:
            self._state = new_state
            self._last_state_change = now
            self._notify_state_change(old_state, new_state, ratio, open_nodes)

        return self._state

    def _notify_state_change(
        self,
        old_state: ClusterDegradationState,
        new_state: ClusterDegradationState,
        ratio: float,
        open_nodes: list[str],
    ) -> None:
        """Notify callbacks and emit events on state change."""
        logger.warning(
            f"[ClusterCircuitBreaker] State change: {old_state.value} → {new_state.value} "
            f"(failure_ratio={ratio:.1%}, open_nodes={len(open_nodes)})"
        )

        # Emit event if available
        if _HAS_EVENTS and publish_sync is not None:
            try:
                # Use CLUSTER_HEALTH_CHANGED event type
                publish_sync(
                    DataEventType.CLUSTER_HEALTH_CHANGED,
                    {
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "failure_ratio": ratio,
                        "open_nodes": open_nodes,
                        "is_degraded": new_state != ClusterDegradationState.HEALTHY,
                        "is_critical": new_state == ClusterDegradationState.CRITICAL,
                    },
                )
            except Exception as e:
                logger.debug(f"[ClusterCircuitBreaker] Event emission failed: {e}")

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"[ClusterCircuitBreaker] Callback error: {e}")

    def get_state(self) -> ClusterDegradationState:
        """Get current cluster degradation state."""
        return self._state

    def get_status(self) -> ClusterCircuitStatus:
        """Get detailed cluster circuit status."""
        ratio, total, open_count, open_nodes = self._calculate_failure_ratio()
        return ClusterCircuitStatus(
            state=self._state,
            total_nodes=total,
            open_nodes=open_count,
            failure_ratio=ratio,
            degraded_since=self._degraded_since,
            critical_since=self._critical_since,
            open_node_ids=open_nodes,
        )

    def should_pause_new_work(self) -> bool:
        """Check if new work should be paused due to cluster degradation.

        Returns True when cluster is in CRITICAL state (>40% failure).
        """
        self.update_state()
        return self._state == ClusterDegradationState.CRITICAL

    def should_reduce_concurrency(self) -> bool:
        """Check if concurrency should be reduced due to degradation.

        Returns True when cluster is DEGRADED or CRITICAL.
        """
        self.update_state()
        return self._state != ClusterDegradationState.HEALTHY

    def get_recommended_concurrency_factor(self) -> float:
        """Get recommended concurrency reduction factor.

        Returns:
            1.0 for healthy, 0.5 for degraded, 0.1 for critical
        """
        self.update_state()
        if self._state == ClusterDegradationState.HEALTHY:
            return 1.0
        elif self._state == ClusterDegradationState.DEGRADED:
            return 0.5
        else:
            return 0.1

    def add_state_change_callback(
        self,
        callback: Callable[[ClusterDegradationState, ClusterDegradationState], None],
    ) -> None:
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

    def force_healthy(self) -> None:
        """Force cluster to healthy state (manual intervention)."""
        old_state = self._state
        self._state = ClusterDegradationState.HEALTHY
        self._degraded_since = None
        self._critical_since = None
        if old_state != self._state:
            logger.info("[ClusterCircuitBreaker] Forced to HEALTHY state")


# Module-level singleton access
_cluster_breaker: ClusterCircuitBreaker | None = None


def get_cluster_circuit_breaker() -> ClusterCircuitBreaker:
    """Get the cluster-level circuit breaker singleton.

    January 2026: Use this to check cluster health before scheduling work.

    Example:
        cluster_cb = get_cluster_circuit_breaker()
        if cluster_cb.should_pause_new_work():
            return  # Don't schedule - cluster degraded
    """
    global _cluster_breaker
    if _cluster_breaker is None:
        _cluster_breaker = ClusterCircuitBreaker.get_instance()
    return _cluster_breaker



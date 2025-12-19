"""NodeHealthOrchestrator - Unified monitoring for node health events (December 2025).

This module provides centralized monitoring of node health events across the cluster.
It subscribes to health-related events and provides:

1. Node health state tracking across the cluster
2. Health history per node
3. Recovery coordination and tracking
4. Integration with alerting systems

Event Integration:
- Subscribes to NODE_UNHEALTHY: Tracks when nodes become unhealthy
- Subscribes to NODE_RECOVERED: Tracks node recovery
- Subscribes to HEALTH_CHECK_PASSED: Tracks successful health checks
- Subscribes to HEALTH_CHECK_FAILED: Tracks failed health checks
- Subscribes to RECOVERY_INITIATED: Tracks auto-recovery starts
- Subscribes to RECOVERY_COMPLETED: Tracks successful recoveries
- Subscribes to RECOVERY_FAILED: Tracks failed recovery attempts
- Subscribes to RESOURCE_CONSTRAINT: Tracks resource pressure

Usage:
    from app.monitoring.node_health_orchestrator import (
        NodeHealthOrchestrator,
        wire_health_events,
        get_health_orchestrator,
    )

    # Wire health events to orchestrator
    orchestrator = wire_health_events()

    # Get cluster health status
    status = orchestrator.get_status()
    print(f"Healthy nodes: {status['healthy_nodes']}/{status['total_nodes']}")

    # Get node health history
    history = orchestrator.get_node_history("gh200-a")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeHealthState(Enum):
    """Health state of a node."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


@dataclass
class NodeHealth:
    """Health state of a single node."""

    node_name: str
    node_ip: str = ""
    state: NodeHealthState = NodeHealthState.UNKNOWN
    last_health_check_time: float = 0.0
    last_state_change_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_error: str = ""
    recovery_attempts: int = 0
    last_recovery_time: float = 0.0

    # Resource metrics (from latest check)
    gpu_utilization: float = 0.0
    memory_used_percent: float = 0.0
    disk_used_percent: float = 0.0

    # Constraints (current active constraints)
    active_constraints: List[str] = field(default_factory=list)


@dataclass
class ClusterHealthStats:
    """Aggregate cluster health statistics."""

    total_nodes: int = 0
    healthy_nodes: int = 0
    unhealthy_nodes: int = 0
    recovering_nodes: int = 0
    unknown_nodes: int = 0
    total_health_checks: int = 0
    total_failures: int = 0
    total_recoveries: int = 0
    total_recovery_failures: int = 0
    last_activity_time: float = 0.0


class NodeHealthOrchestrator:
    """Orchestrates node health monitoring across the cluster.

    Subscribes to health events and maintains a unified view of node health
    states across all cluster nodes.
    """

    def __init__(
        self,
        node_stale_threshold_seconds: float = 300.0,  # 5 minutes
        max_history_per_node: int = 100,
        recovery_cooldown_seconds: float = 60.0,
    ):
        """Initialize NodeHealthOrchestrator.

        Args:
            node_stale_threshold_seconds: Time after which node is considered stale
            max_history_per_node: Maximum history entries per node
            recovery_cooldown_seconds: Minimum time between recovery attempts
        """
        self.node_stale_threshold_seconds = node_stale_threshold_seconds
        self.max_history_per_node = max_history_per_node
        self.recovery_cooldown_seconds = recovery_cooldown_seconds

        # Node tracking
        self._nodes: Dict[str, NodeHealth] = {}
        self._node_history: Dict[str, List[Dict[str, Any]]] = {}

        # Statistics
        self._total_health_checks: int = 0
        self._total_failures: int = 0
        self._total_recoveries: int = 0
        self._total_recovery_failures: int = 0
        self._subscribed: bool = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to health-related events from the event bus.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # Subscribe to health events
            bus.subscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy)
            bus.subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered)
            bus.subscribe(DataEventType.HEALTH_CHECK_PASSED, self._on_health_check_passed)
            bus.subscribe(DataEventType.HEALTH_CHECK_FAILED, self._on_health_check_failed)
            bus.subscribe(DataEventType.RESOURCE_CONSTRAINT, self._on_resource_constraint)
            bus.subscribe(DataEventType.RECOVERY_INITIATED, self._on_recovery_initiated)
            bus.subscribe(DataEventType.RECOVERY_COMPLETED, self._on_recovery_completed)
            bus.subscribe(DataEventType.RECOVERY_FAILED, self._on_recovery_failed)

            self._subscribed = True
            logger.info("[NodeHealthOrchestrator] Subscribed to health events")
            return True

        except Exception as e:
            logger.warning(f"[NodeHealthOrchestrator] Failed to subscribe to events: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from health events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy)
            bus.unsubscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered)
            bus.unsubscribe(DataEventType.HEALTH_CHECK_PASSED, self._on_health_check_passed)
            bus.unsubscribe(DataEventType.HEALTH_CHECK_FAILED, self._on_health_check_failed)
            bus.unsubscribe(DataEventType.RESOURCE_CONSTRAINT, self._on_resource_constraint)
            bus.unsubscribe(DataEventType.RECOVERY_INITIATED, self._on_recovery_initiated)
            bus.unsubscribe(DataEventType.RECOVERY_COMPLETED, self._on_recovery_completed)
            bus.unsubscribe(DataEventType.RECOVERY_FAILED, self._on_recovery_failed)
            self._subscribed = False

        except Exception:
            pass

    def _get_or_create_node(self, node_name: str, node_ip: str = "") -> NodeHealth:
        """Get or create a node health record."""
        if node_name not in self._nodes:
            self._nodes[node_name] = NodeHealth(
                node_name=node_name,
                node_ip=node_ip,
            )
        elif node_ip and not self._nodes[node_name].node_ip:
            self._nodes[node_name].node_ip = node_ip
        return self._nodes[node_name]

    def _on_node_unhealthy(self, event: Any) -> None:
        """Handle NODE_UNHEALTHY event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", "unknown")
        node_ip = payload.get("node_ip", "")
        error = payload.get("error", "")

        logger.warning(f"[NodeHealthOrchestrator] Node unhealthy: {node_name} - {error}")

        node = self._get_or_create_node(node_name, node_ip)
        node.state = NodeHealthState.UNHEALTHY
        node.last_state_change_time = time.time()
        node.consecutive_failures += 1
        node.consecutive_successes = 0
        node.last_error = error

        # Update resource metrics if provided
        if "gpu_utilization" in payload:
            node.gpu_utilization = payload["gpu_utilization"]
        if "disk_used_percent" in payload:
            node.disk_used_percent = payload["disk_used_percent"]

        self._total_failures += 1

        self._add_to_history(node_name, "unhealthy", {
            "error": error,
            "consecutive_failures": node.consecutive_failures,
            "timestamp": time.time(),
        })

    def _on_node_recovered(self, event: Any) -> None:
        """Handle NODE_RECOVERED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", "unknown")
        node_ip = payload.get("node_ip", "")

        logger.info(f"[NodeHealthOrchestrator] Node recovered: {node_name}")

        node = self._get_or_create_node(node_name, node_ip)
        node.state = NodeHealthState.HEALTHY
        node.last_state_change_time = time.time()
        node.consecutive_failures = 0
        node.consecutive_successes += 1
        node.last_error = ""
        node.active_constraints = []

        self._add_to_history(node_name, "recovered", {
            "timestamp": time.time(),
        })

    def _on_health_check_passed(self, event: Any) -> None:
        """Handle HEALTH_CHECK_PASSED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", payload.get("host", "unknown"))
        node_ip = payload.get("node_ip", "")

        node = self._get_or_create_node(node_name, node_ip)
        node.last_health_check_time = time.time()
        node.consecutive_successes += 1

        # Only change state to healthy if not in recovering state
        if node.state != NodeHealthState.RECOVERING:
            if node.state == NodeHealthState.UNHEALTHY:
                # Require multiple consecutive successes to mark healthy
                if node.consecutive_successes >= 3:
                    node.state = NodeHealthState.HEALTHY
                    node.consecutive_failures = 0
                    node.last_state_change_time = time.time()
            elif node.state == NodeHealthState.UNKNOWN:
                node.state = NodeHealthState.HEALTHY
                node.last_state_change_time = time.time()

        self._total_health_checks += 1

    def _on_health_check_failed(self, event: Any) -> None:
        """Handle HEALTH_CHECK_FAILED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", payload.get("host", "unknown"))
        node_ip = payload.get("node_ip", "")
        error = payload.get("error", "Health check failed")

        node = self._get_or_create_node(node_name, node_ip)
        node.last_health_check_time = time.time()
        node.consecutive_failures += 1
        node.consecutive_successes = 0
        node.last_error = error

        # Mark unhealthy after multiple consecutive failures
        if node.consecutive_failures >= 2:
            node.state = NodeHealthState.UNHEALTHY
            node.last_state_change_time = time.time()

        self._total_health_checks += 1
        self._total_failures += 1

    def _on_resource_constraint(self, event: Any) -> None:
        """Handle RESOURCE_CONSTRAINT event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", payload.get("host", "unknown"))
        constraint = payload.get("constraint", payload.get("alert", "unknown constraint"))

        node = self._get_or_create_node(node_name)

        # Track active constraints
        if constraint not in node.active_constraints:
            node.active_constraints.append(constraint)

        # Update resource metrics if provided
        if "memory_percent" in payload:
            node.memory_used_percent = payload["memory_percent"]
        if "disk_percent" in payload:
            node.disk_used_percent = payload["disk_percent"]
        if "gpu_utilization" in payload:
            node.gpu_utilization = payload["gpu_utilization"]

        logger.debug(f"[NodeHealthOrchestrator] Resource constraint on {node_name}: {constraint}")

        self._add_to_history(node_name, "resource_constraint", {
            "constraint": constraint,
            "timestamp": time.time(),
        })

    def _on_recovery_initiated(self, event: Any) -> None:
        """Handle RECOVERY_INITIATED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", payload.get("host", "unknown"))

        logger.info(f"[NodeHealthOrchestrator] Recovery initiated for: {node_name}")

        node = self._get_or_create_node(node_name)
        node.state = NodeHealthState.RECOVERING
        node.last_state_change_time = time.time()
        node.recovery_attempts += 1
        node.last_recovery_time = time.time()

        self._add_to_history(node_name, "recovery_initiated", {
            "recovery_attempt": node.recovery_attempts,
            "timestamp": time.time(),
        })

    def _on_recovery_completed(self, event: Any) -> None:
        """Handle RECOVERY_COMPLETED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", payload.get("host", "unknown"))

        logger.info(f"[NodeHealthOrchestrator] Recovery completed for: {node_name}")

        node = self._get_or_create_node(node_name)
        node.state = NodeHealthState.HEALTHY
        node.last_state_change_time = time.time()
        node.consecutive_failures = 0
        node.consecutive_successes = 1
        node.last_error = ""
        node.active_constraints = []

        self._total_recoveries += 1

        self._add_to_history(node_name, "recovery_completed", {
            "timestamp": time.time(),
        })

    def _on_recovery_failed(self, event: Any) -> None:
        """Handle RECOVERY_FAILED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        node_name = payload.get("node_name", payload.get("host", "unknown"))
        error = payload.get("error", "Recovery failed")

        logger.error(f"[NodeHealthOrchestrator] Recovery failed for: {node_name} - {error}")

        node = self._get_or_create_node(node_name)
        node.state = NodeHealthState.UNHEALTHY
        node.last_state_change_time = time.time()
        node.last_error = error

        self._total_recovery_failures += 1

        self._add_to_history(node_name, "recovery_failed", {
            "error": error,
            "recovery_attempt": node.recovery_attempts,
            "timestamp": time.time(),
        })

    def _add_to_history(self, node_name: str, event_type: str, data: Dict[str, Any]) -> None:
        """Add entry to node history."""
        if node_name not in self._node_history:
            self._node_history[node_name] = []

        history = self._node_history[node_name]
        history.append({
            "event_type": event_type,
            **data,
        })

        # Trim history if needed
        if len(history) > self.max_history_per_node:
            self._node_history[node_name] = history[-self.max_history_per_node:]

    def get_node_health(self, node_name: str) -> Optional[NodeHealth]:
        """Get health state of a specific node."""
        return self._nodes.get(node_name)

    def get_healthy_nodes(self) -> List[NodeHealth]:
        """Get all healthy nodes."""
        return [n for n in self._nodes.values() if n.state == NodeHealthState.HEALTHY]

    def get_unhealthy_nodes(self) -> List[NodeHealth]:
        """Get all unhealthy nodes."""
        return [n for n in self._nodes.values() if n.state == NodeHealthState.UNHEALTHY]

    def get_recovering_nodes(self) -> List[NodeHealth]:
        """Get all nodes currently in recovery."""
        return [n for n in self._nodes.values() if n.state == NodeHealthState.RECOVERING]

    def get_node_history(self, node_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get node history."""
        if node_name:
            return {node_name: self._node_history.get(node_name, [])}
        return dict(self._node_history)

    def get_stats(self) -> ClusterHealthStats:
        """Get aggregate cluster health statistics."""
        healthy = len(self.get_healthy_nodes())
        unhealthy = len(self.get_unhealthy_nodes())
        recovering = len(self.get_recovering_nodes())
        unknown = len([n for n in self._nodes.values() if n.state == NodeHealthState.UNKNOWN])

        last_activity = 0.0
        for node in self._nodes.values():
            last_activity = max(
                last_activity,
                node.last_health_check_time,
                node.last_state_change_time,
            )

        return ClusterHealthStats(
            total_nodes=len(self._nodes),
            healthy_nodes=healthy,
            unhealthy_nodes=unhealthy,
            recovering_nodes=recovering,
            unknown_nodes=unknown,
            total_health_checks=self._total_health_checks,
            total_failures=self._total_failures,
            total_recoveries=self._total_recoveries,
            total_recovery_failures=self._total_recovery_failures,
            last_activity_time=last_activity,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status for monitoring."""
        stats = self.get_stats()

        return {
            "subscribed": self._subscribed,
            "total_nodes": stats.total_nodes,
            "healthy_nodes": stats.healthy_nodes,
            "unhealthy_nodes": stats.unhealthy_nodes,
            "recovering_nodes": stats.recovering_nodes,
            "unknown_nodes": stats.unknown_nodes,
            "total_health_checks": stats.total_health_checks,
            "total_failures": stats.total_failures,
            "total_recoveries": stats.total_recoveries,
            "total_recovery_failures": stats.total_recovery_failures,
            "last_activity_time": stats.last_activity_time,
            "node_names": list(self._nodes.keys()),
            "unhealthy_node_names": [n.node_name for n in self.get_unhealthy_nodes()],
        }

    def should_attempt_recovery(self, node_name: str) -> bool:
        """Check if recovery should be attempted for a node.

        Considers recovery cooldown and current state.
        """
        node = self._nodes.get(node_name)
        if not node:
            return False

        # Only attempt recovery for unhealthy nodes
        if node.state != NodeHealthState.UNHEALTHY:
            return False

        # Respect recovery cooldown
        time_since_last_recovery = time.time() - node.last_recovery_time
        if time_since_last_recovery < self.recovery_cooldown_seconds:
            return False

        return True


# Singleton instance
_health_orchestrator: Optional[NodeHealthOrchestrator] = None


def wire_health_events(
    node_stale_threshold_seconds: float = 300.0,
) -> NodeHealthOrchestrator:
    """Wire health events to the orchestrator.

    This enables centralized monitoring of node health state across
    the cluster.

    Args:
        node_stale_threshold_seconds: Time after which node is stale

    Returns:
        NodeHealthOrchestrator instance
    """
    global _health_orchestrator

    if _health_orchestrator is None:
        _health_orchestrator = NodeHealthOrchestrator(
            node_stale_threshold_seconds=node_stale_threshold_seconds,
        )
        _health_orchestrator.subscribe_to_events()

        logger.info(
            f"[wire_health_events] Health events wired to orchestrator "
            f"(stale_threshold={node_stale_threshold_seconds}s)"
        )

    return _health_orchestrator


def get_health_orchestrator() -> Optional[NodeHealthOrchestrator]:
    """Get the global health orchestrator if configured."""
    return _health_orchestrator


def reset_health_orchestrator() -> None:
    """Reset the health orchestrator singleton (for testing)."""
    global _health_orchestrator
    if _health_orchestrator:
        _health_orchestrator.unsubscribe()
    _health_orchestrator = None


__all__ = [
    "NodeHealthOrchestrator",
    "NodeHealth",
    "NodeHealthState",
    "ClusterHealthStats",
    "wire_health_events",
    "get_health_orchestrator",
    "reset_health_orchestrator",
]

"""Canonical node status definitions for cluster management.

December 2025 Phase 6: Consolidated from 5 duplicate definitions:
- node_health_monitor.py: NodeStatus enum
- idle_resource_daemon.py: NodeStatus dataclass
- cluster_status_monitor.py: NodeStatus dataclass
- cluster_watchdog_daemon.py: WatchdogNodeStatus dataclass
- p2p_integration.py: P2PNodeStatus dataclass

This module provides the canonical definitions. Other modules should import from here
and provide backward-compatible aliases if needed.

Migration Status:
- idle_resource_daemon.py: Import from here, alias to old name
- cluster_status_monitor.py: Import from here, alias to old name
- cluster_watchdog_daemon.py: Keep WatchdogNodeStatus, import NodeHealthState
- p2p_integration.py: Keep P2PNodeStatus, import NodeHealthState
- node_health_monitor.py: Import NodeHealthState, alias to NodeStatus

Example migration:
    # In idle_resource_daemon.py:
    from app.coordination.node_status import NodeMonitoringStatus as NodeStatus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.config.ports import P2P_DEFAULT_PORT


class NodeHealthState(str, Enum):
    """Node health state for cluster monitoring.

    Canonical enum for node health classification.
    December 2025: Consolidated from health_check_orchestrator.py and
    node_health_monitor.py to single source of truth.
    """

    HEALTHY = "healthy"  # Node responding normally
    DEGRADED = "degraded"  # 1-2 consecutive failures, still usable
    UNHEALTHY = "unhealthy"  # 3+ failures, avoid for new work
    EVICTED = "evicted"  # 5+ failures, no tasks assigned
    UNKNOWN = "unknown"  # Initial state, no health data yet
    OFFLINE = "offline"  # Node confirmed offline
    PROVIDER_DOWN = "provider_down"  # Provider reports node as down
    RETIRED = "retired"  # Manually removed from cluster


@dataclass
class NodeMonitoringStatus:
    """Unified node status for all monitoring modules.

    Canonical dataclass combining fields from:
    - idle_resource_daemon.NodeStatus
    - cluster_status_monitor.NodeStatus
    - cluster_watchdog_daemon.WatchdogNodeStatus
    - p2p_integration.P2PNodeStatus

    All fields are optional with sensible defaults to support
    different use cases (health monitoring, resource tracking, P2P status).
    """

    # Identity (required)
    node_id: str

    # Network (all have host, some have port)
    host: str = ""
    port: int = P2P_DEFAULT_PORT  # Default from app.config.ports

    # Health state
    health_state: NodeHealthState = NodeHealthState.UNKNOWN
    is_reachable: bool = False
    response_time_ms: float = 0.0
    consecutive_failures: int = 0
    last_check_time: datetime | None = None
    last_success_time: datetime | None = None

    # GPU resources
    gpu_utilization: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_free_gb: float = 0.0
    gpu_type: str = ""

    # Provider info
    provider: str = "unknown"  # vast, runpod, vultr, nebius, hetzner, lambda
    ssh_cmd: str = ""  # SSH command for this node

    # Role and capabilities
    role: str = "worker"  # coordinator, worker, voter
    is_coordinator: bool = False
    is_voter: bool = False

    # Workload info
    running_jobs: list[str] = field(default_factory=list)
    pending_jobs: int = 0
    completed_jobs: int = 0

    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if node is in a healthy state."""
        return self.health_state in (NodeHealthState.HEALTHY, NodeHealthState.DEGRADED)

    @property
    def is_available(self) -> bool:
        """Check if node is available for new work."""
        return self.is_reachable and self.health_state not in (
            NodeHealthState.EVICTED,
            NodeHealthState.OFFLINE,
        )

    @property
    def gpu_memory_used_percent(self) -> float:
        """Calculate GPU memory usage percentage."""
        if self.gpu_memory_total_gb <= 0:
            return 0.0
        return (self.gpu_memory_used_gb / self.gpu_memory_total_gb) * 100

    @property
    def is_gpu_node(self) -> bool:
        """Check if this is a GPU-capable node."""
        return self.gpu_memory_total_gb > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "health_state": self.health_state.value,
            "is_reachable": self.is_reachable,
            "response_time_ms": self.response_time_ms,
            "consecutive_failures": self.consecutive_failures,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_free_gb": self.gpu_memory_free_gb,
            "gpu_type": self.gpu_type,
            "provider": self.provider,
            "role": self.role,
            "is_coordinator": self.is_coordinator,
            "is_voter": self.is_voter,
            "running_jobs": self.running_jobs,
            "pending_jobs": self.pending_jobs,
            "completed_jobs": self.completed_jobs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeMonitoringStatus:
        """Create from dictionary."""
        # Parse health state
        health_state_str = data.get("health_state", "unknown")
        try:
            health_state = NodeHealthState(health_state_str)
        except ValueError:
            health_state = NodeHealthState.UNKNOWN

        # Parse timestamps
        last_check = data.get("last_check_time")
        if isinstance(last_check, str):
            last_check = datetime.fromisoformat(last_check)

        last_success = data.get("last_success_time")
        if isinstance(last_success, str):
            last_success = datetime.fromisoformat(last_success)

        return cls(
            node_id=data.get("node_id", ""),
            host=data.get("host", ""),
            port=data.get("port", P2P_DEFAULT_PORT),
            health_state=health_state,
            is_reachable=data.get("is_reachable", False),
            response_time_ms=data.get("response_time_ms", 0.0),
            consecutive_failures=data.get("consecutive_failures", 0),
            last_check_time=last_check,
            last_success_time=last_success,
            gpu_utilization=data.get("gpu_utilization", 0.0),
            gpu_memory_total_gb=data.get("gpu_memory_total_gb", 0.0),
            gpu_memory_used_gb=data.get("gpu_memory_used_gb", 0.0),
            gpu_memory_free_gb=data.get("gpu_memory_free_gb", 0.0),
            gpu_type=data.get("gpu_type", ""),
            provider=data.get("provider", "unknown"),
            role=data.get("role", "worker"),
            is_coordinator=data.get("is_coordinator", False),
            is_voter=data.get("is_voter", False),
            running_jobs=data.get("running_jobs", []),
            pending_jobs=data.get("pending_jobs", 0),
            completed_jobs=data.get("completed_jobs", 0),
            metadata=data.get("metadata", {}),
        )


# Backward-compatible aliases
# These allow gradual migration from old class names
NodeStatus = NodeMonitoringStatus  # For idle_resource_daemon.py
ClusterNodeStatus = NodeMonitoringStatus  # For cluster_status_monitor.py


def get_node_status(node_id: str, host: str = "", **kwargs) -> NodeMonitoringStatus:
    """Factory function for creating NodeMonitoringStatus.

    Convenience function for quick status creation.

    Args:
        node_id: Node identifier
        host: Node hostname or IP
        **kwargs: Additional fields to set

    Returns:
        NodeMonitoringStatus instance
    """
    return NodeMonitoringStatus(node_id=node_id, host=host, **kwargs)

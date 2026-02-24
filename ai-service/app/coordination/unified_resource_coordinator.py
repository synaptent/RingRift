"""Unified Resource Coordinator - Single decision point for all resource questions.

December 2025 - Consolidates 5-6 separate resource management systems into
a single coordination point for consistent, system-wide resource decisions.

This module aggregates:
- ResourceOptimizer (PID control for dynamic scaling)
- ResourceTargets (goal-based resource allocation)
- ResourceMonitoringCoordinator (real-time metrics)
- BandwidthManager (network throughput limits)
- QueueMonitor (backpressure detection)

Usage:
    from app.coordination.unified_resource_coordinator import (
        get_unified_resource_coordinator,
        UnifiedResourceCoordinator,
    )

    coordinator = get_unified_resource_coordinator()

    # Check if a task can spawn
    allowed, reason = coordinator.can_spawn_task("selfplay", "runpod-h100")
    if not allowed:
        print(f"Cannot spawn: {reason}")

    # Get recommended resource allocation
    allocation = coordinator.get_recommended_allocation("hex8_2p")
    print(f"Recommended GPU hours: {allocation.gpu_hours}")

    # Report resource usage
    coordinator.report_usage("runpod-h100", cpu=0.5, gpu=0.8, memory=0.6)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Canonical types (December 2025 consolidation)
from app.config.thresholds import DISK_CRITICAL_PERCENT
from app.coordination.types import BackpressureLevel, TaskType

logger = logging.getLogger(__name__)


# BackpressureLevel and TaskType imported from app.coordination.types
# Additional local TaskType values for GPU/CPU CMAES if needed
class _LocalTaskType(str, Enum):
    """Local task type extensions (for backward compatibility)."""
    GPU_CMAES = "gpu_cmaes"
    CPU_CMAES = "cpu_cmaes"


@dataclass
class ResourceAllocation:
    """Recommended resource allocation for a config."""
    config_key: str
    gpu_hours: float = 1.0
    cpu_hours: float = 1.0
    memory_gb: float = 8.0
    priority: int = 50
    max_concurrent_tasks: int = 4
    recommended_nodes: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class NodeResourceStatus:
    """Current resource status of a node."""
    node_id: str
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    bandwidth_mbps: float = 0.0
    last_updated: float = field(default_factory=time.time)
    is_healthy: bool = True
    active_tasks: int = 0


class UnifiedResourceCoordinator:
    """Single coordination point for all resource decisions.

    Provides unified admission control, resource allocation recommendations,
    and backpressure detection across the cluster.
    """

    def __init__(self):
        """Initialize the unified resource coordinator."""
        self._node_status: dict[str, NodeResourceStatus] = {}
        self._backpressure_level = BackpressureLevel.NONE
        self._config_allocations: dict[str, ResourceAllocation] = {}

        # Lazy-loaded subsystems
        self._optimizer = None
        self._targets = None
        self._monitoring = None
        self._bandwidth = None
        self._queue_monitor = None

        # Thresholds
        self._cpu_threshold = 0.85
        self._gpu_threshold = 0.90
        self._memory_threshold = 0.85
        self._disk_threshold = DISK_CRITICAL_PERCENT / 100.0  # Convert percent to fraction

        # Statistics
        self._decisions_made = 0
        self._tasks_allowed = 0
        self._tasks_denied = 0

        logger.info("[UnifiedResourceCoordinator] Initialized")

    def _get_optimizer(self):
        """Lazy-load resource optimizer."""
        if self._optimizer is None:
            try:
                from app.coordination.resource_optimizer import get_resource_optimizer
                self._optimizer = get_resource_optimizer()
            except ImportError:
                pass
        return self._optimizer

    def _get_monitoring(self):
        """Lazy-load resource monitoring coordinator."""
        if self._monitoring is None:
            try:
                from app.coordination.resource_monitoring_coordinator import (
                    get_resource_coordinator,
                )
                self._monitoring = get_resource_coordinator()
            except ImportError:
                pass
        return self._monitoring

    def _get_bandwidth_manager(self):
        """Lazy-load bandwidth manager."""
        if self._bandwidth is None:
            try:
                from app.coordination.sync_bandwidth import get_bandwidth_manager
                self._bandwidth = get_bandwidth_manager()
            except ImportError:
                pass
        return self._bandwidth

    def _get_queue_monitor(self):
        """Lazy-load queue monitor."""
        if self._queue_monitor is None:
            try:
                from app.coordination.work_queue import get_work_queue
                self._queue_monitor = get_work_queue()
            except ImportError:
                pass
        return self._queue_monitor

    def can_spawn_task(
        self,
        task_type: str | TaskType,
        node: str,
        config_key: str = "",
    ) -> tuple[bool, str]:
        """Unified admission control for spawning tasks.

        Checks all resource systems to determine if a task can be spawned
        on the specified node.

        Args:
            task_type: Type of task to spawn
            node: Target node ID
            config_key: Optional config identifier (e.g., "hex8_2p")

        Returns:
            Tuple of (allowed, reason) where reason explains the decision
        """
        self._decisions_made += 1

        # Convert string to enum if needed
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                pass

        # Check 1: Backpressure
        if self._backpressure_level in (BackpressureLevel.HIGH, BackpressureLevel.CRITICAL):
            self._tasks_denied += 1
            return False, f"backpressure_{self._backpressure_level.value}"

        # Check 2: Node health
        node_status = self._node_status.get(node)
        if node_status and not node_status.is_healthy:
            self._tasks_denied += 1
            return False, "node_unhealthy"

        # Check 3: Resource capacity
        if node_status:
            if node_status.cpu_percent > self._cpu_threshold:
                self._tasks_denied += 1
                return False, f"cpu_overloaded_{node_status.cpu_percent:.1%}"

            if node_status.gpu_percent > self._gpu_threshold:
                self._tasks_denied += 1
                return False, f"gpu_overloaded_{node_status.gpu_percent:.1%}"

            if node_status.memory_percent > self._memory_threshold:
                self._tasks_denied += 1
                return False, f"memory_overloaded_{node_status.memory_percent:.1%}"

        # Check 4: Optimizer capacity (if available)
        optimizer = self._get_optimizer()
        if optimizer and hasattr(optimizer, "has_capacity"):
            if not optimizer.has_capacity(node, str(task_type)):
                self._tasks_denied += 1
                return False, "optimizer_no_capacity"

        # Check 5: Bandwidth for sync tasks
        if task_type == TaskType.SYNC:
            bandwidth = self._get_bandwidth_manager()
            if bandwidth and hasattr(bandwidth, "can_transfer"):
                if not bandwidth.can_transfer(node):
                    self._tasks_denied += 1
                    return False, "bandwidth_saturated"

        # Check 6: Queue depth
        queue = self._get_queue_monitor()
        if queue:
            pending = queue.get_pending_count()
            running = queue.get_running_count()
            # If queue is very deep, be more conservative
            if pending > 50 and running > 20:
                self._tasks_denied += 1
                return False, f"queue_overloaded_pending_{pending}_running_{running}"

        # All checks passed
        self._tasks_allowed += 1
        return True, "ok"

    def get_recommended_allocation(
        self,
        config_key: str,
        task_type: str | TaskType = TaskType.SELFPLAY,
    ) -> ResourceAllocation:
        """Get recommended resource allocation for a config.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            task_type: Type of task

        Returns:
            ResourceAllocation with recommended settings
        """
        # Return cached allocation if fresh
        if config_key in self._config_allocations:
            cached = self._config_allocations[config_key]
            # Cache valid for 5 minutes
            if hasattr(cached, "_timestamp") and time.time() - cached._timestamp < 300:
                return cached

        # Build new allocation based on current state
        allocation = ResourceAllocation(config_key=config_key)

        # Get healthy nodes
        healthy_nodes = [
            node_id for node_id, status in self._node_status.items()
            if status.is_healthy and status.gpu_percent < self._gpu_threshold
        ]
        allocation.recommended_nodes = healthy_nodes[:5]  # Top 5

        # Adjust based on backpressure
        if self._backpressure_level == BackpressureLevel.HIGH:
            allocation.max_concurrent_tasks = 2
            allocation.gpu_hours = 0.5
            allocation.reason = "reduced_due_to_backpressure"
        elif self._backpressure_level == BackpressureLevel.CRITICAL:
            allocation.max_concurrent_tasks = 1
            allocation.gpu_hours = 0.25
            allocation.reason = "minimal_due_to_critical_backpressure"
        else:
            allocation.reason = "normal_allocation"

        # Cache the allocation
        allocation._timestamp = time.time()
        self._config_allocations[config_key] = allocation

        return allocation

    def report_usage(
        self,
        node: str,
        cpu: float = 0.0,
        gpu: float = 0.0,
        memory: float = 0.0,
        disk: float = 0.0,
        bandwidth_mbps: float = 0.0,
        active_tasks: int = 0,
    ) -> None:
        """Report resource usage from a node.

        Args:
            node: Node ID
            cpu: CPU utilization (0.0-1.0)
            gpu: GPU utilization (0.0-1.0)
            memory: Memory utilization (0.0-1.0)
            disk: Disk utilization (0.0-1.0)
            bandwidth_mbps: Current bandwidth usage in Mbps
            active_tasks: Number of active tasks on node
        """
        status = self._node_status.get(node)
        if status is None:
            status = NodeResourceStatus(node_id=node)
            self._node_status[node] = status

        status.cpu_percent = cpu
        status.gpu_percent = gpu
        status.memory_percent = memory
        status.disk_percent = disk
        status.bandwidth_mbps = bandwidth_mbps
        status.active_tasks = active_tasks
        status.last_updated = time.time()

        # Determine health
        status.is_healthy = all([
            cpu < 0.95,
            gpu < 0.98,
            memory < 0.95,
            disk < 0.95,
        ])

        # Update global backpressure based on aggregate metrics
        self._update_backpressure()

    def _update_backpressure(self) -> None:
        """Update global backpressure level based on aggregate metrics."""
        if not self._node_status:
            self._backpressure_level = BackpressureLevel.NONE
            return

        # Calculate aggregate metrics
        total_nodes = len(self._node_status)
        healthy_nodes = sum(1 for s in self._node_status.values() if s.is_healthy)
        avg_gpu = sum(s.gpu_percent for s in self._node_status.values()) / total_nodes
        avg_cpu = sum(s.cpu_percent for s in self._node_status.values()) / total_nodes

        # Determine backpressure level
        if healthy_nodes < total_nodes * 0.5:
            self._backpressure_level = BackpressureLevel.CRITICAL
        elif avg_gpu > 0.95 or avg_cpu > 0.95:
            self._backpressure_level = BackpressureLevel.HIGH
        elif avg_gpu > 0.85 or avg_cpu > 0.85:
            self._backpressure_level = BackpressureLevel.MEDIUM
        elif avg_gpu > 0.70 or avg_cpu > 0.70:
            self._backpressure_level = BackpressureLevel.LOW
        else:
            self._backpressure_level = BackpressureLevel.NONE

    def get_backpressure_level(self) -> BackpressureLevel:
        """Get current backpressure level."""
        return self._backpressure_level

    def get_cluster_health_summary(self) -> dict[str, Any]:
        """Get summary of cluster resource health.

        Returns:
            Dict with cluster health metrics
        """
        if not self._node_status:
            return {
                "total_nodes": 0,
                "healthy_nodes": 0,
                "avg_cpu": 0.0,
                "avg_gpu": 0.0,
                "avg_memory": 0.0,
                "backpressure": self._backpressure_level.value,
                "decisions_made": self._decisions_made,
                "tasks_allowed": self._tasks_allowed,
                "tasks_denied": self._tasks_denied,
            }

        total_nodes = len(self._node_status)
        healthy_nodes = sum(1 for s in self._node_status.values() if s.is_healthy)
        avg_cpu = sum(s.cpu_percent for s in self._node_status.values()) / total_nodes
        avg_gpu = sum(s.gpu_percent for s in self._node_status.values()) / total_nodes
        avg_memory = sum(s.memory_percent for s in self._node_status.values()) / total_nodes

        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "avg_cpu": avg_cpu,
            "avg_gpu": avg_gpu,
            "avg_memory": avg_memory,
            "backpressure": self._backpressure_level.value,
            "decisions_made": self._decisions_made,
            "tasks_allowed": self._tasks_allowed,
            "tasks_denied": self._tasks_denied,
            "denial_rate": self._tasks_denied / max(1, self._decisions_made),
        }

    def get_node_status(self, node: str) -> NodeResourceStatus | None:
        """Get resource status for a specific node."""
        return self._node_status.get(node)

    def list_healthy_nodes(self) -> list[str]:
        """Get list of healthy nodes sorted by available capacity."""
        healthy = [
            (node_id, status)
            for node_id, status in self._node_status.items()
            if status.is_healthy
        ]
        # Sort by available GPU capacity (lower utilization = more capacity)
        healthy.sort(key=lambda x: x[1].gpu_percent)
        return [node_id for node_id, _ in healthy]

    def mark_node_unhealthy(self, node: str, reason: str = "") -> None:
        """Mark a node as unhealthy."""
        if node in self._node_status:
            self._node_status[node].is_healthy = False
            logger.warning(f"[UnifiedResourceCoordinator] Node {node} marked unhealthy: {reason}")

    def mark_node_healthy(self, node: str) -> None:
        """Mark a node as healthy."""
        if node in self._node_status:
            self._node_status[node].is_healthy = True
            logger.info(f"[UnifiedResourceCoordinator] Node {node} marked healthy")

    def health_check(self) -> "HealthCheckResult":
        """Check coordinator health for CoordinatorProtocol compliance.

        December 2025: Added for unified daemon health monitoring.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # Check for severe backpressure as a degraded state
        if self._backpressure_level == BackpressureLevel.CRITICAL:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Critical backpressure ({self._backpressure_level.value})",
            )

        # Check if we have node data
        if not self._node_status:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="No node status data yet",
            )

        # Count healthy vs unhealthy nodes
        healthy_count = sum(1 for s in self._node_status.values() if s.is_healthy)
        total_count = len(self._node_status)
        healthy_ratio = healthy_count / total_count if total_count > 0 else 1.0

        if healthy_ratio < 0.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Only {healthy_count}/{total_count} nodes healthy",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Healthy ({healthy_count}/{total_count} nodes, decisions: {self._decisions_made})",
        )


# Singleton instance
_coordinator: UnifiedResourceCoordinator | None = None


def get_unified_resource_coordinator() -> UnifiedResourceCoordinator:
    """Get the singleton UnifiedResourceCoordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = UnifiedResourceCoordinator()
    return _coordinator


__all__ = [
    "BackpressureLevel",
    "NodeResourceStatus",
    "ResourceAllocation",
    "TaskType",
    "UnifiedResourceCoordinator",
    "get_unified_resource_coordinator",
]

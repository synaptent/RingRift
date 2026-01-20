"""ResourceMonitoringCoordinator - Unified capacity/backpressure coordination (December 2025).

This module provides centralized monitoring of resource utilization and
backpressure across the cluster. It tracks capacity changes, detects
resource constraints, and coordinates backpressure responses.

Event Integration:
- Subscribes to CLUSTER_CAPACITY_CHANGED: Track cluster-wide capacity changes
- Subscribes to NODE_CAPACITY_UPDATED: Track per-node capacity updates
- Subscribes to BACKPRESSURE_ACTIVATED: Track backpressure engagement
- Subscribes to BACKPRESSURE_RELEASED: Track backpressure release
- Subscribes to RESOURCE_CONSTRAINT: Track resource pressure events
- Subscribes to JOB_PREEMPTED: Track job preemptions for contention analysis

Key Responsibilities:
1. Track resource utilization across all nodes
2. Coordinate backpressure activation/release
3. Provide capacity planning metrics
4. Alert on resource constraint patterns

Usage:
    from app.coordination.resource_monitoring_coordinator import (
        ResourceMonitoringCoordinator,
        wire_resource_events,
        get_resource_coordinator,
    )

    # Wire resource events
    coordinator = wire_resource_events()

    # Check if backpressure is active
    if coordinator.is_backpressure_active():
        print("Cluster under backpressure, slowing down")

    # Get resource status
    status = coordinator.get_status()
    print(f"GPU utilization: {status['avg_gpu_utilization']}%")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.async_context import fire_and_forget

# Canonical types (December 2025 consolidation)
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
from app.coordination.types import BackpressureLevel

logger = logging.getLogger(__name__)

# Import centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import ResourceMonitoringDefaults
    _DEFAULT_GPU_THRESHOLD = ResourceMonitoringDefaults.BACKPRESSURE_GPU_THRESHOLD
    _DEFAULT_MEMORY_THRESHOLD = ResourceMonitoringDefaults.BACKPRESSURE_MEMORY_THRESHOLD
    _DEFAULT_DISK_THRESHOLD = ResourceMonitoringDefaults.BACKPRESSURE_DISK_THRESHOLD
except ImportError:
    _DEFAULT_GPU_THRESHOLD = 70.0  # Session 17.42: Lowered from 90 to prevent OOM
    _DEFAULT_MEMORY_THRESHOLD = 85.0
    _DEFAULT_DISK_THRESHOLD = 90.0


# December 2025: Import ResourceType from canonical source
from app.coordination.types import ResourceType

# ResourceType is now imported from app.coordination.types
# Canonical values: CPU, GPU, MEMORY, DISK, NETWORK, HYBRID, IO

# BackpressureLevel imported from app.coordination.types
# Legacy values NONE/LOW/MEDIUM/HIGH/CRITICAL are now part of unified enum


@dataclass
class NodeResourceState:
    """Resource state for a single node."""

    node_id: str
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_utilization: float = 0.0
    memory_used_percent: float = 0.0
    disk_used_percent: float = 0.0
    task_slots_available: int = 0
    task_slots_total: int = 0
    last_update_time: float = field(default_factory=time.time)
    constraints: list[str] = field(default_factory=list)
    backpressure_active: bool = False
    backpressure_level: BackpressureLevel = BackpressureLevel.NONE

    @property
    def is_stale(self) -> bool:
        """Check if resource data is stale (>60 seconds old)."""
        return time.time() - self.last_update_time > 60.0

    @property
    def available_capacity_percent(self) -> float:
        """Get available task slot capacity as percentage."""
        if self.task_slots_total == 0:
            return 0.0
        return (self.task_slots_available / self.task_slots_total) * 100.0


@dataclass
class BackpressureEvent:
    """Record of a backpressure event."""

    node_id: str
    activated: bool  # True = activated, False = released
    level: BackpressureLevel
    reason: str
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0  # Filled in when released


ResourceAlert = BackpressureEvent


@dataclass
class ResourceStats:
    """Aggregate resource statistics."""

    total_nodes: int = 0
    healthy_nodes: int = 0
    constrained_nodes: int = 0
    avg_gpu_utilization: float = 0.0
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    total_task_slots: int = 0
    available_task_slots: int = 0
    backpressure_active_nodes: int = 0
    cluster_backpressure_level: BackpressureLevel = BackpressureLevel.NONE


class ResourceMonitoringCoordinator:
    """Coordinates resource monitoring and backpressure across the cluster.

    Tracks resource utilization, coordinates backpressure responses, and
    provides unified visibility into cluster capacity.
    """

    def __init__(
        self,
        backpressure_gpu_threshold: float = _DEFAULT_GPU_THRESHOLD,
        backpressure_memory_threshold: float = _DEFAULT_MEMORY_THRESHOLD,
        backpressure_disk_threshold: float = _DEFAULT_DISK_THRESHOLD,
        max_backpressure_history: int = 200,
    ):
        """Initialize ResourceMonitoringCoordinator.

        Args:
            backpressure_gpu_threshold: GPU % to activate backpressure
            backpressure_memory_threshold: Memory % to activate backpressure
            backpressure_disk_threshold: Disk % to activate backpressure
            max_backpressure_history: Max backpressure events to retain
        """
        self.backpressure_gpu_threshold = backpressure_gpu_threshold
        self.backpressure_memory_threshold = backpressure_memory_threshold
        self.backpressure_disk_threshold = backpressure_disk_threshold
        self.max_backpressure_history = max_backpressure_history

        # Node resource states
        self._nodes: dict[str, NodeResourceState] = {}

        # Backpressure tracking
        self._backpressure_history: list[BackpressureEvent] = []
        self._active_backpressure: dict[str, BackpressureEvent] = {}  # node_id -> event
        self._cluster_backpressure = BackpressureLevel.NONE

        # Constraint tracking
        self._constraint_counts: dict[str, int] = {}  # constraint type -> count

        # Callbacks
        self._backpressure_callbacks: list[Callable[[str, bool, BackpressureLevel], None]] = []
        self._constraint_callbacks: list[Callable[[str, str], None]] = []

        # Subscription state
        self._subscribed = False

        # Cluster capacity tracking for change detection (December 2025)
        self._last_emitted_capacity: dict[str, int] = {"total_gpus": 0, "available_gpus": 0}
        self._capacity_change_threshold = 0.1  # Emit when capacity changes by 10%+

    def subscribe_to_events(self) -> bool:
        """Subscribe to resource-related events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()

            # Use enum directly (router normalizes both enum and .value)
            router.subscribe(DataEventType.CLUSTER_CAPACITY_CHANGED, self._on_cluster_capacity_changed)
            router.subscribe(DataEventType.NODE_CAPACITY_UPDATED, self._on_node_capacity_updated)
            router.subscribe(DataEventType.BACKPRESSURE_ACTIVATED, self._on_backpressure_activated)
            router.subscribe(DataEventType.BACKPRESSURE_RELEASED, self._on_backpressure_released)
            router.subscribe(DataEventType.RESOURCE_CONSTRAINT, self._on_resource_constraint)
            router.subscribe(DataEventType.JOB_PREEMPTED, self._on_job_preempted)

            self._subscribed = True
            logger.info("[ResourceMonitoringCoordinator] Subscribed to resource events")
            return True

        except ImportError:
            logger.warning("[ResourceMonitoringCoordinator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[ResourceMonitoringCoordinator] Failed to subscribe: {e}")
            return False

    async def _on_cluster_capacity_changed(self, event: Any) -> None:
        """Handle CLUSTER_CAPACITY_CHANGED event."""
        payload = event.payload

        # Update cluster-level capacity
        total_slots = payload.get("total_task_slots", 0)
        available_slots = payload.get("available_task_slots", 0)

        logger.debug(
            f"[ResourceMonitoringCoordinator] Cluster capacity: "
            f"{available_slots}/{total_slots} slots available"
        )

        # Check if we need to adjust cluster backpressure
        if total_slots > 0:
            utilization = (total_slots - available_slots) / total_slots
            self._update_cluster_backpressure(utilization)

    async def _on_node_capacity_updated(self, event: Any) -> None:
        """Handle NODE_CAPACITY_UPDATED event."""
        payload = event.payload
        node_id = payload.get("node_id", "")

        if not node_id:
            return

        # Get or create node state
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeResourceState(node_id=node_id)

        node = self._nodes[node_id]

        # Update resource metrics
        node.gpu_utilization = payload.get("gpu_utilization", node.gpu_utilization)
        node.gpu_memory_used = payload.get("gpu_memory_used", node.gpu_memory_used)
        node.gpu_memory_total = payload.get("gpu_memory_total", node.gpu_memory_total)
        node.cpu_utilization = payload.get("cpu_utilization", node.cpu_utilization)
        node.memory_used_percent = payload.get("memory_used_percent", node.memory_used_percent)
        node.disk_used_percent = payload.get("disk_used_percent", node.disk_used_percent)
        node.task_slots_available = payload.get("task_slots_available", node.task_slots_available)
        node.task_slots_total = payload.get("task_slots_total", node.task_slots_total)
        node.last_update_time = time.time()

        # Check for threshold violations
        self._check_node_thresholds(node)

    async def _on_backpressure_activated(self, event: Any) -> None:
        """Handle BACKPRESSURE_ACTIVATED event."""
        payload = event.payload
        node_id = payload.get("node_id", "")
        level_str = payload.get("level", "medium")
        reason = payload.get("reason", "")

        try:
            level = BackpressureLevel(level_str)
        except ValueError:
            level = BackpressureLevel.MEDIUM

        # Record backpressure activation
        bp_event = BackpressureEvent(
            node_id=node_id,
            activated=True,
            level=level,
            reason=reason,
        )

        self._active_backpressure[node_id] = bp_event
        self._backpressure_history.append(bp_event)

        # Trim history
        if len(self._backpressure_history) > self.max_backpressure_history:
            self._backpressure_history = self._backpressure_history[-self.max_backpressure_history:]

        # Update node state
        if node_id in self._nodes:
            self._nodes[node_id].backpressure_active = True
            self._nodes[node_id].backpressure_level = level

        # Update cluster backpressure level
        self._recalculate_cluster_backpressure()

        # Notify callbacks
        for callback in self._backpressure_callbacks:
            try:
                callback(node_id, True, level)
            except Exception as e:
                logger.error(f"[ResourceMonitoringCoordinator] Callback error: {e}")

        logger.warning(
            f"[ResourceMonitoringCoordinator] Backpressure activated on {node_id}: "
            f"level={level.value}, reason={reason}"
        )

    async def _on_backpressure_released(self, event: Any) -> None:
        """Handle BACKPRESSURE_RELEASED event."""
        payload = event.payload
        node_id = payload.get("node_id", "")

        # Calculate duration if we have the activation event
        if node_id in self._active_backpressure:
            activation = self._active_backpressure.pop(node_id)
            duration = time.time() - activation.timestamp

            # Record release event
            release_event = BackpressureEvent(
                node_id=node_id,
                activated=False,
                level=BackpressureLevel.NONE,
                reason="released",
                duration=duration,
            )
            self._backpressure_history.append(release_event)

        # Update node state
        if node_id in self._nodes:
            self._nodes[node_id].backpressure_active = False
            self._nodes[node_id].backpressure_level = BackpressureLevel.NONE

        # Update cluster backpressure level
        self._recalculate_cluster_backpressure()

        # Notify callbacks
        for callback in self._backpressure_callbacks:
            try:
                callback(node_id, False, BackpressureLevel.NONE)
            except Exception as e:
                logger.error(f"[ResourceMonitoringCoordinator] Callback error: {e}")

        logger.info(f"[ResourceMonitoringCoordinator] Backpressure released on {node_id}")

    async def _on_resource_constraint(self, event: Any) -> None:
        """Handle RESOURCE_CONSTRAINT event."""
        payload = event.payload
        node_id = payload.get("node_id", "")
        constraint_type = payload.get("constraint_type", "")
        message = payload.get("message", "")

        # Track constraint
        self._constraint_counts[constraint_type] = self._constraint_counts.get(constraint_type, 0) + 1

        # Update node constraints
        if node_id in self._nodes and constraint_type not in self._nodes[node_id].constraints:
            self._nodes[node_id].constraints.append(constraint_type)

        # Notify callbacks
        for callback in self._constraint_callbacks:
            try:
                callback(node_id, constraint_type)
            except Exception as e:
                logger.error(f"[ResourceMonitoringCoordinator] Constraint callback error: {e}")

        logger.warning(
            f"[ResourceMonitoringCoordinator] Resource constraint on {node_id}: "
            f"{constraint_type} - {message}"
        )

    async def _on_job_preempted(self, event: Any) -> None:
        """Handle JOB_PREEMPTED event.

        Tracks job preemptions for resource contention analysis.
        Preemptions indicate that higher priority work is displacing lower priority work,
        which is useful for capacity planning and detecting resource pressure patterns.
        """
        payload = event.payload
        host = payload.get("host", "")
        preempted_job_type = payload.get("preempted_job_type", "")
        preempting_job_type = payload.get("preempting_job_type", "")
        runtime_seconds = payload.get("runtime_seconds", 0)

        # Track preemption count
        self._preemption_counts = getattr(self, "_preemption_counts", {})
        self._preemption_counts[host] = self._preemption_counts.get(host, 0) + 1

        # Preemption after short runtime may indicate scheduling inefficiency
        if runtime_seconds < 60:
            logger.warning(
                f"[ResourceMonitoringCoordinator] Quick preemption on {host}: "
                f"{preempted_job_type} after only {runtime_seconds:.0f}s "
                f"(preempted by {preempting_job_type})"
            )
        else:
            logger.info(
                f"[ResourceMonitoringCoordinator] Job preempted on {host}: "
                f"{preempted_job_type} -> {preempting_job_type}"
            )

    def _check_node_thresholds(self, node: NodeResourceState) -> None:
        """Check if node exceeds backpressure thresholds."""
        violations = []

        if node.gpu_utilization > self.backpressure_gpu_threshold:
            violations.append(f"GPU {node.gpu_utilization:.0f}%")

        if node.memory_used_percent > self.backpressure_memory_threshold:
            violations.append(f"Memory {node.memory_used_percent:.0f}%")

        if node.disk_used_percent > self.backpressure_disk_threshold:
            violations.append(f"Disk {node.disk_used_percent:.0f}%")

        if violations and not node.backpressure_active:
            # Activate backpressure (December 2025)
            reason = ", ".join(violations)
            level = self._determine_backpressure_level(node)
            self._activate_node_backpressure(node.node_id, level, reason)
            logger.debug(
                f"[ResourceMonitoringCoordinator] Threshold violations on {node.node_id}: "
                f"{reason}"
            )
        elif not violations and node.backpressure_active:
            # Release backpressure (December 2025)
            self._release_node_backpressure(node.node_id)

    def _determine_backpressure_level(self, node: NodeResourceState) -> BackpressureLevel:
        """Determine backpressure level based on node metrics."""
        max_util = max(node.gpu_utilization, node.memory_used_percent, node.disk_used_percent)
        if max_util > 95:
            return BackpressureLevel.CRITICAL
        elif max_util > 90:
            return BackpressureLevel.HIGH
        elif max_util > 85:
            return BackpressureLevel.MEDIUM
        return BackpressureLevel.LOW

    def _activate_node_backpressure(self, node_id: str, level: BackpressureLevel, reason: str) -> None:
        """Activate backpressure for a node and emit event (December 2025)."""
        bp_event = BackpressureEvent(
            node_id=node_id,
            activated=True,
            level=level,
            reason=reason,
        )
        self._active_backpressure[node_id] = bp_event
        self._backpressure_history.append(bp_event)

        if node_id in self._nodes:
            self._nodes[node_id].backpressure_active = True
            self._nodes[node_id].backpressure_level = level

        self._recalculate_cluster_backpressure()

        # Emit BACKPRESSURE_ACTIVATED event
        self._emit_backpressure_event(node_id, True, level, reason)

        # Notify callbacks
        for callback in self._backpressure_callbacks:
            try:
                callback(node_id, True, level)
            except Exception as e:
                logger.error(f"[ResourceMonitoringCoordinator] Callback error: {e}")

        logger.warning(f"[ResourceMonitoringCoordinator] Backpressure activated: {node_id} ({level.value})")

    def _release_node_backpressure(self, node_id: str) -> None:
        """Release backpressure for a node and emit event (December 2025)."""
        if node_id in self._active_backpressure:
            activation = self._active_backpressure.pop(node_id)
            duration = time.time() - activation.timestamp

            release_event = BackpressureEvent(
                node_id=node_id,
                activated=False,
                level=BackpressureLevel.NONE,
                reason="thresholds cleared",
                duration=duration,
            )
            self._backpressure_history.append(release_event)

        if node_id in self._nodes:
            self._nodes[node_id].backpressure_active = False
            self._nodes[node_id].backpressure_level = BackpressureLevel.NONE

        self._recalculate_cluster_backpressure()

        # Emit BACKPRESSURE_RELEASED event
        self._emit_backpressure_event(node_id, False, BackpressureLevel.NONE, "thresholds cleared")

        # Notify callbacks
        for callback in self._backpressure_callbacks:
            try:
                callback(node_id, False, BackpressureLevel.NONE)
            except Exception as e:
                logger.error(f"[ResourceMonitoringCoordinator] Callback error: {e}")

        logger.info(f"[ResourceMonitoringCoordinator] Backpressure released: {node_id}")

    def _emit_backpressure_event(self, node_id: str, activated: bool, level: BackpressureLevel, reason: str) -> None:
        """Emit BACKPRESSURE_ACTIVATED or BACKPRESSURE_RELEASED event.

        January 2026: Migrated to safe_emit_event for consistent event handling.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event

            if activated:
                safe_emit_event(
                    "BACKPRESSURE_ACTIVATED",
                    {
                        "node_id": node_id,
                        "level": level.value,
                        "reason": reason,
                        "resource_type": "",
                        "utilization": 0.0,
                    },
                    context="resource_monitoring_coordinator",
                )
            else:
                safe_emit_event(
                    "BACKPRESSURE_RELEASED",
                    {
                        "node_id": node_id,
                        "previous_level": level.value,
                    },
                    context="resource_monitoring_coordinator",
                )

            logger.debug(f"[ResourceMonitoringCoordinator] Emitted backpressure event for {node_id}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[ResourceMonitoringCoordinator] Failed to emit backpressure event: {e}")

    def _update_cluster_backpressure(self, utilization: float) -> None:
        """Update cluster backpressure based on utilization."""
        if utilization > 0.95:
            self._cluster_backpressure = BackpressureLevel.CRITICAL
        elif utilization > 0.85:
            self._cluster_backpressure = BackpressureLevel.HIGH
        elif utilization > 0.75:
            self._cluster_backpressure = BackpressureLevel.MEDIUM
        elif utilization > 0.6:
            self._cluster_backpressure = BackpressureLevel.LOW
        else:
            self._cluster_backpressure = BackpressureLevel.NONE

    def _recalculate_cluster_backpressure(self) -> None:
        """Recalculate cluster-level backpressure from node states."""
        if not self._active_backpressure:
            self._cluster_backpressure = BackpressureLevel.NONE
            return

        # Use highest active backpressure level
        max_level = BackpressureLevel.NONE
        for bp in self._active_backpressure.values():
            if bp.level.value > max_level.value:
                max_level = bp.level

        self._cluster_backpressure = max_level

    def update_node_resources(
        self,
        node_id: str,
        gpu_utilization: float | None = None,
        cpu_utilization: float | None = None,
        memory_used_percent: float | None = None,
        disk_used_percent: float | None = None,
        task_slots_available: int | None = None,
        task_slots_total: int | None = None,
    ) -> NodeResourceState:
        """Manually update node resource state.

        Returns:
            The updated NodeResourceState
        """
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeResourceState(node_id=node_id)

        node = self._nodes[node_id]

        if gpu_utilization is not None:
            node.gpu_utilization = gpu_utilization
        if cpu_utilization is not None:
            node.cpu_utilization = cpu_utilization
        if memory_used_percent is not None:
            node.memory_used_percent = memory_used_percent
        if disk_used_percent is not None:
            node.disk_used_percent = disk_used_percent
        if task_slots_available is not None:
            node.task_slots_available = task_slots_available
        if task_slots_total is not None:
            node.task_slots_total = task_slots_total

        node.last_update_time = time.time()
        self._check_node_thresholds(node)

        # Check if cluster capacity changed significantly (December 2025)
        self._check_and_emit_capacity_change()

        return node

    def _check_and_emit_capacity_change(self) -> None:
        """Check if cluster capacity changed and emit event if so (December 2025).

        This wires the previously orphaned CLUSTER_CAPACITY_CHANGED event.
        Also emits CAPACITY_LOW when available GPUs drop below minimum threshold.
        """
        # Calculate current cluster capacity
        total_gpus = 0
        available_gpus = 0
        total_nodes = len(self._nodes)
        healthy_nodes = 0

        for node in self._nodes.values():
            # Estimate GPUs from task slots (1 GPU ≈ 2 task slots)
            node_gpus = max(1, node.task_slots_total // 2) if node.task_slots_total else 1
            node_available = max(0, node.task_slots_available // 2) if node.task_slots_available else 0

            total_gpus += node_gpus
            available_gpus += node_available

            # Node is healthy if it was updated recently (within 5 minutes)
            if time.time() - node.last_update_time < 300:
                healthy_nodes += 1

        # Check if capacity changed significantly
        last = self._last_emitted_capacity
        if last["total_gpus"] > 0:
            change_ratio = abs(available_gpus - last["available_gpus"]) / max(1, last["total_gpus"])
        else:
            change_ratio = 1.0 if available_gpus > 0 else 0.0

        if change_ratio >= self._capacity_change_threshold or total_gpus != last["total_gpus"]:
            # Emit event
            try:
                from app.coordination.event_router import emit_cluster_capacity_changed
                from app.core.async_context import fire_and_forget

                fire_and_forget(
                    emit_cluster_capacity_changed(
                        total_gpus=total_gpus,
                        available_gpus=available_gpus,
                        total_nodes=total_nodes,
                        healthy_nodes=healthy_nodes,
                        source="resource_monitoring_coordinator",
                    ),
                    name="emit_cluster_capacity_changed",
                )

                self._last_emitted_capacity = {
                    "total_gpus": total_gpus,
                    "available_gpus": available_gpus,
                }

                logger.debug(
                    f"[ResourceMonitoringCoordinator] Emitted CLUSTER_CAPACITY_CHANGED: "
                    f"{available_gpus}/{total_gpus} GPUs, {healthy_nodes}/{total_nodes} nodes"
                )

            except (ImportError, RuntimeError) as e:
                # RuntimeError covers "no running event loop" in sync contexts
                logger.debug(f"Could not emit capacity change: {e}")

        # December 29, 2025: Emit CAPACITY_LOW when available GPUs drop below threshold
        min_gpus = getattr(self, "_min_gpu_threshold", 3)  # Default: 3 GPUs minimum
        was_low = getattr(self, "_capacity_was_low", False)

        if available_gpus < min_gpus and not was_low:
            # Capacity dropped below minimum - emit CAPACITY_LOW
            try:
                from app.distributed.data_events import emit_capacity_low
                from app.core.async_context import fire_and_forget

                fire_and_forget(
                    emit_capacity_low(
                        current_gpus=available_gpus,
                        min_gpus=min_gpus,
                        provider="",  # Cluster-wide
                        source="resource_monitoring_coordinator",
                    ),
                    name="emit_capacity_low",
                )
                self._capacity_was_low = True
                logger.warning(
                    f"[ResourceMonitoringCoordinator] CAPACITY_LOW: "
                    f"{available_gpus} GPUs < {min_gpus} minimum"
                )

            except (ImportError, RuntimeError) as e:
                logger.debug(f"Could not emit capacity_low: {e}")

        elif available_gpus >= min_gpus and was_low:
            # Capacity restored above minimum - emit CAPACITY_RESTORED
            try:
                from app.distributed.data_events import emit_capacity_restored
                from app.core.async_context import fire_and_forget

                fire_and_forget(
                    emit_capacity_restored(
                        current_gpus=available_gpus,
                        min_gpus=min_gpus,
                        source="resource_monitoring_coordinator",
                    ),
                    name="emit_capacity_restored",
                )
                self._capacity_was_low = False
                logger.info(
                    f"[ResourceMonitoringCoordinator] CAPACITY_RESTORED: "
                    f"{available_gpus} GPUs >= {min_gpus} minimum"
                )

            except (ImportError, RuntimeError) as e:
                logger.debug(f"Could not emit capacity_restored: {e}")

    def on_backpressure_change(
        self, callback: Callable[[str, bool, BackpressureLevel], None]
    ) -> None:
        """Register callback for backpressure changes.

        Args:
            callback: Function(node_id, activated, level)
        """
        self._backpressure_callbacks.append(callback)

    def on_constraint(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for resource constraints.

        Args:
            callback: Function(node_id, constraint_type)
        """
        self._constraint_callbacks.append(callback)

    def is_backpressure_active(self, node_id: str | None = None) -> bool:
        """Check if backpressure is active.

        Args:
            node_id: Specific node to check, or None for cluster-wide

        Returns:
            True if backpressure is active
        """
        if node_id:
            return node_id in self._active_backpressure
        return len(self._active_backpressure) > 0

    def get_backpressure_level(self, node_id: str | None = None) -> BackpressureLevel:
        """Get current backpressure level.

        Args:
            node_id: Specific node, or None for cluster level

        Returns:
            Current backpressure level
        """
        if node_id:
            if node_id in self._nodes:
                return self._nodes[node_id].backpressure_level
            return BackpressureLevel.NONE
        return self._cluster_backpressure

    def get_node_state(self, node_id: str) -> NodeResourceState | None:
        """Get resource state for a specific node."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[NodeResourceState]:
        """Get resource states for all nodes."""
        return list(self._nodes.values())

    def get_constrained_nodes(self) -> list[NodeResourceState]:
        """Get nodes that have active constraints."""
        return [n for n in self._nodes.values() if n.constraints or n.backpressure_active]

    def get_backpressure_history(self, limit: int = 50) -> list[BackpressureEvent]:
        """Get recent backpressure events."""
        return self._backpressure_history[-limit:]

    def get_stats(self) -> ResourceStats:
        """Get aggregate resource statistics."""
        nodes = list(self._nodes.values())
        healthy_nodes = [n for n in nodes if not n.is_stale and not n.backpressure_active]
        constrained = [n for n in nodes if n.constraints or n.backpressure_active]

        total_slots = sum(n.task_slots_total for n in nodes)
        available_slots = sum(n.task_slots_available for n in nodes)

        avg_gpu = sum(n.gpu_utilization for n in nodes) / len(nodes) if nodes else 0.0
        avg_cpu = sum(n.cpu_utilization for n in nodes) / len(nodes) if nodes else 0.0
        avg_mem = sum(n.memory_used_percent for n in nodes) / len(nodes) if nodes else 0.0

        return ResourceStats(
            total_nodes=len(nodes),
            healthy_nodes=len(healthy_nodes),
            constrained_nodes=len(constrained),
            avg_gpu_utilization=avg_gpu,
            avg_cpu_utilization=avg_cpu,
            avg_memory_utilization=avg_mem,
            total_task_slots=total_slots,
            available_task_slots=available_slots,
            backpressure_active_nodes=len(self._active_backpressure),
            cluster_backpressure_level=self._cluster_backpressure,
        )

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring."""
        stats = self.get_stats()

        return {
            "total_nodes": stats.total_nodes,
            "healthy_nodes": stats.healthy_nodes,
            "constrained_nodes": stats.constrained_nodes,
            "avg_gpu_utilization": round(stats.avg_gpu_utilization, 1),
            "avg_cpu_utilization": round(stats.avg_cpu_utilization, 1),
            "avg_memory_utilization": round(stats.avg_memory_utilization, 1),
            "total_task_slots": stats.total_task_slots,
            "available_task_slots": stats.available_task_slots,
            "backpressure_active": self.is_backpressure_active(),
            "backpressure_level": self._cluster_backpressure.value,
            "backpressure_nodes": list(self._active_backpressure.keys()),
            "constraint_counts": dict(self._constraint_counts),
            "subscribed": self._subscribed,
        }

    def health_check(self) -> HealthCheckResult:
        """Perform health check on resource monitoring (December 2025).

        Returns:
            HealthCheckResult with health status including:
            - healthy: Overall health status
            - node_health_ratio: Ratio of healthy to total nodes
            - backpressure_active: Whether cluster is under backpressure
            - subscription_status: Event subscription health
        """
        stats = self.get_stats()

        # Calculate node health ratio
        total = stats.total_nodes
        healthy = stats.healthy_nodes
        node_health_ratio = healthy / max(total, 1)

        # Check if we have too many constrained nodes
        constrained_ratio = stats.constrained_nodes / max(total, 1)

        # Overall health criteria
        healthy_status = (
            self._subscribed  # Must be subscribed to events
            and node_health_ratio >= 0.5  # At least 50% nodes healthy
            and constrained_ratio < 0.8  # Less than 80% constrained
        )

        # Determine status based on health
        if healthy_status:
            status = CoordinatorStatus.RUNNING
            message = f"Monitoring {total} nodes, {healthy} healthy"
        elif self._subscribed:
            status = CoordinatorStatus.DEGRADED
            message = f"Degraded: {stats.constrained_nodes} constrained nodes"
        else:
            status = CoordinatorStatus.STOPPED
            message = "Not subscribed to events"

        return HealthCheckResult(
            healthy=healthy_status,
            status=status,
            message=message,
            details={
                "total_nodes": total,
                "healthy_nodes": healthy,
                "constrained_nodes": stats.constrained_nodes,
                "node_health_ratio": round(node_health_ratio, 3),
                "constrained_ratio": round(constrained_ratio, 3),
                "backpressure_active": self.is_backpressure_active(),
                "backpressure_level": self._cluster_backpressure.value,
                "subscribed": self._subscribed,
                "avg_gpu_utilization": round(stats.avg_gpu_utilization, 1),
            },
        )


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_resource_coordinator: ResourceMonitoringCoordinator | None = None


def get_resource_coordinator() -> ResourceMonitoringCoordinator:
    """Get the global ResourceMonitoringCoordinator singleton."""
    global _resource_coordinator
    if _resource_coordinator is None:
        _resource_coordinator = ResourceMonitoringCoordinator()
    return _resource_coordinator


def wire_resource_events() -> ResourceMonitoringCoordinator:
    """Wire resource events to the coordinator.

    Returns:
        The wired ResourceMonitoringCoordinator instance
    """
    coordinator = get_resource_coordinator()
    coordinator.subscribe_to_events()
    return coordinator


def is_cluster_under_backpressure() -> bool:
    """Convenience function to check if cluster is under backpressure."""
    return get_resource_coordinator().is_backpressure_active()


def get_cluster_capacity() -> dict[str, int]:
    """Convenience function to get cluster task slot capacity."""
    stats = get_resource_coordinator().get_stats()
    return {
        "total": stats.total_task_slots,
        "available": stats.available_task_slots,
    }

def update_node_resources(
    node_id: str,
    gpu_utilization: float | None = None,
    cpu_utilization: float | None = None,
    memory_used_percent: float | None = None,
    disk_used_percent: float | None = None,
    task_slots_available: int | None = None,
    task_slots_total: int | None = None,
) -> NodeResourceState:
    """Convenience function to update node resources."""
    return get_resource_coordinator().update_node_resources(
        node_id=node_id,
        gpu_utilization=gpu_utilization,
        cpu_utilization=cpu_utilization,
        memory_used_percent=memory_used_percent,
        disk_used_percent=disk_used_percent,
        task_slots_available=task_slots_available,
        task_slots_total=task_slots_total,
    )

def check_resource_thresholds(node_state: NodeResourceState) -> None:
    """Convenience function to check resource thresholds for a node."""
    get_resource_coordinator()._check_node_thresholds(node_state)


__all__ = [
    "BackpressureEvent",
    "BackpressureLevel",
    "NodeResourceState",
    "ResourceAlert",
    "ResourceMonitoringCoordinator",
    "ResourceStats",
    "ResourceType",
    "check_resource_thresholds",
    "get_cluster_capacity",
    "get_resource_coordinator",
    "is_cluster_under_backpressure",
    "update_node_resources",
    "wire_resource_events",
]

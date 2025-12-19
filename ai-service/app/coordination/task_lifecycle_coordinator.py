"""TaskLifecycleCoordinator - Unified task event monitoring (December 2025).

This module provides centralized monitoring of task lifecycle events across
the cluster. It tracks task spawning, completion, failures, and heartbeats
to provide complete visibility into distributed task execution.

Event Integration:
- Subscribes to TASK_SPAWNED: Track new task creation
- Subscribes to TASK_COMPLETED: Track successful completions
- Subscribes to TASK_FAILED: Track task failures
- Subscribes to TASK_HEARTBEAT: Track active tasks
- Subscribes to TASK_ORPHANED: Track orphaned tasks
- Subscribes to TASK_CANCELLED: Track cancelled tasks

Key Responsibilities:
1. Track all tasks across all nodes
2. Detect orphaned tasks (no heartbeat for threshold period)
3. Provide task statistics by type, node, and status
4. Alert on failure patterns and orphaned tasks

Usage:
    from app.coordination.task_lifecycle_coordinator import (
        TaskLifecycleCoordinator,
        wire_task_events,
        get_task_lifecycle_coordinator,
    )

    # Wire task lifecycle events
    coordinator = wire_task_events()

    # Get task statistics
    stats = coordinator.get_stats()
    print(f"Active tasks: {stats['active_tasks']}")
    print(f"Orphaned tasks: {stats['orphaned_tasks']}")

    # Get tasks by node
    tasks = coordinator.get_tasks_by_node("gh200-a")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ORPHANED = "orphaned"


@dataclass
class TrackedTask:
    """Information about a tracked task."""

    task_id: str
    task_type: str
    node_id: str
    status: TaskStatus = TaskStatus.RUNNING
    spawned_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    heartbeat_count: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0

    @property
    def age(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.spawned_at

    @property
    def time_since_heartbeat(self) -> float:
        """Get time since last heartbeat."""
        return time.time() - self.last_heartbeat

    def is_alive(self, heartbeat_threshold: float = 60.0) -> bool:
        """Check if task is still alive based on heartbeat."""
        if self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        return self.time_since_heartbeat < heartbeat_threshold


@dataclass
class TaskLifecycleStats:
    """Aggregate task lifecycle statistics."""

    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    orphaned_tasks: int = 0
    total_spawned: int = 0
    average_duration: float = 0.0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_node: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    failure_rate: float = 0.0


class TaskLifecycleCoordinator:
    """Coordinates task lifecycle monitoring across the cluster.

    Tracks all tasks from spawn to completion/failure, detects orphaned
    tasks, and provides unified visibility into distributed task execution.
    """

    def __init__(
        self,
        heartbeat_threshold_seconds: float = 60.0,
        orphan_check_interval_seconds: float = 30.0,
        max_history: int = 1000,
    ):
        """Initialize TaskLifecycleCoordinator.

        Args:
            heartbeat_threshold_seconds: Time without heartbeat to mark orphaned
            orphan_check_interval_seconds: How often to check for orphans
            max_history: Maximum completed tasks to retain in history
        """
        self.heartbeat_threshold = heartbeat_threshold_seconds
        self.orphan_check_interval = orphan_check_interval_seconds
        self.max_history = max_history

        # Active tasks by task_id
        self._active_tasks: Dict[str, TrackedTask] = {}

        # Completed task history
        self._completed_tasks: List[TrackedTask] = []

        # Orphaned tasks (kept separate for visibility)
        self._orphaned_tasks: Dict[str, TrackedTask] = {}

        # Statistics
        self._total_spawned = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_cancelled = 0
        self._total_duration = 0.0

        # Callbacks
        self._orphan_callbacks: List[Callable[[TrackedTask], None]] = []
        self._failure_callbacks: List[Callable[[TrackedTask], None]] = []

        # Subscription state
        self._subscribed = False

        # Last orphan check time
        self._last_orphan_check = time.time()

    def subscribe_to_events(self) -> bool:
        """Subscribe to task lifecycle events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()

            bus.subscribe(DataEventType.TASK_SPAWNED, self._on_task_spawned)
            bus.subscribe(DataEventType.TASK_COMPLETED, self._on_task_completed)
            bus.subscribe(DataEventType.TASK_FAILED, self._on_task_failed)
            bus.subscribe(DataEventType.TASK_HEARTBEAT, self._on_task_heartbeat)
            bus.subscribe(DataEventType.TASK_ORPHANED, self._on_task_orphaned)
            bus.subscribe(DataEventType.TASK_CANCELLED, self._on_task_cancelled)

            self._subscribed = True
            logger.info("[TaskLifecycleCoordinator] Subscribed to task events")
            return True

        except ImportError:
            logger.warning("[TaskLifecycleCoordinator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[TaskLifecycleCoordinator] Failed to subscribe: {e}")
            return False

    async def _on_task_spawned(self, event) -> None:
        """Handle TASK_SPAWNED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")
        task_type = payload.get("task_type", "")
        node_id = payload.get("node_id", "")

        task = TrackedTask(
            task_id=task_id,
            task_type=task_type,
            node_id=node_id,
            status=TaskStatus.RUNNING,
        )

        self._active_tasks[task_id] = task
        self._total_spawned += 1

        logger.debug(
            f"[TaskLifecycleCoordinator] Task spawned: {task_id} ({task_type}) "
            f"on {node_id}"
        )

    async def _on_task_completed(self, event) -> None:
        """Handle TASK_COMPLETED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        task = self._active_tasks.pop(task_id, None)
        if task is None:
            # Task wasn't tracked (possibly started before coordinator)
            task = TrackedTask(
                task_id=task_id,
                task_type=payload.get("task_type", "unknown"),
                node_id=payload.get("node_id", ""),
            )

        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.duration = payload.get("duration", task.age)
        task.result = payload.get("result", {})

        self._record_completion(task)
        self._total_completed += 1
        self._total_duration += task.duration

        logger.debug(
            f"[TaskLifecycleCoordinator] Task completed: {task_id} "
            f"(duration={task.duration:.1f}s)"
        )

    async def _on_task_failed(self, event) -> None:
        """Handle TASK_FAILED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        task = self._active_tasks.pop(task_id, None)
        if task is None:
            task = TrackedTask(
                task_id=task_id,
                task_type=payload.get("task_type", "unknown"),
                node_id=payload.get("node_id", ""),
            )

        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        task.duration = task.age
        task.error = payload.get("error", "Unknown error")

        self._record_completion(task)
        self._total_failed += 1

        # Notify failure callbacks
        for callback in self._failure_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"[TaskLifecycleCoordinator] Failure callback error: {e}")

        logger.warning(
            f"[TaskLifecycleCoordinator] Task failed: {task_id} - {task.error}"
        )

    async def _on_task_heartbeat(self, event) -> None:
        """Handle TASK_HEARTBEAT event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.last_heartbeat = time.time()
            task.heartbeat_count += 1
        elif task_id in self._orphaned_tasks:
            # Task recovered from orphaned state
            task = self._orphaned_tasks.pop(task_id)
            task.status = TaskStatus.RUNNING
            task.last_heartbeat = time.time()
            self._active_tasks[task_id] = task
            logger.info(f"[TaskLifecycleCoordinator] Task recovered: {task_id}")

    async def _on_task_orphaned(self, event) -> None:
        """Handle TASK_ORPHANED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        if task_id in self._active_tasks:
            task = self._active_tasks.pop(task_id)
            task.status = TaskStatus.ORPHANED
            self._orphaned_tasks[task_id] = task

            for callback in self._orphan_callbacks:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"[TaskLifecycleCoordinator] Orphan callback error: {e}")

            logger.warning(f"[TaskLifecycleCoordinator] Task orphaned: {task_id}")

    async def _on_task_cancelled(self, event) -> None:
        """Handle TASK_CANCELLED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        task = self._active_tasks.pop(task_id, None)
        if task is None:
            task = self._orphaned_tasks.pop(task_id, None)

        if task:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            task.duration = task.age

            self._record_completion(task)
            self._total_cancelled += 1

            logger.info(f"[TaskLifecycleCoordinator] Task cancelled: {task_id}")

    def _record_completion(self, task: TrackedTask) -> None:
        """Record a completed task in history."""
        self._completed_tasks.append(task)

        # Trim history
        if len(self._completed_tasks) > self.max_history:
            self._completed_tasks = self._completed_tasks[-self.max_history :]

    def check_for_orphans(self) -> List[TrackedTask]:
        """Check for orphaned tasks and mark them.

        Returns:
            List of newly orphaned tasks
        """
        now = time.time()

        # Rate limit the check
        if now - self._last_orphan_check < self.orphan_check_interval:
            return []

        self._last_orphan_check = now
        newly_orphaned = []

        # Check active tasks for stale heartbeats
        orphan_task_ids = []
        for task_id, task in self._active_tasks.items():
            if not task.is_alive(self.heartbeat_threshold):
                orphan_task_ids.append(task_id)
                newly_orphaned.append(task)

        # Move orphaned tasks
        for task_id in orphan_task_ids:
            task = self._active_tasks.pop(task_id)
            task.status = TaskStatus.ORPHANED
            self._orphaned_tasks[task_id] = task

            # Notify callbacks
            for callback in self._orphan_callbacks:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"[TaskLifecycleCoordinator] Orphan callback error: {e}")

            logger.warning(
                f"[TaskLifecycleCoordinator] Detected orphaned task: {task_id} "
                f"(no heartbeat for {task.time_since_heartbeat:.0f}s)"
            )

        return newly_orphaned

    def register_task(
        self,
        task_id: str,
        task_type: str,
        node_id: str,
    ) -> TrackedTask:
        """Manually register a task for tracking.

        Args:
            task_id: Task identifier
            task_type: Type of task
            node_id: Node running the task

        Returns:
            The created TrackedTask
        """
        task = TrackedTask(
            task_id=task_id,
            task_type=task_type,
            node_id=node_id,
            status=TaskStatus.RUNNING,
        )

        self._active_tasks[task_id] = task
        self._total_spawned += 1

        return task

    def update_heartbeat(self, task_id: str) -> bool:
        """Update heartbeat for a task.

        Returns:
            True if task was found and updated
        """
        if task_id in self._active_tasks:
            self._active_tasks[task_id].last_heartbeat = time.time()
            self._active_tasks[task_id].heartbeat_count += 1
            return True
        return False

    def on_orphan(self, callback: Callable[[TrackedTask], None]) -> None:
        """Register a callback for orphaned tasks."""
        self._orphan_callbacks.append(callback)

    def on_failure(self, callback: Callable[[TrackedTask], None]) -> None:
        """Register a callback for failed tasks."""
        self._failure_callbacks.append(callback)

    def get_active_tasks(self) -> List[TrackedTask]:
        """Get all active tasks."""
        return list(self._active_tasks.values())

    def get_task(self, task_id: str) -> Optional[TrackedTask]:
        """Get a specific task by ID."""
        return self._active_tasks.get(task_id) or self._orphaned_tasks.get(task_id)

    def get_tasks_by_node(self, node_id: str) -> List[TrackedTask]:
        """Get all tasks on a specific node."""
        return [t for t in self._active_tasks.values() if t.node_id == node_id]

    def get_tasks_by_type(self, task_type: str) -> List[TrackedTask]:
        """Get all tasks of a specific type."""
        return [t for t in self._active_tasks.values() if t.task_type == task_type]

    def get_orphaned_tasks(self) -> List[TrackedTask]:
        """Get all orphaned tasks."""
        return list(self._orphaned_tasks.values())

    def get_history(self, limit: int = 50) -> List[TrackedTask]:
        """Get recent task completion history."""
        return self._completed_tasks[-limit:]

    def get_stats(self) -> TaskLifecycleStats:
        """Get aggregate task lifecycle statistics."""
        # Count by type
        by_type: Dict[str, int] = {}
        for task in self._active_tasks.values():
            by_type[task.task_type] = by_type.get(task.task_type, 0) + 1

        # Count by node
        by_node: Dict[str, int] = {}
        for task in self._active_tasks.values():
            by_node[task.node_id] = by_node.get(task.node_id, 0) + 1

        # Count by status
        by_status = {
            "running": len(self._active_tasks),
            "orphaned": len(self._orphaned_tasks),
            "completed": self._total_completed,
            "failed": self._total_failed,
            "cancelled": self._total_cancelled,
        }

        # Calculate failure rate
        total_finished = self._total_completed + self._total_failed
        failure_rate = (
            self._total_failed / total_finished if total_finished > 0 else 0.0
        )

        # Average duration
        avg_duration = (
            self._total_duration / self._total_completed
            if self._total_completed > 0
            else 0.0
        )

        return TaskLifecycleStats(
            active_tasks=len(self._active_tasks),
            completed_tasks=self._total_completed,
            failed_tasks=self._total_failed,
            cancelled_tasks=self._total_cancelled,
            orphaned_tasks=len(self._orphaned_tasks),
            total_spawned=self._total_spawned,
            average_duration=avg_duration,
            by_type=by_type,
            by_node=by_node,
            by_status=by_status,
            failure_rate=failure_rate,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring."""
        stats = self.get_stats()

        # Check for orphans
        self.check_for_orphans()

        return {
            "active_tasks": stats.active_tasks,
            "orphaned_tasks": stats.orphaned_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "total_spawned": stats.total_spawned,
            "failure_rate": round(stats.failure_rate * 100, 1),
            "average_duration": round(stats.average_duration, 1),
            "by_type": stats.by_type,
            "by_node": stats.by_node,
            "subscribed": self._subscribed,
            "heartbeat_threshold": self.heartbeat_threshold,
        }


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_task_lifecycle_coordinator: Optional[TaskLifecycleCoordinator] = None


def get_task_lifecycle_coordinator() -> TaskLifecycleCoordinator:
    """Get the global TaskLifecycleCoordinator singleton."""
    global _task_lifecycle_coordinator
    if _task_lifecycle_coordinator is None:
        _task_lifecycle_coordinator = TaskLifecycleCoordinator()
    return _task_lifecycle_coordinator


def wire_task_events(
    heartbeat_threshold: float = 60.0,
) -> TaskLifecycleCoordinator:
    """Wire task lifecycle events to the coordinator.

    Args:
        heartbeat_threshold: Seconds without heartbeat to mark orphaned

    Returns:
        The wired TaskLifecycleCoordinator instance
    """
    global _task_lifecycle_coordinator
    _task_lifecycle_coordinator = TaskLifecycleCoordinator(
        heartbeat_threshold_seconds=heartbeat_threshold,
    )
    _task_lifecycle_coordinator.subscribe_to_events()
    return _task_lifecycle_coordinator


def get_task_stats() -> TaskLifecycleStats:
    """Convenience function to get task statistics."""
    return get_task_lifecycle_coordinator().get_stats()


def get_active_task_count() -> int:
    """Convenience function to get count of active tasks."""
    return len(get_task_lifecycle_coordinator().get_active_tasks())


__all__ = [
    "TaskLifecycleCoordinator",
    "TaskStatus",
    "TrackedTask",
    "TaskLifecycleStats",
    "get_task_lifecycle_coordinator",
    "wire_task_events",
    "get_task_stats",
    "get_active_task_count",
]

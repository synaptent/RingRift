"""Coordination Facade - Simplified API for Training Coordination (December 2025).

This module provides a unified, simplified interface for common coordination
operations, hiding the complexity of 75+ internal coordinator classes.

Use this facade for:
- Task spawning and status checking
- Training job management
- Cluster health monitoring
- Event subscription

For advanced use cases, you can still access the underlying coordinators directly.

Usage:
    from app.coordination.facade import CoordinationFacade

    coord = CoordinationFacade()

    # Check if we can spawn a task
    if coord.can_spawn_task("selfplay", "node-001"):
        task_id = coord.spawn_task("selfplay", "node-001", games=100)

    # Start training
    training_id = coord.start_training("hex8", 2)

    # Get cluster health
    health = coord.get_cluster_health()
    print(f"Healthy nodes: {health['healthy_nodes']}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable

# December 2025: Import TaskStatus from canonical source
from app.coordination.types import TaskStatus

# December 2025: Use centralized timeout constants
from app.config.coordination_defaults import JobTimeoutDefaults
from app.coordination.event_utils import make_config_key

logger = logging.getLogger(__name__)

# TaskStatus is now imported from app.coordination.types
# Canonical values: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMED_OUT, ORPHANED


class TrainingStatus(str, Enum):
    """Training job status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ClusterHealth:
    """Summary of cluster health."""
    total_nodes: int
    healthy_nodes: int
    degraded_nodes: int
    unhealthy_nodes: int
    evicted_nodes: int
    available_node_ids: list[str]
    timestamp: str


@dataclass
class TaskInfo:
    """Information about a task."""
    task_id: str
    task_type: str
    node_id: str
    status: TaskStatus
    started_at: float
    runtime_seconds: float


class CoordinationFacade:
    """Simplified interface for common coordination operations.

    This facade provides a stable API while internal coordinators may change.
    It handles lazy initialization of underlying components.
    """

    def __init__(self):
        self._task_coordinator = None
        self._training_coordinator = None
        self._node_monitor = None
        self._event_router = None

    # =========================================================================
    # Task Operations
    # =========================================================================

    def can_spawn_task(self, task_type: str, node_id: str) -> bool:
        """Check if a task can be spawned on a node.

        Args:
            task_type: Type of task (selfplay, training, export, etc.)
            node_id: Target node identifier

        Returns:
            True if task can be spawned
        """
        # Check node availability
        if not self._is_node_available(node_id):
            return False

        # Check task limits
        coordinator = self._get_task_coordinator()
        if coordinator is None:
            return True  # No coordinator, allow by default

        try:
            from app.coordination.task_coordinator import TaskType
            tt = TaskType(task_type) if task_type in [t.value for t in TaskType] else None
            if tt:
                return coordinator.can_spawn(tt, node_id)
        except ValueError as e:
            # Invalid task type value
            logger.debug(f"Invalid task type {task_type}: {e}")
        except (ImportError, ModuleNotFoundError) as e:
            # TaskType not importable (rare but possible)
            logger.warning(f"TaskType import failed, allowing spawn: {e}")

        return True

    def spawn_task(
        self,
        task_type: str,
        node_id: str,
        timeout_seconds: float = float(JobTimeoutDefaults.GPU_SELFPLAY),
        **metadata,
    ) -> str | None:
        """Spawn a task on a node.

        Args:
            task_type: Type of task
            node_id: Target node
            timeout_seconds: Maximum runtime
            **metadata: Additional task metadata

        Returns:
            Task ID if spawned, None if failed
        """
        coordinator = self._get_task_coordinator()
        if coordinator is None:
            logger.warning("No task coordinator available")
            return None

        try:
            from app.coordination.task_coordinator import TaskType
            tt = TaskType(task_type) if task_type in [t.value for t in TaskType] else TaskType.SELFPLAY

            metadata['timeout_seconds'] = timeout_seconds
            task_id = coordinator.register_task(tt, node_id, metadata=metadata)
            return task_id
        except ValueError as e:
            # Invalid task type or validation error - caller may want to know
            logger.warning(f"Task validation failed for {task_type} on {node_id}: {e}")
            return None
        except (ConnectionError, TimeoutError, OSError) as e:
            # Network/connectivity issues - retryable
            logger.error(f"Network error spawning task on {node_id}: {e}")
            return None
        except (SystemExit, KeyboardInterrupt):
            # Signal exceptions must propagate (Dec 2025)
            raise
        except Exception as e:
            # Unexpected error - log with traceback for debugging
            logger.error(f"Unexpected error spawning task {task_type} on {node_id}: {e}", exc_info=True)
            return None

    def get_task_status(self, task_id: str) -> TaskInfo | None:
        """Get status of a task.

        Args:
            task_id: Task identifier

        Returns:
            TaskInfo or None if not found
        """
        coordinator = self._get_task_coordinator()
        if coordinator is None:
            return None

        try:
            task = coordinator.registry.get_task(task_id)
            if task:
                return TaskInfo(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    node_id=task.node_id,
                    status=TaskStatus(task.status) if task.status in [s.value for s in TaskStatus] else TaskStatus.RUNNING,
                    started_at=task.started_at,
                    runtime_seconds=task.runtime_seconds(),
                )
        except (KeyError, AttributeError, ValueError, RuntimeError) as e:
            logger.debug(f"Failed to get task status: {e}")

        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled
        """
        coordinator = self._get_task_coordinator()
        if coordinator is None:
            return False

        try:
            coordinator.registry.update_task_status(task_id, "cancelled")
            return True
        except (KeyError, ValueError, RuntimeError, OSError) as e:
            logger.error(f"Failed to cancel task: {e}")
            return False

    def get_active_tasks(self, node_id: str | None = None) -> list[TaskInfo]:
        """Get list of active tasks.

        Args:
            node_id: Optional filter by node

        Returns:
            List of active TaskInfo objects
        """
        coordinator = self._get_task_coordinator()
        if coordinator is None:
            return []

        try:
            tasks = coordinator.registry.get_active_tasks()
            if node_id:
                tasks = [t for t in tasks if t.node_id == node_id]

            return [
                TaskInfo(
                    task_id=t.task_id,
                    task_type=t.task_type.value,
                    node_id=t.node_id,
                    status=TaskStatus.RUNNING,
                    started_at=t.started_at,
                    runtime_seconds=t.runtime_seconds(),
                )
                for t in tasks
            ]
        except (KeyError, AttributeError, RuntimeError) as e:
            logger.debug(f"Failed to get active tasks: {e}")
            return []

    # =========================================================================
    # Training Operations
    # =========================================================================

    def start_training(
        self,
        board_type: str,
        num_players: int,
        **kwargs,
    ) -> str | None:
        """Start a training job.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players
            **kwargs: Additional training parameters

        Returns:
            Training job ID or None if failed
        """
        coordinator = self._get_training_coordinator()
        if coordinator is None:
            logger.warning("No training coordinator available")
            return None

        try:
            config_key = make_config_key(board_type, num_players)
            return coordinator.start_training(config_key, **kwargs)
        except ValueError as e:
            # Invalid config or parameters
            logger.warning(f"Invalid training config {make_config_key(board_type, num_players)}: {e}")
            return None
        except FileNotFoundError as e:
            # Missing training data or model
            logger.error(f"Missing training resources for {make_config_key(board_type, num_players)}: {e}")
            return None
        except (SystemExit, KeyboardInterrupt):
            # Signal exceptions must propagate (Dec 2025)
            raise
        except Exception as e:
            logger.error(f"Failed to start training for {make_config_key(board_type, num_players)}: {e}", exc_info=True)
            return None

    def get_training_status(self, board_type: str, num_players: int) -> TrainingStatus:
        """Get status of a training job.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            TrainingStatus
        """
        coordinator = self._get_training_coordinator()
        if coordinator is None:
            return TrainingStatus.NOT_STARTED

        try:
            config_key = make_config_key(board_type, num_players)
            status = coordinator.get_status(config_key)
            if status and status.get("running"):
                return TrainingStatus.RUNNING
            elif status and status.get("completed"):
                return TrainingStatus.COMPLETED
        except (KeyError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get training status for {config_key}: {e}")

        return TrainingStatus.NOT_STARTED

    def stop_training(self, board_type: str, num_players: int) -> bool:
        """Stop a running training job.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            True if stopped
        """
        coordinator = self._get_training_coordinator()
        if coordinator is None:
            return False

        try:
            config_key = make_config_key(board_type, num_players)
            coordinator.stop_training(config_key)
            return True
        except KeyError:
            # No training running for this config - not an error
            logger.debug(f"No active training to stop for {make_config_key(board_type, num_players)}")
            return True  # Still success - nothing to stop
        except (ProcessLookupError, OSError) as e:
            # Process already terminated
            logger.debug(f"Training process already stopped for {make_config_key(board_type, num_players)}: {e}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop training for {make_config_key(board_type, num_players)}: {e}", exc_info=True)
            return False

    # =========================================================================
    # Cluster Health Operations
    # =========================================================================

    def get_cluster_health(self) -> ClusterHealth:
        """Get cluster health summary.

        Returns:
            ClusterHealth object
        """
        monitor = self._get_node_monitor()

        if monitor is None:
            return ClusterHealth(
                total_nodes=0,
                healthy_nodes=0,
                degraded_nodes=0,
                unhealthy_nodes=0,
                evicted_nodes=0,
                available_node_ids=[],
                timestamp=datetime.now().isoformat(),
            )

        summary = monitor.get_cluster_summary()
        return ClusterHealth(
            total_nodes=summary["total_nodes"],
            healthy_nodes=summary["healthy"],
            degraded_nodes=summary["degraded"],
            unhealthy_nodes=summary["unhealthy"],
            evicted_nodes=summary["evicted"],
            available_node_ids=summary["available_nodes"],
            timestamp=summary["timestamp"],
        )

    def is_node_healthy(self, node_id: str) -> bool:
        """Check if a specific node is healthy.

        Args:
            node_id: Node identifier

        Returns:
            True if healthy
        """
        return self._is_node_available(node_id)

    def get_available_nodes(self) -> list[str]:
        """Get list of nodes available for task assignment.

        Returns:
            List of node IDs
        """
        monitor = self._get_node_monitor()
        if monitor is None:
            return []
        return monitor.get_available_nodes()

    def evict_node(self, node_id: str) -> bool:
        """Force evict a node from task assignment.

        Args:
            node_id: Node identifier

        Returns:
            True if evicted
        """
        monitor = self._get_node_monitor()
        if monitor is None:
            return False
        return monitor.force_evict(node_id)

    def recover_node(self, node_id: str) -> bool:
        """Force recover an evicted node.

        Args:
            node_id: Node identifier

        Returns:
            True if recovered
        """
        monitor = self._get_node_monitor()
        if monitor is None:
            return False
        return monitor.force_recover(node_id)

    # =========================================================================
    # Event Operations
    # =========================================================================

    def subscribe(self, event_type: str, callback: Callable) -> str:
        """Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to
            callback: Callback function (sync or async)

        Returns:
            Subscription ID
        """
        router = self._get_event_router()
        if router is None:
            logger.warning("No event router available")
            return ""

        try:
            return router.subscribe(event_type, callback)
        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            logger.error(f"Failed to subscribe: {e}")
            return ""

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: Subscription ID from subscribe()

        Returns:
            True if unsubscribed
        """
        router = self._get_event_router()
        if router is None:
            return False

        try:
            router.unsubscribe(subscription_id)
            return True
        except (KeyError, ValueError, RuntimeError) as e:
            logger.debug(f"Could not unsubscribe {subscription_id}: {e}")
            return False

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _is_node_available(self, node_id: str) -> bool:
        """Check if node is available via monitor."""
        monitor = self._get_node_monitor()
        if monitor is None:
            return True  # No monitor, assume available
        return monitor.is_node_available(node_id)

    def _get_task_coordinator(self) -> "Any | None":
        """Lazy load task coordinator.

        Returns:
            TaskCoordinator instance or None if unavailable
        """
        if self._task_coordinator is None:
            try:
                from app.coordination.task_coordinator import get_task_coordinator
                self._task_coordinator = get_task_coordinator()
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug(f"Could not load task coordinator: {e}")
        return self._task_coordinator

    def _get_training_coordinator(self) -> "Any | None":
        """Lazy load training coordinator.

        Returns:
            TrainingCoordinator instance or None if unavailable
        """
        if self._training_coordinator is None:
            try:
                from app.coordination.training_coordinator import get_training_coordinator
                self._training_coordinator = get_training_coordinator()
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug(f"Could not load training coordinator: {e}")
        return self._training_coordinator

    def _get_node_monitor(self) -> "Any | None":
        """Lazy load node health monitor.

        December 2025: node_health_monitor is deprecated in favor of
        health_check_orchestrator, but this facade method is kept for
        backward compatibility with existing callers.

        Returns:
            HealthCheckOrchestrator instance or None if unavailable
        """
        if self._node_monitor is None:
            try:
                # Dec 2025: Use health_facade (unified interface)
                from app.coordination.health_facade import get_health_orchestrator
                self._node_monitor = get_health_orchestrator()
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug(f"Could not load health orchestrator: {e}")
        return self._node_monitor

    def _get_event_router(self) -> "Any | None":
        """Lazy load event router.

        Returns:
            EventRouter instance or None if unavailable
        """
        if self._event_router is None:
            try:
                from app.coordination.event_router import get_event_router
                self._event_router = get_event_router()
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug(f"Could not load event router: {e}")
        return self._event_router

    def health_check(self) -> "HealthCheckResult":
        """Check health of the coordination facade and underlying components.

        Returns HealthCheckResult with:
        - Status of each underlying coordinator
        - Cluster health summary
        - Event router status

        December 2025: Added for DaemonManager integration.
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        components = {}
        all_healthy = True

        # Check task coordinator
        tc = self._get_task_coordinator()
        if tc is not None:
            components["task_coordinator"] = "available"
        else:
            components["task_coordinator"] = "unavailable"
            all_healthy = False

        # Check training coordinator
        trc = self._get_training_coordinator()
        if trc is not None:
            components["training_coordinator"] = "available"
        else:
            components["training_coordinator"] = "unavailable"

        # Check node monitor
        nm = self._get_node_monitor()
        if nm is not None:
            components["node_monitor"] = "available"
        else:
            components["node_monitor"] = "unavailable"

        # Check event router
        er = self._get_event_router()
        if er is not None:
            components["event_router"] = "available"
        else:
            components["event_router"] = "unavailable"
            all_healthy = False

        # Get cluster health if available
        try:
            cluster_health = self.get_cluster_health()
            components["cluster"] = {
                "total_nodes": cluster_health.total_nodes,
                "healthy_nodes": cluster_health.healthy_nodes,
                "available_nodes": len(cluster_health.available_node_ids),
            }
        except Exception as e:
            components["cluster"] = {"error": str(e)}

        message = "Coordination facade healthy" if all_healthy else "Some coordinators unavailable"

        return HealthCheckResult(
            healthy=all_healthy,
            status=CoordinatorStatus.RUNNING if all_healthy else CoordinatorStatus.DEGRADED,
            message=message,
            details=components,
        )


# Global instance
_facade: CoordinationFacade | None = None


def get_coordination_facade() -> CoordinationFacade:
    """Get the global coordination facade instance."""
    global _facade
    if _facade is None:
        _facade = CoordinationFacade()
    return _facade


# Convenience functions for common operations
def can_spawn_task(task_type: str, node_id: str) -> bool:
    """Check if a task can be spawned. Uses global facade."""
    return get_coordination_facade().can_spawn_task(task_type, node_id)


def spawn_task(task_type: str, node_id: str, **kwargs) -> str | None:
    """Spawn a task. Uses global facade."""
    return get_coordination_facade().spawn_task(task_type, node_id, **kwargs)


def get_cluster_health() -> ClusterHealth:
    """Get cluster health. Uses global facade."""
    return get_coordination_facade().get_cluster_health()


def get_available_nodes() -> list[str]:
    """Get available nodes. Uses global facade."""
    return get_coordination_facade().get_available_nodes()

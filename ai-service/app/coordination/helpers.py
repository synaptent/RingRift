"""Coordination Helper Module - Safe coordination utilities.

This module provides safe, reusable coordination functions to eliminate duplicate
try/except import patterns found across 13+ scripts in the codebase.

Instead of this pattern in every script:
    try:
        from app.coordination import (
            TaskCoordinator, TaskType, can_spawn,
            OrchestratorRole, acquire_orchestrator_role, get_registry,
        )
        HAS_COORDINATION = True
    except ImportError:
        HAS_COORDINATION = False
        TaskCoordinator = None
        TaskType = None

Use this:
    from app.coordination.helpers import (
        has_coordination, can_spawn_safe, register_task_safe,
        acquire_role_safe, has_role, get_coordinator_safe,
    )

Usage:
    # Check if coordination is available
    if has_coordination():
        coordinator = get_coordinator_safe()

    # Or use safe wrappers that handle unavailability gracefully
    allowed, reason = can_spawn_safe(TaskType.SELFPLAY, "node-1")
    if allowed:
        register_task_safe(task_id, TaskType.SELFPLAY, "node-1", os.getpid())
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
from typing import Any

logger = logging.getLogger(__name__)

# Try to import coordination components
_HAS_COORDINATION = False
_TaskCoordinator = None
_TaskType = None
_TaskLimits = None
_OrchestratorRole = None
_OrchestratorRegistry = None
_Safeguards = None
_CircuitBreaker = None
_CircuitState = None

# Import functions
_can_spawn = None
_get_coordinator = None
_get_registry = None
_acquire_orchestrator_role = None
_release_orchestrator_role = None
_check_before_spawn = None

try:
    from app.coordination import (
        CircuitBreaker,
        CircuitState,
        OrchestratorRegistry,
        OrchestratorRole,
        Safeguards,
        TaskCoordinator,
        TaskLimits,
        TaskType,
        acquire_orchestrator_role,
        can_spawn,
        check_before_spawn,
        get_coordinator,
        get_registry,
        release_orchestrator_role,
    )
    _HAS_COORDINATION = True
    _TaskCoordinator = TaskCoordinator
    _TaskType = TaskType
    _TaskLimits = TaskLimits
    _OrchestratorRole = OrchestratorRole
    _OrchestratorRegistry = OrchestratorRegistry
    _Safeguards = Safeguards
    _CircuitBreaker = CircuitBreaker
    _CircuitState = CircuitState
    _can_spawn = can_spawn
    _get_coordinator = get_coordinator
    _get_registry = get_registry
    _acquire_orchestrator_role = acquire_orchestrator_role
    _release_orchestrator_role = release_orchestrator_role
    _check_before_spawn = check_before_spawn
except ImportError as e:
    logger.debug(f"Coordination module not available: {e}")


def has_coordination() -> bool:
    """Check if the coordination module is available.

    Returns:
        True if app.coordination is importable, False otherwise.
    """
    return _HAS_COORDINATION


def get_task_types() -> type | None:
    """Get the TaskType enum if available.

    Returns:
        TaskType enum or None if not available.
    """
    return _TaskType


def get_orchestrator_roles() -> type | None:
    """Get the OrchestratorRole enum if available.

    Returns:
        OrchestratorRole enum or None if not available.
    """
    return _OrchestratorRole


# =============================================================================
# Coordinator Functions
# =============================================================================

def get_coordinator_safe() -> Any | None:
    """Get the task coordinator instance if available.

    Returns:
        TaskCoordinator instance or None if not available.
    """
    if not _HAS_COORDINATION or _get_coordinator is None:
        return None
    try:
        return _get_coordinator()
    except Exception as e:
        logger.debug(f"Failed to get coordinator: {e}")
        return None


def can_spawn_safe(
    task_type: Any,
    node_id: str | None = None
) -> tuple[bool, str]:
    """Safely check if a task can be spawned.

    Args:
        task_type: TaskType enum value
        node_id: Node identifier (defaults to hostname)

    Returns:
        Tuple of (allowed: bool, reason: str)
        If coordination unavailable, returns (True, "coordination_unavailable")
    """
    if not _HAS_COORDINATION or _get_coordinator is None:
        return (True, "coordination_unavailable")

    if node_id is None:
        node_id = socket.gethostname()

    try:
        coordinator = _get_coordinator()
        return coordinator.can_spawn_task(task_type, node_id)
    except Exception as e:
        logger.warning(f"can_spawn check failed: {e}")
        return (True, f"check_failed: {e}")


def register_task_safe(
    task_id: str,
    task_type: Any,
    node_id: str | None = None,
    pid: int | None = None
) -> bool:
    """Safely register a task with the coordinator.

    Args:
        task_id: Unique task identifier
        task_type: TaskType enum value
        node_id: Node identifier (defaults to hostname)
        pid: Process ID (defaults to current process)

    Returns:
        True if registration succeeded, False otherwise.
    """
    if not _HAS_COORDINATION:
        return False

    coordinator = get_coordinator_safe()
    if coordinator is None:
        return False

    if node_id is None:
        node_id = socket.gethostname()
    if pid is None:
        pid = os.getpid()

    try:
        coordinator.register_task(task_id, task_type, node_id, pid)
        return True
    except Exception as e:
        logger.warning(f"Task registration failed: {e}")
        return False


def complete_task_safe(task_id: str) -> bool:
    """Safely mark a task as completed.

    Args:
        task_id: Task identifier to complete

    Returns:
        True if completion succeeded, False otherwise.
    """
    if not _HAS_COORDINATION:
        return False

    coordinator = get_coordinator_safe()
    if coordinator is None:
        return False

    try:
        coordinator.complete_task(task_id)
        return True
    except Exception as e:
        logger.warning(f"Task completion failed: {e}")
        return False


def fail_task_safe(task_id: str, error: str = "") -> bool:
    """Safely mark a task as failed.

    Args:
        task_id: Task identifier to fail
        error: Error message

    Returns:
        True if operation succeeded, False otherwise.
    """
    if not _HAS_COORDINATION:
        return False

    coordinator = get_coordinator_safe()
    if coordinator is None:
        return False

    try:
        coordinator.fail_task(task_id, error)
        return True
    except Exception as e:
        logger.warning(f"Task failure recording failed: {e}")
        return False


# =============================================================================
# Orchestrator Role Functions
# =============================================================================

def get_registry_safe() -> Any | None:
    """Get the orchestrator registry if available.

    Returns:
        OrchestratorRegistry instance or None if not available.
    """
    if not _HAS_COORDINATION or _get_registry is None:
        return None
    try:
        return _get_registry()
    except Exception as e:
        logger.debug(f"Failed to get registry: {e}")
        return None


def acquire_role_safe(role: Any) -> bool:
    """Safely attempt to acquire an orchestrator role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        True if role was acquired, False otherwise.
    """
    if not _HAS_COORDINATION or _acquire_orchestrator_role is None:
        return False

    try:
        return _acquire_orchestrator_role(role)
    except Exception as e:
        logger.warning(f"Role acquisition failed: {e}")
        return False


def release_role_safe(role: Any) -> bool:
    """Safely release an orchestrator role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        True if role was released, False otherwise.
    """
    if not _HAS_COORDINATION or _release_orchestrator_role is None:
        return False

    try:
        _release_orchestrator_role(role)
        return True
    except Exception as e:
        logger.warning(f"Role release failed: {e}")
        return False


def has_role(role: Any) -> bool:
    """Check if the current process holds a role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        True if role is held, False otherwise.
    """
    registry = get_registry_safe()
    if registry is None:
        return False

    try:
        return registry.is_role_held(role)
    except Exception as e:
        logger.debug(f"Role check failed: {e}")
        return False


def get_role_holder(role: Any) -> Any | None:
    """Get information about who holds a role.

    Args:
        role: OrchestratorRole enum value

    Returns:
        OrchestratorInfo if role is held, None otherwise.
    """
    registry = get_registry_safe()
    if registry is None:
        return None

    try:
        return registry.get_role_holder(role)
    except Exception as e:
        logger.debug(f"Could not get role holder for {role}: {e}")
        return None


# =============================================================================
# Safeguards Functions
# =============================================================================

def check_spawn_allowed(
    task_type: str = "unknown",
    config_key: str = ""
) -> tuple[bool, str]:
    """Check safeguards before spawning a task.

    Args:
        task_type: Type of task being spawned
        config_key: Configuration key (e.g., "square8_2p")

    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    if not _HAS_COORDINATION or _check_before_spawn is None:
        return (True, "safeguards_unavailable")

    try:
        return _check_before_spawn(task_type, config_key)
    except Exception as e:
        logger.warning(f"Safeguard check failed: {e}")
        return (True, f"check_failed: {e}")


def get_safeguards() -> Any | None:
    """Get the Safeguards instance if available.

    Returns:
        Safeguards instance or None if not available.
    """
    if not _HAS_COORDINATION or _Safeguards is None:
        return None
    try:
        return _Safeguards()
    except Exception:
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def get_current_node_id() -> str:
    """Get the current node's identifier.

    Returns:
        Hostname of current machine.
    """
    return socket.gethostname()


def is_unified_loop_running() -> bool:
    """Check if a unified loop orchestrator is already running.

    Useful for daemons that should defer to the main orchestrator.

    Returns:
        True if unified loop holds the role, False otherwise.
    """
    if not _HAS_COORDINATION or _OrchestratorRole is None:
        return False

    try:
        return has_role(_OrchestratorRole.UNIFIED_LOOP)
    except Exception:
        return False


def warn_if_orchestrator_running(daemon_name: str = "daemon") -> None:
    """Print a warning if the unified orchestrator is already running.

    Args:
        daemon_name: Name of the daemon for the warning message.
    """
    if not _HAS_COORDINATION or _OrchestratorRole is None:
        return

    registry = get_registry_safe()
    if registry is None:
        return

    try:
        if registry.is_role_held(_OrchestratorRole.UNIFIED_LOOP):
            holder = registry.get_role_holder(_OrchestratorRole.UNIFIED_LOOP)
            existing_pid = holder.pid if holder else "unknown"
            print(f"[{daemon_name}] WARNING: Unified orchestrator is running (PID {existing_pid})")
            print(f"[{daemon_name}] The orchestrator handles this work - this {daemon_name} may duplicate work")
    except Exception as e:
        logger.debug(f"Could not check orchestrator status: {e}")


# =============================================================================
# Queue Backpressure Functions
# =============================================================================

# Queue types
_QueueType = None
_should_throttle_production = None
_should_stop_production = None
_get_throttle_factor = None
_report_queue_depth = None

try:
    from app.coordination import (
        QueueType,
        get_throttle_factor,
        report_queue_depth,
        should_stop_production,
        should_throttle_production,
    )
    _QueueType = QueueType
    _should_throttle_production = should_throttle_production
    _should_stop_production = should_stop_production
    _get_throttle_factor = get_throttle_factor
    _report_queue_depth = report_queue_depth
except ImportError:
    pass


def get_queue_types() -> type | None:
    """Get the QueueType enum if available."""
    return _QueueType


def should_throttle_safe(queue_type: Any = None) -> bool:
    """Safely check if production should be throttled.

    Args:
        queue_type: QueueType enum value (uses default if None)

    Returns:
        True if throttling should occur, False otherwise.
    """
    if _should_throttle_production is None:
        return False
    try:
        if queue_type is None and _QueueType is not None:
            queue_type = _QueueType.TRAINING_DATA
        return _should_throttle_production(queue_type) if queue_type else False
    except Exception:
        return False


def should_stop_safe(queue_type: Any = None) -> bool:
    """Safely check if production should stop entirely.

    Returns:
        True if production should stop, False otherwise.
    """
    if _should_stop_production is None:
        return False
    try:
        if queue_type is None and _QueueType is not None:
            queue_type = _QueueType.TRAINING_DATA
        return _should_stop_production(queue_type) if queue_type else False
    except Exception:
        return False


def get_throttle_factor_safe(queue_type: Any = None) -> float:
    """Safely get the throttle factor (1.0 = no throttle, 0.0 = full throttle).

    Returns:
        Throttle factor between 0.0 and 1.0, defaults to 1.0.
    """
    if _get_throttle_factor is None:
        return 1.0
    try:
        if queue_type is None and _QueueType is not None:
            queue_type = _QueueType.TRAINING_DATA
        return _get_throttle_factor(queue_type) if queue_type else 1.0
    except Exception:
        return 1.0


def report_queue_depth_safe(queue_type: Any, depth: int) -> None:
    """Safely report queue depth for backpressure calculation."""
    if _report_queue_depth is None:
        return
    with contextlib.suppress(Exception):
        _report_queue_depth(queue_type, depth)


# =============================================================================
# Sync Mutex Functions
# =============================================================================

_sync_lock = None
_acquire_sync_lock = None
_release_sync_lock = None

try:
    from app.coordination import (
        acquire_sync_lock,
        release_sync_lock,
        sync_lock,
    )
    _sync_lock = sync_lock
    _acquire_sync_lock = acquire_sync_lock
    _release_sync_lock = release_sync_lock
except ImportError:
    pass


def has_sync_lock() -> bool:
    """Check if sync lock functionality is available."""
    return _sync_lock is not None


def get_sync_lock_context() -> Any | None:
    """Get the sync_lock context manager if available.

    Returns:
        sync_lock context manager or None.
    """
    return _sync_lock


def acquire_sync_lock_safe(host: str, timeout: float = 120.0) -> bool:
    """Safely acquire a sync lock for a host.

    Args:
        host: Host identifier
        timeout: Lock timeout in seconds

    Returns:
        True if lock acquired, False otherwise.
    """
    if _acquire_sync_lock is None:
        return True  # Allow operation if no lock available
    try:
        return _acquire_sync_lock(host, timeout)
    except Exception as e:
        logger.warning(f"Failed to acquire sync lock for {host}: {e}")
        return True  # Allow operation on error


def release_sync_lock_safe(host: str) -> None:
    """Safely release a sync lock for a host."""
    if _release_sync_lock is None:
        return
    try:
        _release_sync_lock(host)
    except Exception as e:
        logger.warning(f"Failed to release sync lock for {host}: {e}")


# =============================================================================
# Bandwidth Management Functions
# =============================================================================

_TransferPriority = None
_request_bandwidth = None
_release_bandwidth = None
_bandwidth_allocation = None

try:
    from app.coordination import (
        TransferPriority,
        bandwidth_allocation,
        release_bandwidth,
        request_bandwidth,
    )
    _TransferPriority = TransferPriority
    _request_bandwidth = request_bandwidth
    _release_bandwidth = release_bandwidth
    _bandwidth_allocation = bandwidth_allocation
except ImportError:
    pass


def has_bandwidth_manager() -> bool:
    """Check if bandwidth management is available."""
    return _request_bandwidth is not None


def get_transfer_priorities() -> type | None:
    """Get the TransferPriority enum if available."""
    return _TransferPriority


def request_bandwidth_safe(
    host: str,
    requested_mbps: float = 100.0,
    priority: Any = None,
) -> tuple[bool, float]:
    """Safely request bandwidth allocation.

    Args:
        host: Host identifier
        requested_mbps: Requested bandwidth in Mbps
        priority: TransferPriority enum value

    Returns:
        Tuple of (granted: bool, allocated_mbps: float)
    """
    if _request_bandwidth is None:
        return (True, requested_mbps)  # Allow full bandwidth if not managed
    try:
        if priority is None and _TransferPriority is not None:
            priority = _TransferPriority.NORMAL
        return _request_bandwidth(host, requested_mbps, priority)
    except Exception as e:
        logger.warning(f"Bandwidth request failed for {host}: {e}")
        return (True, requested_mbps)


def release_bandwidth_safe(host: str) -> None:
    """Safely release bandwidth allocation."""
    if _release_bandwidth is None:
        return
    try:
        _release_bandwidth(host)
    except Exception as e:
        logger.warning(f"Bandwidth release failed for {host}: {e}")


def get_bandwidth_context() -> Any | None:
    """Get the bandwidth_allocation context manager if available."""
    return _bandwidth_allocation


# =============================================================================
# Duration Scheduling Functions
# =============================================================================

_can_schedule_task = None
_register_running_task = None
_record_task_completion = None
_estimate_task_duration = None

try:
    from app.coordination import (
        can_schedule_task,
        estimate_task_duration,
        record_task_completion,
        register_running_task,
    )
    _can_schedule_task = can_schedule_task
    _register_running_task = register_running_task
    _record_task_completion = record_task_completion
    _estimate_task_duration = estimate_task_duration
except ImportError:
    pass


def has_duration_scheduler() -> bool:
    """Check if duration scheduling is available."""
    return _can_schedule_task is not None


def can_schedule_task_safe(task_type: str, estimated_duration: float = 60.0) -> bool:
    """Safely check if a task can be scheduled based on resource availability.

    Args:
        task_type: Type of task to schedule
        estimated_duration: Estimated duration in seconds

    Returns:
        True if task can be scheduled, False otherwise.
    """
    if _can_schedule_task is None:
        return True
    try:
        return _can_schedule_task(task_type, estimated_duration)
    except Exception:
        return True


def register_running_task_safe(
    task_id: str,
    task_type: str,
    estimated_duration: float = 60.0,
) -> bool:
    """Safely register a task as running for duration tracking.

    Returns:
        True if registered successfully, False otherwise.
    """
    if _register_running_task is None:
        return False
    try:
        _register_running_task(task_id, task_type, estimated_duration)
        return True
    except Exception as e:
        logger.debug(f"Failed to register running task: {e}")
        return False


def record_task_completion_safe(
    task_id: str,
    task_type: str,
    actual_duration: float,
) -> bool:
    """Safely record task completion for duration learning.

    Returns:
        True if recorded successfully, False otherwise.
    """
    if _record_task_completion is None:
        return False
    try:
        _record_task_completion(task_id, task_type, actual_duration)
        return True
    except Exception as e:
        logger.debug(f"Failed to record task completion: {e}")
        return False


def estimate_duration_safe(task_type: str, default: float = 60.0) -> float:
    """Safely estimate task duration based on historical data.

    Args:
        task_type: Type of task
        default: Default duration if no data available

    Returns:
        Estimated duration in seconds.
    """
    if _estimate_task_duration is None:
        return default
    try:
        return _estimate_task_duration(task_type) or default
    except Exception:
        return default


# =============================================================================
# Cross-Process Events Functions
# =============================================================================

_publish_event = None
_poll_events = None
_ack_event = None
_subscribe_process = None
_CrossProcessEventPoller = None

try:
    from app.coordination import (
        CrossProcessEventPoller,
        ack_event,
        poll_events,
        publish_event,
        subscribe_process,
    )
    _publish_event = publish_event
    _poll_events = poll_events
    _ack_event = ack_event
    _subscribe_process = subscribe_process
    _CrossProcessEventPoller = CrossProcessEventPoller
except ImportError:
    pass


def has_cross_process_events() -> bool:
    """Check if cross-process events are available."""
    return _publish_event is not None


def publish_event_safe(event_type: str, payload: dict[str, Any] | None = None) -> bool:
    """Safely publish a cross-process event.

    Returns:
        True if published, False otherwise.
    """
    if _publish_event is None:
        return False
    try:
        _publish_event(event_type, payload or {})
        return True
    except Exception as e:
        logger.debug(f"Failed to publish event: {e}")
        return False


def poll_events_safe(
    event_types: list[str] | None = None,
    limit: int = 100,
) -> list[Any]:
    """Safely poll for cross-process events.

    Returns:
        List of events, empty list if unavailable or error.
    """
    if _poll_events is None:
        return []
    try:
        return _poll_events(event_types, limit) or []
    except Exception:
        return []


def ack_event_safe(event_id: int) -> bool:
    """Safely acknowledge a cross-process event."""
    if _ack_event is None:
        return False
    try:
        _ack_event(event_id)
        return True
    except Exception:
        return False


def subscribe_process_safe(process_name: str | None = None) -> bool:
    """Safely subscribe the current process to events."""
    if _subscribe_process is None:
        return False
    try:
        _subscribe_process(process_name)
        return True
    except Exception:
        return False


def get_event_poller_class() -> type | None:
    """Get the CrossProcessEventPoller class if available."""
    return _CrossProcessEventPoller


# =============================================================================
# Resource Targets Functions
# =============================================================================

_get_resource_targets = None
_get_host_targets = None
_get_cluster_summary = None
_should_scale_up_targets = None
_should_scale_down_targets = None
_set_backpressure = None

try:
    from app.coordination import (
        get_cluster_summary,
        get_host_targets,
        get_resource_targets,
        set_backpressure,
        should_scale_down as should_scale_down_targets,
        should_scale_up as should_scale_up_targets,
    )
    _get_resource_targets = get_resource_targets
    _get_host_targets = get_host_targets
    _get_cluster_summary = get_cluster_summary
    _should_scale_up_targets = should_scale_up_targets
    _should_scale_down_targets = should_scale_down_targets
    _set_backpressure = set_backpressure
except ImportError:
    pass


def has_resource_targets() -> bool:
    """Check if resource targets are available."""
    return _get_resource_targets is not None


def get_resource_targets_safe() -> Any | None:
    """Safely get the resource target manager."""
    if _get_resource_targets is None:
        return None
    try:
        return _get_resource_targets()
    except Exception:
        return None


def get_host_targets_safe(host: str) -> Any | None:
    """Safely get targets for a specific host."""
    if _get_host_targets is None:
        return None
    try:
        return _get_host_targets(host)
    except Exception:
        return None


def get_cluster_summary_safe() -> dict[str, Any]:
    """Safely get cluster summary."""
    if _get_cluster_summary is None:
        return {}
    try:
        return _get_cluster_summary() or {}
    except Exception:
        return {}


def should_scale_up_safe(host: str) -> bool:
    """Safely check if a host should scale up."""
    if _should_scale_up_targets is None:
        return False
    try:
        return _should_scale_up_targets(host)
    except Exception:
        return False


def should_scale_down_safe(host: str) -> bool:
    """Safely check if a host should scale down."""
    if _should_scale_down_targets is None:
        return False
    try:
        return _should_scale_down_targets(host)
    except Exception:
        return False


def set_backpressure_safe(active: bool) -> None:
    """Safely set backpressure state."""
    if _set_backpressure is None:
        return
    with contextlib.suppress(Exception):
        _set_backpressure(active)


# =============================================================================
# Re-exports for convenience
# =============================================================================

# These allow direct use of the types when coordination is available
TaskCoordinator = _TaskCoordinator
TaskType = _TaskType
TaskLimits = _TaskLimits
OrchestratorRole = _OrchestratorRole
OrchestratorRegistry = _OrchestratorRegistry
Safeguards = _Safeguards
CircuitBreaker = _CircuitBreaker
CircuitState = _CircuitState
QueueType = _QueueType
TransferPriority = _TransferPriority
CrossProcessEventPoller = _CrossProcessEventPoller


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CrossProcessEventPoller",
    "OrchestratorRegistry",
    "OrchestratorRole",
    "QueueType",
    "Safeguards",
    "TaskLimits",
    # Re-exported types (for convenience)
    "TaskType",
    "TransferPriority",
    # Role management
    "acquire_role_safe",
    "can_spawn_safe",
    # Spawn checks
    "check_spawn_allowed",
    "complete_task_safe",
    "fail_task_safe",
    # Safe accessors
    "get_coordinator_safe",
    # Utilities
    "get_current_node_id",
    "get_orchestrator_roles",
    # Queue helpers
    "get_queue_types",
    "get_registry_safe",
    "get_role_holder",
    "get_safeguards",
    "get_task_types",
    # Type checking helpers
    "has_coordination",
    "has_role",
    "is_unified_loop_running",
    "register_task_safe",
    "release_role_safe",
    "should_throttle_safe",
    "warn_if_orchestrator_running",
]

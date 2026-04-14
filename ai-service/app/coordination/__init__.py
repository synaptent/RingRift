"""Cluster coordination package for unified resource management.

Provides centralized task coordination to prevent uncontrolled task spawning.

Primary modules:
1. task_coordinator - SQLite-backed coordination with rate limiting, backpressure
2. orchestrator_registry - Role-based mutual exclusion with heartbeat liveness
3. safeguards - Circuit breakers, resource monitoring, spawn rate tracking
4. queue_monitor - Queue depth monitoring with backpressure signals
5. bandwidth_manager - Network bandwidth allocation for transfers
6. sync_mutex - Cross-process mutex for rsync operations
7. p2p_backend - REST API client for P2P orchestrator cluster
8. job_scheduler - Priority-based job scheduling with Elo curriculum
9. stage_events - Event-driven pipeline orchestration with callbacks

December 2025: Reorganized into submodule exports for maintainability.
Imports are organized into:
- _exports_core.py - Task coordination, orchestrator registry, resources, health
- _exports_sync.py - Sync operations (bandwidth, mutex, WAL, integrity)
- _exports_daemon.py - Daemon management
- _exports_events.py - Event system (router, emitters, stage events)
- _exports_orchestrators.py - High-level orchestrators
- _exports_utils.py - Utilities and helpers

Usage:
    # Task coordination (canonical)
    from app.coordination import TaskCoordinator, TaskType
    coordinator = TaskCoordinator.get_instance()
    if coordinator.can_spawn_task(TaskType.SELFPLAY, "node-1")[0]:
        coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1")

    # Orchestrator role management
    from app.coordination import acquire_orchestrator_role, OrchestratorRole
    if acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR):
        # This process now holds the orchestrator role
        pass

    # Backpressure checking
    from app.coordination import should_throttle_production, QueueType
    if should_throttle_production(QueueType.TRAINING_DATA):
        # Slow down or skip data production
        pass
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

# =============================================================================
# Lazy package exports (April 2026)
# =============================================================================
_EXPORT_MODULES = {
    "app.coordination._exports_core": "_exports_core.py",
    "app.coordination._exports_sync": "_exports_sync.py",
    "app.coordination._exports_daemon": "_exports_daemon.py",
    "app.coordination._exports_events": "_exports_events.py",
    "app.coordination._exports_orchestrators": "_exports_orchestrators.py",
    "app.coordination._exports_utils": "_exports_utils.py",
}
_EXPORT_NAME_TO_MODULE: dict[str, str] = {}
_LAZY_EXPORT_CACHE: dict[str, object] = {}


def _read_declared_exports(filename: str) -> list[str]:
    """Read ``__all__`` from a sibling export module without importing it."""

    path = Path(__file__).with_name(filename)
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, list):
                        return [str(item) for item in value]
    raise RuntimeError(f"Unable to resolve __all__ from {path}")


for _module_name, _filename in _EXPORT_MODULES.items():
    for _export_name in _read_declared_exports(_filename):
        _EXPORT_NAME_TO_MODULE[_export_name] = _module_name


def _load_export(name: str) -> object:
    """Load one re-exported symbol lazily and cache it."""

    if name in _LAZY_EXPORT_CACHE:
        return _LAZY_EXPORT_CACHE[name]
    module_name = _EXPORT_NAME_TO_MODULE[name]
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    _LAZY_EXPORT_CACHE[name] = value
    return value


def _resolve_export(name: str) -> object:
    """Resolve one exported symbol and bind it into module globals."""

    value = _load_export(name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Resolve historical package re-exports lazily."""

    if name in _EXPORT_NAME_TO_MODULE:
        return _resolve_export(name)
    if name in {"core_utils", "core_events"}:
        value = importlib.import_module(f"app.coordination.{name}")
        _LAZY_EXPORT_CACHE[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def initialize_all_coordinators(
    auto_trigger_pipeline: bool = False,
    heartbeat_threshold: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    wrap_handlers: bool = True,
) -> dict:
    """Initialize all orchestrators and coordinators with event wiring (December 2025).

    This is the single entry point to bootstrap all coordination infrastructure.
    It wires all event subscriptions and returns a status dictionary.

    Features:
    - Retry logic with exponential backoff for failed subscriptions
    - Validation that subscriptions actually succeeded
    - Emits COORDINATOR_INIT_FAILED for persistent failures
    - Optionally wraps handlers with resilience (exception boundaries + timeouts)

    Args:
        auto_trigger_pipeline: If True, pipeline stages auto-trigger downstream
        heartbeat_threshold: Seconds without heartbeat to mark tasks orphaned
        max_retries: Maximum retry attempts per coordinator
        retry_delay: Base delay for exponential backoff
        wrap_handlers: If True, wrap handlers with resilience

    Returns:
        Dict with initialization status for each orchestrator
    """
    from app.coordination.lifecycle import (
        initialize_all_coordinators as _initialize_all_coordinators,
    )

    return _initialize_all_coordinators(
        auto_trigger_pipeline=auto_trigger_pipeline,
        heartbeat_threshold=heartbeat_threshold,
        max_retries=max_retries,
        retry_delay=retry_delay,
        wrap_handlers=wrap_handlers,
    )


def get_all_coordinator_status() -> dict:
    """Get unified status from all orchestrators and coordinators.

    Returns:
        Dict with status from each orchestrator
    """
    from app.coordination.status_reporting import (
        get_all_coordinator_status as _get_all_coordinator_status,
    )

    return _get_all_coordinator_status()


def get_system_health() -> dict:
    """Get aggregated system health from all coordinators (December 2025).

    Returns:
        Dict with health information including overall_health score,
        status string, per-coordinator health, issues list, and handler_health.
    """
    from app.coordination.status_reporting import (
        get_system_health as _get_system_health,
    )

    return _get_system_health()


async def shutdown_all_coordinators(
    timeout_seconds: float = 30.0,
    emit_events: bool = True,
) -> dict:
    """Gracefully shutdown all coordinators (December 2025).

    Args:
        timeout_seconds: Maximum time to wait for graceful shutdown
        emit_events: Whether to emit shutdown events

    Returns:
        Dict with shutdown status for each coordinator
    """
    from app.coordination.lifecycle import (
        shutdown_all_coordinators as _shutdown_all_coordinators,
    )

    return await _shutdown_all_coordinators(
        timeout_seconds=timeout_seconds,
        emit_events=emit_events,
    )

def stop_coordinator_heartbeats() -> None:
    """Stop the coordinator heartbeat background task."""
    from app.coordination.lifecycle import stop_coordinator_heartbeats as _stop

    _stop()


def is_heartbeat_running() -> bool:
    """Check if heartbeat manager is running."""
    from app.coordination.lifecycle import is_heartbeat_running as _is_running

    return _is_running()


def start_coordinator_heartbeats(interval_seconds: float = 30.0) -> bool:
    """Start the coordinator heartbeat background task."""
    from app.coordination.lifecycle import (
        start_coordinator_heartbeats as _start_coordinator_heartbeats,
    )

    return _start_coordinator_heartbeats(interval_seconds=interval_seconds)


# =============================================================================
# Combined __all__ from all submodules
# =============================================================================
__all__ = [
    *_EXPORT_NAME_TO_MODULE.keys(),
    # Consolidated modules
    "core_utils",
    "core_events",
    # Functions defined in this file
    "get_all_coordinator_status",
    "get_system_health",
    "initialize_all_coordinators",
    "is_heartbeat_running",
    "shutdown_all_coordinators",
    "start_coordinator_heartbeats",
    "stop_coordinator_heartbeats",
]

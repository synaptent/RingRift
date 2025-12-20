"""Thread Integration for Training Components.

Provides ThreadSpawner adoption utilities for training components that use
background threads. This module enables supervised thread execution with
restart policies, health checks, and unified shutdown.

Usage:
    from app.training.thread_integration import (
        get_training_thread_spawner,
        spawn_eval_thread,
        spawn_prefetch_thread,
        spawn_checkpoint_thread,
        spawn_heartbeat_thread,
    )

    # Spawn supervised evaluation thread
    thread = spawn_eval_thread(
        target=eval_loop,
        name="background_eval",
        restart_policy=RestartPolicy.ON_FAILURE,
    )

    # Check health
    health = get_training_thread_spawner().health_check()

    # Graceful shutdown
    get_training_thread_spawner().shutdown()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.core.thread_spawner import (
    RestartPolicy,
    SpawnedThread,
    ThreadGroup,
    ThreadSpawner,
    ThreadState,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RestartPolicy",
    "ThreadState",
    "TrainingThreadGroup",
    "get_training_thread_spawner",
    "spawn_checkpoint_thread",
    "spawn_eval_thread",
    "spawn_heartbeat_thread",
    "spawn_prefetch_thread",
    "spawn_selfplay_monitor_thread",
]


# =============================================================================
# Training Thread Spawner Singleton
# =============================================================================

_training_spawner: ThreadSpawner | None = None


def get_training_thread_spawner() -> ThreadSpawner:
    """Get the training-specific thread spawner.

    Returns a dedicated ThreadSpawner for training components that is
    separate from the global spawner.

    Returns:
        ThreadSpawner instance for training
    """
    global _training_spawner
    if _training_spawner is None:
        _training_spawner = ThreadSpawner(
            default_restart_policy=RestartPolicy.ON_FAILURE,
        )
    return _training_spawner


def reset_training_thread_spawner() -> None:
    """Reset the training thread spawner (for testing)."""
    global _training_spawner
    if _training_spawner:
        _training_spawner.shutdown(timeout=5.0)
    _training_spawner = None


# =============================================================================
# Specialized Thread Spawners
# =============================================================================

def spawn_eval_thread(
    target: Callable[..., Any],
    name: str = "background_eval",
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE,
    max_restarts: int = 5,
    on_error: Callable[[SpawnedThread, Exception], None] | None = None,
) -> SpawnedThread:
    """Spawn a supervised background evaluation thread.

    For use with BackgroundEvaluator._eval_loop.

    Args:
        target: Evaluation loop function
        name: Thread name
        args: Arguments for target
        kwargs: Keyword arguments for target
        restart_policy: Restart behavior on failure
        max_restarts: Maximum restart attempts
        on_error: Error callback

    Returns:
        SpawnedThread handle

    Example:
        from app.training.thread_integration import spawn_eval_thread

        def eval_loop():
            while not current_thread().should_stop():
                # ... evaluation logic
                time.sleep(5.0)

        thread = spawn_eval_thread(target=eval_loop)
    """
    spawner = get_training_thread_spawner()
    return spawner.spawn(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        restart_policy=restart_policy,
        max_restarts=max_restarts,
        daemon=True,
        on_error=on_error or _default_eval_error_handler,
    )


def spawn_prefetch_thread(
    target: Callable[..., Any],
    name: str = "data_prefetch",
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE,
    max_restarts: int = 3,
) -> SpawnedThread:
    """Spawn a supervised data prefetch thread.

    For use with PinnedMemoryLoader._prefetch_worker.

    Args:
        target: Prefetch worker function
        name: Thread name
        args: Arguments for target
        kwargs: Keyword arguments for target
        restart_policy: Restart behavior on failure
        max_restarts: Maximum restart attempts

    Returns:
        SpawnedThread handle
    """
    spawner = get_training_thread_spawner()
    return spawner.spawn(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        restart_policy=restart_policy,
        max_restarts=max_restarts,
        daemon=True,
        on_error=_default_prefetch_error_handler,
    )


def spawn_checkpoint_thread(
    target: Callable[..., Any],
    name: str = "checkpoint_save",
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    restart_policy: RestartPolicy = RestartPolicy.NEVER,
) -> SpawnedThread:
    """Spawn a supervised checkpoint save thread.

    For use with UnifiedCheckpointManager._save_thread.

    Args:
        target: Checkpoint save function
        name: Thread name
        args: Arguments for target
        kwargs: Keyword arguments for target
        restart_policy: Restart behavior on failure

    Returns:
        SpawnedThread handle
    """
    spawner = get_training_thread_spawner()
    return spawner.spawn(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        restart_policy=restart_policy,
        daemon=True,
        on_complete=_checkpoint_complete_handler,
        on_error=_checkpoint_error_handler,
    )


def spawn_heartbeat_thread(
    target: Callable[..., Any],
    name: str = "heartbeat_monitor",
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    restart_policy: RestartPolicy = RestartPolicy.ALWAYS,
    max_restarts: int = 10,
) -> SpawnedThread:
    """Spawn a supervised heartbeat monitor thread.

    For use with HeartbeatMonitor._monitor_loop.

    Args:
        target: Heartbeat monitor function
        name: Thread name
        args: Arguments for target
        kwargs: Keyword arguments for target
        restart_policy: Restart behavior on failure
        max_restarts: Maximum restart attempts

    Returns:
        SpawnedThread handle
    """
    spawner = get_training_thread_spawner()
    return spawner.spawn(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        restart_policy=restart_policy,
        max_restarts=max_restarts,
        daemon=True,
    )


def spawn_selfplay_monitor_thread(
    target: Callable[..., Any],
    name: str = "selfplay_monitor",
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE,
    max_restarts: int = 3,
) -> SpawnedThread:
    """Spawn a supervised selfplay monitor thread.

    For use with BackgroundSelfplayManager.

    Args:
        target: Selfplay monitor function
        name: Thread name
        args: Arguments for target
        kwargs: Keyword arguments for target
        restart_policy: Restart behavior on failure
        max_restarts: Maximum restart attempts

    Returns:
        SpawnedThread handle
    """
    spawner = get_training_thread_spawner()
    return spawner.spawn(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        restart_policy=restart_policy,
        max_restarts=max_restarts,
        daemon=True,
    )


# =============================================================================
# Thread Groups for Training
# =============================================================================

class TrainingThreadGroup:
    """Predefined thread groups for training components."""

    EVALUATION = "evaluation"
    DATA_LOADING = "data_loading"
    CHECKPOINTING = "checkpointing"
    MONITORING = "monitoring"
    SELFPLAY = "selfplay"


def get_training_group(name: str) -> ThreadGroup:
    """Get or create a training thread group.

    Args:
        name: Group name from TrainingThreadGroup

    Returns:
        ThreadGroup instance
    """
    spawner = get_training_thread_spawner()
    group = spawner.get_group(name)
    if group is None:
        group = spawner.create_group(name)
    return group


# =============================================================================
# Error Handlers
# =============================================================================

def _default_eval_error_handler(thread: SpawnedThread, error: Exception) -> None:
    """Default error handler for evaluation threads."""
    logger.error(
        f"[EvalThread:{thread.name}] Evaluation thread failed: {error}. "
        f"Restarts: {thread.restarts}/{thread.max_restarts}"
    )

    # Optionally publish event
    try:
        from app.core.event_bus import ErrorEvent, get_event_bus
        bus = get_event_bus()
        bus.publish_sync(ErrorEvent(
            topic="training.eval.error",
            error_type=type(error).__name__,
            error_message=str(error),
            source=thread.name,
        ))
    except ImportError:
        pass


def _default_prefetch_error_handler(thread: SpawnedThread, error: Exception) -> None:
    """Default error handler for prefetch threads."""
    logger.error(
        f"[PrefetchThread:{thread.name}] Data prefetch failed: {error}. "
        f"Restarts: {thread.restarts}/{thread.max_restarts}"
    )


def _checkpoint_complete_handler(thread: SpawnedThread) -> None:
    """Handler for checkpoint completion."""
    duration = thread.duration or 0
    logger.info(
        f"[CheckpointThread:{thread.name}] Checkpoint saved in {duration:.2f}s"
    )


def _checkpoint_error_handler(thread: SpawnedThread, error: Exception) -> None:
    """Error handler for checkpoint threads."""
    logger.error(
        f"[CheckpointThread:{thread.name}] Checkpoint save failed: {error}"
    )

    # Checkpoint failures are critical - publish event
    try:
        from app.core.event_bus import ErrorEvent, get_event_bus
        bus = get_event_bus()
        bus.publish_sync(ErrorEvent(
            topic="training.checkpoint.error",
            error_type=type(error).__name__,
            error_message=str(error),
            source=thread.name,
            metadata={"critical": True},
        ))
    except ImportError:
        pass


# =============================================================================
# Migration Helpers
# =============================================================================

def migrate_to_supervised(
    _start_method: Callable[[], None],
    _stop_method: Callable[[], None],
    thread_attr: str,
    running_attr: str,
    loop_method: Callable[[], None],
    instance: Any,
    thread_name: str,
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE,
) -> Callable[[], SpawnedThread]:
    """Create a supervised start method from legacy thread code.

    This helper assists in migrating bare threading.Thread usage to
    supervised ThreadSpawner execution.

    Args:
        start_method: Original start() method
        stop_method: Original stop() method
        thread_attr: Name of _thread attribute (e.g., "_eval_thread")
        running_attr: Name of _running attribute (e.g., "_running")
        loop_method: The loop method to run (e.g., self._eval_loop)
        instance: The object instance
        thread_name: Name for the supervised thread
        restart_policy: Restart policy

    Returns:
        New supervised start method

    Example:
        # In BackgroundEvaluator.__init__:
        self.start = migrate_to_supervised(
            start_method=self.start,
            stop_method=self.stop,
            thread_attr="_eval_thread",
            running_attr="_running",
            loop_method=self._eval_loop,
            instance=self,
            thread_name="background_eval",
        )
    """
    def supervised_start() -> SpawnedThread:
        # Set running flag
        setattr(instance, running_attr, True)

        # Spawn supervised thread
        thread = spawn_eval_thread(
            target=loop_method,
            name=thread_name,
            restart_policy=restart_policy,
        )

        # Store reference for compatibility
        setattr(instance, thread_attr, thread._thread)
        instance._supervised_thread = thread

        logger.info(f"[{thread_name}] Started supervised thread")
        return thread

    return supervised_start


# =============================================================================
# Health Integration
# =============================================================================

def get_training_threads_health() -> dict[str, Any]:
    """Get health status of all training threads.

    Returns:
        Health status dict suitable for API response
    """
    spawner = get_training_thread_spawner()
    health = spawner.health_check()
    stats = spawner.get_stats()

    return {
        "status": "healthy" if health["healthy"] else "degraded",
        "threads": {
            "running": health["running"],
            "completed": health["completed"],
            "failed": health["failed"],
            "total_spawned": stats["threads_spawned"],
            "total_restarts": stats["total_restarts"],
        },
        "unhealthy_threads": health["unhealthy_threads"],
    }

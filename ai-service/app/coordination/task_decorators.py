"""Task Lifecycle Decorators for RingRift AI (December 2025).

Provides decorators that automatically handle task lifecycle management:
- Task registration with TaskCoordinator
- Event emission on start/complete/fail
- Automatic cleanup on exceptions
- Performance tracking

Usage:
    from app.coordination.task_decorators import (
        coordinate_task,
        coordinate_async_task,
        TaskContext,
    )

    @coordinate_task(task_type="selfplay", emit_events=True)
    def run_selfplay_batch(board_type: str, num_games: int) -> dict:
        # Task is automatically registered and events emitted
        games = generate_games(board_type, num_games)
        return {"games_generated": len(games)}


    @coordinate_async_task(task_type="training")
    async def train_model(config: TrainingConfig) -> TrainingResult:
        # Async tasks work the same way
        result = await trainer.train(config)
        return result

Benefits:
- Eliminates boilerplate task lifecycle code
- Ensures consistent event emission
- Automatic error handling and cleanup
- Performance metrics collection
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variables for generic decorators
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


@dataclass
class TaskContext:
    """Context information for a coordinated task.

    Accessible within decorated functions via get_current_task_context().
    """

    task_id: str
    task_type: str
    start_time: float
    board_type: Optional[str] = None
    num_players: Optional[int] = None
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def elapsed_seconds(self) -> float:
        """Get elapsed time since task start."""
        return time.time() - self.start_time


# Thread-local storage for task context
import threading
_task_context_var = threading.local()


def get_current_task_context() -> Optional[TaskContext]:
    """Get the current task context if inside a coordinated task.

    Returns:
        TaskContext or None if not inside a coordinated task

    Example:
        @coordinate_task(task_type="selfplay")
        def generate_games():
            ctx = get_current_task_context()
            print(f"Running task {ctx.task_id}")
    """
    return getattr(_task_context_var, 'context', None)


def _set_task_context(ctx: Optional[TaskContext]) -> None:
    """Set the current task context (internal use)."""
    _task_context_var.context = ctx


def _generate_task_id(task_type: str) -> str:
    """Generate a unique task ID."""
    timestamp = int(time.time())
    unique = uuid.uuid4().hex[:8]
    return f"{task_type}_{timestamp}_{unique}"


def _extract_board_info(
    args: tuple,
    kwargs: dict,
    board_type_param: str = "board_type",
    num_players_param: str = "num_players",
) -> tuple[Optional[str], Optional[int]]:
    """Extract board_type and num_players from function arguments."""
    board_type = kwargs.get(board_type_param)
    num_players = kwargs.get(num_players_param)

    # Try to get from config object if present
    if not board_type:
        config = kwargs.get("config") or (args[0] if args and hasattr(args[0], 'board_type') else None)
        if config and hasattr(config, 'board_type'):
            board_type = getattr(config, 'board_type')
            if hasattr(config, 'num_players'):
                num_players = getattr(config, 'num_players')

    return board_type, num_players


def _emit_task_started(
    ctx: TaskContext,
    emit_events: bool,
) -> None:
    """Emit task started event."""
    if not emit_events:
        return

    try:
        from app.coordination.event_emitters import emit_training_started

        if ctx.task_type == "training":
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(emit_training_started(
                    job_id=ctx.task_id,
                    board_type=ctx.board_type or "",
                    num_players=ctx.num_players or 0,
                    **ctx.metadata,
                ))
            except RuntimeError:
                # No running event loop, skip event emission
                pass

        logger.debug(f"[coordinate_task] Started {ctx.task_type} task {ctx.task_id}")

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[coordinate_task] Event emission failed: {e}")


async def _emit_task_complete_async(
    ctx: TaskContext,
    result: Any,
    emit_events: bool,
) -> None:
    """Emit task complete event (async)."""
    if not emit_events:
        return

    try:
        from app.coordination.event_emitters import emit_task_complete

        # Extract result data
        result_data = {}
        if isinstance(result, dict):
            result_data = result
        elif hasattr(result, '__dict__'):
            result_data = result.__dict__

        await emit_task_complete(
            task_id=ctx.task_id,
            task_type=ctx.task_type,
            success=True,
            node_id=ctx.node_id or "",
            duration_seconds=ctx.elapsed_seconds(),
            result_data=result_data,
        )

        logger.debug(
            f"[coordinate_task] Completed {ctx.task_type} task {ctx.task_id} "
            f"in {ctx.elapsed_seconds():.1f}s"
        )

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[coordinate_task] Event emission failed: {e}")


def _emit_task_complete_sync(
    ctx: TaskContext,
    result: Any,
    emit_events: bool,
) -> None:
    """Emit task complete event (sync, fire-and-forget)."""
    if not emit_events:
        return

    try:
        # Try to schedule async emission
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(_emit_task_complete_async(ctx, result, emit_events))
        except RuntimeError:
            # No event loop, use sync fallback
            from app.coordination.event_emitters import emit_training_complete_sync

            if ctx.task_type == "training":
                emit_training_complete_sync(
                    job_id=ctx.task_id,
                    board_type=ctx.board_type or "",
                    num_players=ctx.num_players or 0,
                    success=True,
                )

        logger.debug(
            f"[coordinate_task] Completed {ctx.task_type} task {ctx.task_id} "
            f"in {ctx.elapsed_seconds():.1f}s"
        )

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[coordinate_task] Event emission failed: {e}")


async def _emit_task_failed_async(
    ctx: TaskContext,
    error: Exception,
    emit_events: bool,
) -> None:
    """Emit task failed event (async)."""
    if not emit_events:
        return

    try:
        from app.coordination.event_emitters import emit_task_complete

        await emit_task_complete(
            task_id=ctx.task_id,
            task_type=ctx.task_type,
            success=False,
            node_id=ctx.node_id or "",
            duration_seconds=ctx.elapsed_seconds(),
            result_data={"error": str(error)},
        )

        logger.debug(
            f"[coordinate_task] Failed {ctx.task_type} task {ctx.task_id}: {error}"
        )

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[coordinate_task] Event emission failed: {e}")


def _emit_task_failed_sync(
    ctx: TaskContext,
    error: Exception,
    emit_events: bool,
) -> None:
    """Emit task failed event (sync, fire-and-forget)."""
    if not emit_events:
        return

    try:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(_emit_task_failed_async(ctx, error, emit_events))
        except RuntimeError:
            from app.coordination.event_emitters import emit_training_complete_sync

            if ctx.task_type == "training":
                emit_training_complete_sync(
                    job_id=ctx.task_id,
                    board_type=ctx.board_type or "",
                    num_players=ctx.num_players or 0,
                    success=False,
                )

        logger.debug(
            f"[coordinate_task] Failed {ctx.task_type} task {ctx.task_id}: {error}"
        )

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[coordinate_task] Event emission failed: {e}")


def coordinate_task(
    task_type: str,
    emit_events: bool = True,
    register_with_coordinator: bool = True,
    board_type_param: str = "board_type",
    num_players_param: str = "num_players",
) -> Callable[[F], F]:
    """Decorator for coordinated synchronous tasks.

    Automatically handles:
    - Task ID generation
    - Event emission on start/complete/fail
    - TaskCoordinator registration (optional)
    - TaskContext availability

    Args:
        task_type: Type of task (selfplay, training, evaluation, sync, etc.)
        emit_events: Whether to emit StageEvent events
        register_with_coordinator: Whether to register with TaskCoordinator
        board_type_param: Parameter name for board_type extraction
        num_players_param: Parameter name for num_players extraction

    Returns:
        Decorated function

    Example:
        @coordinate_task(task_type="selfplay")
        def run_selfplay(board_type: str, num_games: int) -> dict:
            ctx = get_current_task_context()
            games = generate(board_type, num_games)
            return {"games_generated": len(games)}
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate task ID and context
            task_id = _generate_task_id(task_type)
            board_type, num_players = _extract_board_info(
                args, kwargs, board_type_param, num_players_param
            )

            ctx = TaskContext(
                task_id=task_id,
                task_type=task_type,
                start_time=time.time(),
                board_type=board_type,
                num_players=num_players,
            )

            # Set task context
            _set_task_context(ctx)

            # Register with coordinator if enabled
            if register_with_coordinator:
                try:
                    from app.coordination.task_coordinator import get_task_coordinator
                    coordinator = get_task_coordinator()
                    if coordinator:
                        coordinator.register_task(task_id, task_type, ctx.metadata)
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"[coordinate_task] Coordinator registration failed: {e}")

            # Emit start event
            _emit_task_started(ctx, emit_events)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Emit completion event
                _emit_task_complete_sync(ctx, result, emit_events)

                return result

            except Exception as e:
                # Emit failure event
                _emit_task_failed_sync(ctx, e, emit_events)
                raise

            finally:
                # Clear context
                _set_task_context(None)

        return wrapper  # type: ignore

    return decorator


def coordinate_async_task(
    task_type: str,
    emit_events: bool = True,
    register_with_coordinator: bool = True,
    board_type_param: str = "board_type",
    num_players_param: str = "num_players",
) -> Callable[[F], F]:
    """Decorator for coordinated asynchronous tasks.

    Same as coordinate_task but for async functions.

    Args:
        task_type: Type of task (selfplay, training, evaluation, sync, etc.)
        emit_events: Whether to emit StageEvent events
        register_with_coordinator: Whether to register with TaskCoordinator
        board_type_param: Parameter name for board_type extraction
        num_players_param: Parameter name for num_players extraction

    Returns:
        Decorated async function

    Example:
        @coordinate_async_task(task_type="training")
        async def train_model(config: TrainingConfig) -> TrainingResult:
            result = await trainer.train(config)
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate task ID and context
            task_id = _generate_task_id(task_type)
            board_type, num_players = _extract_board_info(
                args, kwargs, board_type_param, num_players_param
            )

            ctx = TaskContext(
                task_id=task_id,
                task_type=task_type,
                start_time=time.time(),
                board_type=board_type,
                num_players=num_players,
            )

            # Set task context
            _set_task_context(ctx)

            # Register with coordinator if enabled
            if register_with_coordinator:
                try:
                    from app.coordination.task_coordinator import get_task_coordinator
                    coordinator = get_task_coordinator()
                    if coordinator:
                        coordinator.register_task(task_id, task_type, ctx.metadata)
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"[coordinate_async_task] Coordinator registration failed: {e}")

            # Emit start event
            _emit_task_started(ctx, emit_events)

            try:
                # Execute async function
                result = await func(*args, **kwargs)

                # Emit completion event
                await _emit_task_complete_async(ctx, result, emit_events)

                return result

            except Exception as e:
                # Emit failure event
                await _emit_task_failed_async(ctx, e, emit_events)
                raise

            finally:
                # Clear context
                _set_task_context(None)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def task_context(
    task_type: str,
    task_id: Optional[str] = None,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    emit_events: bool = True,
):
    """Context manager for coordinated task blocks.

    Use this when you need task coordination but can't use decorators.

    Args:
        task_type: Type of task
        task_id: Optional task ID (generated if not provided)
        board_type: Board type
        num_players: Number of players
        emit_events: Whether to emit events

    Yields:
        TaskContext instance

    Example:
        with task_context(task_type="selfplay", board_type="square8") as ctx:
            print(f"Running task {ctx.task_id}")
            # ... do work ...
    """
    ctx = TaskContext(
        task_id=task_id or _generate_task_id(task_type),
        task_type=task_type,
        start_time=time.time(),
        board_type=board_type,
        num_players=num_players,
    )

    _set_task_context(ctx)
    _emit_task_started(ctx, emit_events)

    try:
        yield ctx
        _emit_task_complete_sync(ctx, {}, emit_events)
    except Exception as e:
        _emit_task_failed_sync(ctx, e, emit_events)
        raise
    finally:
        _set_task_context(None)


__all__ = [
    # Decorators
    "coordinate_task",
    "coordinate_async_task",
    # Context
    "TaskContext",
    "get_current_task_context",
    "task_context",
]

"""Task Lifecycle Integration for Training Components.

Bridges training tasks (data loading, training jobs, evaluation, selfplay)
to the TaskLifecycleCoordinator for unified monitoring and orphan detection.

Usage:
    from app.training.task_lifecycle_integration import (
        register_training_job,
        register_data_loader_task,
        send_training_heartbeat,
        complete_training_task,
        TrainingTaskTracker,
    )

    # Register a training job
    task = register_training_job(
        job_id="job-123",
        config_key="square8_2p",
        node_id="gh200-a",
    )

    # Send heartbeat during training
    send_training_heartbeat(task_id="job-123")

    # Complete the task
    complete_training_task(
        task_id="job-123",
        success=True,
        result={"loss": 0.01, "elo": 1520},
    )
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from app.coordination.task_lifecycle_coordinator import (
    get_task_lifecycle_coordinator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HeartbeatSender",
    "TrainingTaskTracker",
    "TrainingTaskType",
    "complete_training_task",
    "fail_training_task",
    "get_training_task_tracker",
    "register_data_loader_task",
    "register_evaluation_task",
    "register_selfplay_task",
    "register_training_job",
    "send_training_heartbeat",
    "training_task_context",
]


# =============================================================================
# Task Types
# =============================================================================

class TrainingTaskType:
    """Task type constants for training components."""
    TRAINING_JOB = "training_job"
    DATA_LOADING = "data_loading"
    EVALUATION = "evaluation"
    SELFPLAY = "selfplay"
    CHECKPOINT = "checkpoint"
    PREFETCH = "prefetch"


# =============================================================================
# Training Task Tracker
# =============================================================================

@dataclass
class TrainingTaskInfo:
    """Extended info for training tasks."""
    task_id: str
    task_type: str
    config_key: str
    node_id: str
    job_id: str | None = None
    started_at: float = field(default_factory=time.time)
    step: int = 0
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class TrainingTaskTracker:
    """Tracks training tasks and integrates with TaskLifecycleCoordinator.

    Provides:
    - Task registration with training-specific metadata
    - Automatic heartbeat sending
    - Step/epoch progress updates
    - Integration with lifecycle coordinator
    """

    def __init__(self, node_id: str = ""):
        """Initialize tracker.

        Args:
            node_id: Node identifier for this instance
        """
        self.node_id = node_id or self._get_node_id()
        self._tasks: dict[str, TrainingTaskInfo] = {}
        self._coordinator = get_task_lifecycle_coordinator()
        self._lock = threading.Lock()
        self._heartbeat_threads: dict[str, HeartbeatSender] = {}

    def _get_node_id(self) -> str:
        """Get node ID from environment or hostname."""
        import os
        import socket
        return os.environ.get("NODE_ID", socket.gethostname())

    def register_job(
        self,
        job_id: str,
        config_key: str,
        node_id: str | None = None,
        auto_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
    ) -> TrainingTaskInfo:
        """Register a training job.

        Args:
            job_id: Job identifier
            config_key: Configuration key
            node_id: Node ID (uses default if not provided)
            auto_heartbeat: Start automatic heartbeat
            heartbeat_interval: Heartbeat interval in seconds

        Returns:
            TrainingTaskInfo for the registered job
        """
        task_id = f"training:{job_id}"
        node = node_id or self.node_id

        # Register with coordinator
        self._coordinator.register_task(
            task_id=task_id,
            task_type=TrainingTaskType.TRAINING_JOB,
            node_id=node,
        )

        # Create training-specific info
        info = TrainingTaskInfo(
            task_id=task_id,
            task_type=TrainingTaskType.TRAINING_JOB,
            config_key=config_key,
            node_id=node,
            job_id=job_id,
        )

        with self._lock:
            self._tasks[task_id] = info

        if auto_heartbeat:
            self._start_heartbeat(task_id, heartbeat_interval)

        logger.info(f"[TrainingTaskTracker] Registered job: {job_id} ({config_key})")
        return info

    def register_data_loader(
        self,
        loader_id: str,
        config_key: str,
        node_id: str | None = None,
    ) -> TrainingTaskInfo:
        """Register a data loader task.

        Args:
            loader_id: Loader identifier
            config_key: Configuration key
            node_id: Node ID

        Returns:
            TrainingTaskInfo
        """
        task_id = f"data_loader:{loader_id}"
        node = node_id or self.node_id

        self._coordinator.register_task(
            task_id=task_id,
            task_type=TrainingTaskType.DATA_LOADING,
            node_id=node,
        )

        info = TrainingTaskInfo(
            task_id=task_id,
            task_type=TrainingTaskType.DATA_LOADING,
            config_key=config_key,
            node_id=node,
        )

        with self._lock:
            self._tasks[task_id] = info

        logger.debug(f"[TrainingTaskTracker] Registered data loader: {loader_id}")
        return info

    def register_evaluation(
        self,
        eval_id: str,
        config_key: str,
        job_id: str | None = None,
        node_id: str | None = None,
    ) -> TrainingTaskInfo:
        """Register an evaluation task.

        Args:
            eval_id: Evaluation identifier
            config_key: Configuration key
            job_id: Associated training job
            node_id: Node ID

        Returns:
            TrainingTaskInfo
        """
        task_id = f"eval:{eval_id}"
        node = node_id or self.node_id

        self._coordinator.register_task(
            task_id=task_id,
            task_type=TrainingTaskType.EVALUATION,
            node_id=node,
        )

        info = TrainingTaskInfo(
            task_id=task_id,
            task_type=TrainingTaskType.EVALUATION,
            config_key=config_key,
            node_id=node,
            job_id=job_id,
        )

        with self._lock:
            self._tasks[task_id] = info

        logger.debug(f"[TrainingTaskTracker] Registered evaluation: {eval_id}")
        return info

    def register_selfplay(
        self,
        selfplay_id: str,
        config_key: str,
        iteration: int,
        node_id: str | None = None,
    ) -> TrainingTaskInfo:
        """Register a selfplay task.

        Args:
            selfplay_id: Selfplay identifier
            config_key: Configuration key
            iteration: Training iteration
            node_id: Node ID

        Returns:
            TrainingTaskInfo
        """
        task_id = f"selfplay:{selfplay_id}"
        node = node_id or self.node_id

        self._coordinator.register_task(
            task_id=task_id,
            task_type=TrainingTaskType.SELFPLAY,
            node_id=node,
        )

        info = TrainingTaskInfo(
            task_id=task_id,
            task_type=TrainingTaskType.SELFPLAY,
            config_key=config_key,
            node_id=node,
        )
        info.step = iteration

        with self._lock:
            self._tasks[task_id] = info

        logger.debug(f"[TrainingTaskTracker] Registered selfplay: {selfplay_id}")
        return info

    def heartbeat(self, task_id: str, step: int | None = None, metrics: dict[str, float] | None = None) -> bool:
        """Send heartbeat for a task.

        Args:
            task_id: Task identifier
            step: Current training step
            metrics: Current metrics

        Returns:
            True if heartbeat was sent
        """
        result = self._coordinator.update_heartbeat(task_id)

        # Update local info
        with self._lock:
            if task_id in self._tasks:
                if step is not None:
                    self._tasks[task_id].step = step
                if metrics:
                    self._tasks[task_id].metrics.update(metrics)

        return result

    def complete(
        self,
        task_id: str,
        success: bool = True,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark a task as completed.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            result: Result data
        """
        # Stop heartbeat if running
        self._stop_heartbeat(task_id)

        # Remove from local tracking
        with self._lock:
            self._tasks.pop(task_id, None)

        # Emit event to coordinator
        try:
            import asyncio

            from app.coordination.event_emitters import emit_task_completed, emit_task_failed

            if success:
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(emit_task_completed(
                        task_id=task_id,
                        result=result or {},
                    ))
                    task.add_done_callback(
                        lambda t: logger.debug(f"Task complete event error: {t.exception()}")
                        if t.exception() else None
                    )
                except RuntimeError:
                    asyncio.run(emit_task_completed(
                        task_id=task_id,
                        result=result or {},
                    ))
            else:
                error = result.get("error", "Unknown error") if result else "Unknown error"
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(emit_task_failed(
                        task_id=task_id,
                        error=error,
                    ))
                    task.add_done_callback(
                        lambda t: logger.debug(f"Task failed event error: {t.exception()}")
                        if t.exception() else None
                    )
                except RuntimeError:
                    asyncio.run(emit_task_failed(
                        task_id=task_id,
                        error=error,
                    ))
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[TrainingTaskTracker] Failed to emit completion event: {e}")

        logger.info(f"[TrainingTaskTracker] Task completed: {task_id} (success={success})")

    def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: Task identifier
            error: Error message
        """
        self.complete(task_id, success=False, result={"error": error})

    def get_task(self, task_id: str) -> TrainingTaskInfo | None:
        """Get task info by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TrainingTaskInfo]:
        """Get all tracked tasks."""
        with self._lock:
            return list(self._tasks.values())

    def _start_heartbeat(self, task_id: str, interval: float) -> None:
        """Start automatic heartbeat for a task."""
        if task_id in self._heartbeat_threads:
            return

        sender = HeartbeatSender(
            task_id=task_id,
            tracker=self,
            interval=interval,
        )
        sender.start()
        self._heartbeat_threads[task_id] = sender

    def _stop_heartbeat(self, task_id: str) -> None:
        """Stop automatic heartbeat for a task."""
        sender = self._heartbeat_threads.pop(task_id, None)
        if sender:
            sender.stop()

    def shutdown(self) -> None:
        """Shutdown tracker and stop all heartbeats."""
        for task_id in list(self._heartbeat_threads.keys()):
            self._stop_heartbeat(task_id)

        logger.info("[TrainingTaskTracker] Shutdown complete")


class HeartbeatSender:
    """Sends periodic heartbeats for a task."""

    def __init__(
        self,
        task_id: str,
        tracker: TrainingTaskTracker,
        interval: float = 30.0,
    ):
        self.task_id = task_id
        self.tracker = tracker
        self.interval = interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start heartbeat thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"heartbeat-{self.task_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop heartbeat thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _heartbeat_loop(self) -> None:
        """Send heartbeats periodically."""
        while self._running:
            try:
                self.tracker.heartbeat(self.task_id)
            except Exception as e:
                logger.debug(f"[HeartbeatSender] Error: {e}")

            # Sleep in small increments for faster shutdown
            elapsed = 0.0
            while elapsed < self.interval and self._running:
                time.sleep(0.5)
                elapsed += 0.5


# =============================================================================
# Global Instance and Convenience Functions
# =============================================================================

_training_tracker: TrainingTaskTracker | None = None


def get_training_task_tracker() -> TrainingTaskTracker:
    """Get the global training task tracker."""
    global _training_tracker
    if _training_tracker is None:
        _training_tracker = TrainingTaskTracker()
    return _training_tracker


def reset_training_task_tracker() -> None:
    """Reset the training task tracker (for testing)."""
    global _training_tracker
    if _training_tracker:
        _training_tracker.shutdown()
    _training_tracker = None


def register_training_job(
    job_id: str,
    config_key: str,
    node_id: str | None = None,
    auto_heartbeat: bool = True,
) -> TrainingTaskInfo:
    """Register a training job with the global tracker."""
    return get_training_task_tracker().register_job(
        job_id=job_id,
        config_key=config_key,
        node_id=node_id,
        auto_heartbeat=auto_heartbeat,
    )


def register_data_loader_task(
    loader_id: str,
    config_key: str,
) -> TrainingTaskInfo:
    """Register a data loader task with the global tracker."""
    return get_training_task_tracker().register_data_loader(
        loader_id=loader_id,
        config_key=config_key,
    )


def register_evaluation_task(
    eval_id: str,
    config_key: str,
    job_id: str | None = None,
) -> TrainingTaskInfo:
    """Register an evaluation task with the global tracker."""
    return get_training_task_tracker().register_evaluation(
        eval_id=eval_id,
        config_key=config_key,
        job_id=job_id,
    )


def register_selfplay_task(
    selfplay_id: str,
    config_key: str,
    iteration: int,
) -> TrainingTaskInfo:
    """Register a selfplay task with the global tracker."""
    return get_training_task_tracker().register_selfplay(
        selfplay_id=selfplay_id,
        config_key=config_key,
        iteration=iteration,
    )


def send_training_heartbeat(
    task_id: str,
    step: int | None = None,
    metrics: dict[str, float] | None = None,
) -> bool:
    """Send heartbeat for a training task."""
    return get_training_task_tracker().heartbeat(
        task_id=task_id,
        step=step,
        metrics=metrics,
    )


def complete_training_task(
    task_id: str,
    success: bool = True,
    result: dict[str, Any] | None = None,
) -> None:
    """Mark a training task as completed."""
    get_training_task_tracker().complete(
        task_id=task_id,
        success=success,
        result=result,
    )


def fail_training_task(task_id: str, error: str) -> None:
    """Mark a training task as failed."""
    get_training_task_tracker().fail(task_id=task_id, error=error)


@contextmanager
def training_task_context(
    task_type: str,
    task_id: str,
    config_key: str,
    **kwargs: Any,
):
    """Context manager for training tasks.

    Automatically registers task on entry and completes on exit.

    Args:
        task_type: Type of task (from TrainingTaskType)
        task_id: Task identifier
        config_key: Configuration key
        **kwargs: Additional arguments for registration

    Yields:
        TrainingTaskInfo

    Example:
        with training_task_context(
            TrainingTaskType.TRAINING_JOB,
            "job-123",
            "square8_2p",
        ) as task:
            # Training code here
            pass
        # Task automatically completed on exit
    """
    tracker = get_training_task_tracker()

    if task_type == TrainingTaskType.TRAINING_JOB:
        info = tracker.register_job(job_id=task_id, config_key=config_key, **kwargs)
    elif task_type == TrainingTaskType.DATA_LOADING:
        info = tracker.register_data_loader(loader_id=task_id, config_key=config_key)
    elif task_type == TrainingTaskType.EVALUATION:
        info = tracker.register_evaluation(eval_id=task_id, config_key=config_key, **kwargs)
    elif task_type == TrainingTaskType.SELFPLAY:
        info = tracker.register_selfplay(selfplay_id=task_id, config_key=config_key, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    try:
        yield info
        tracker.complete(info.task_id, success=True)
    except Exception as e:
        tracker.fail(info.task_id, str(e))
        raise

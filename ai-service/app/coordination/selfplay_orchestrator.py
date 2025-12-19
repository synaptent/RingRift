"""SelfplayOrchestrator - Unified selfplay event coordination (December 2025).

This module provides centralized monitoring and coordination of selfplay events.
It tracks selfplay tasks across nodes, emits completion events, and coordinates
between background selfplay and the broader pipeline.

Event Integration:
- Subscribes to TASK_SPAWNED: Track new selfplay tasks
- Subscribes to TASK_COMPLETED: Track selfplay completions
- Subscribes to TASK_FAILED: Track selfplay failures
- Emits SELFPLAY_COMPLETE: When canonical selfplay finishes
- Emits GPU_SELFPLAY_COMPLETE: When GPU-accelerated selfplay finishes

Key Responsibilities:
1. Track active selfplay tasks across all nodes
2. Emit completion events to trigger downstream pipeline stages
3. Coordinate background selfplay with main pipeline
4. Provide selfplay statistics and throughput metrics

Usage:
    from app.coordination.selfplay_orchestrator import (
        SelfplayOrchestrator,
        wire_selfplay_events,
        get_selfplay_orchestrator,
        emit_selfplay_completion,
    )

    # Wire selfplay events
    orchestrator = wire_selfplay_events()

    # Get selfplay statistics
    stats = orchestrator.get_stats()
    print(f"Active selfplay tasks: {stats['active_tasks']}")
    print(f"Games generated: {stats['total_games_generated']}")

    # Emit completion event (call when selfplay finishes)
    await emit_selfplay_completion(
        task_id="selfplay-123",
        board_type="square8",
        num_players=2,
        games_generated=500,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SelfplayType(Enum):
    """Types of selfplay tasks."""

    CANONICAL = "canonical"  # Standard MCTS selfplay
    GPU_ACCELERATED = "gpu_selfplay"  # GPU-accelerated selfplay
    HYBRID = "hybrid_selfplay"  # Mixed CPU/GPU selfplay
    BACKGROUND = "background"  # Background pipeline selfplay


@dataclass
class SelfplayTaskInfo:
    """Information about a selfplay task."""

    task_id: str
    selfplay_type: SelfplayType
    node_id: str
    board_type: str = "square8"
    num_players: int = 2
    games_requested: int = 0
    games_generated: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    success: bool = False
    error: str = ""
    iteration: int = 0

    @property
    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def games_per_second(self) -> float:
        """Get games per second throughput."""
        duration = self.duration
        if duration > 0 and self.games_generated > 0:
            return self.games_generated / duration
        return 0.0


@dataclass
class SelfplayStats:
    """Aggregate selfplay statistics."""

    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_games_generated: int = 0
    total_duration_seconds: float = 0.0
    average_games_per_task: float = 0.0
    average_games_per_second: float = 0.0
    by_node: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    last_activity_time: float = 0.0


class SelfplayOrchestrator:
    """Orchestrates selfplay coordination across the cluster.

    Tracks selfplay tasks, emits completion events, and provides
    unified monitoring of selfplay activity.
    """

    def __init__(
        self,
        max_history: int = 500,
        stats_window_seconds: float = 3600.0,  # 1 hour
    ):
        """Initialize SelfplayOrchestrator.

        Args:
            max_history: Maximum number of completed tasks to retain
            stats_window_seconds: Time window for throughput calculations
        """
        self.max_history = max_history
        self.stats_window_seconds = stats_window_seconds

        # Active tasks by task_id
        self._active_tasks: Dict[str, SelfplayTaskInfo] = {}

        # Completed task history
        self._completed_history: List[SelfplayTaskInfo] = []

        # Statistics
        self._total_games_generated: int = 0
        self._total_duration_seconds: float = 0.0

        # Subscription state
        self._subscribed: bool = False

        # Callbacks for completion events
        self._completion_callbacks: List[Callable[[SelfplayTaskInfo], None]] = []

    def subscribe_to_events(self) -> bool:
        """Subscribe to selfplay-related events from the event bus.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()

            # Subscribe to task lifecycle events
            bus.subscribe(DataEventType.TASK_SPAWNED, self._on_task_spawned)
            bus.subscribe(DataEventType.TASK_COMPLETED, self._on_task_completed)
            bus.subscribe(DataEventType.TASK_FAILED, self._on_task_failed)

            self._subscribed = True
            logger.info("[SelfplayOrchestrator] Subscribed to task lifecycle events")
            return True

        except ImportError:
            logger.warning("[SelfplayOrchestrator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[SelfplayOrchestrator] Failed to subscribe: {e}")
            return False

    def subscribe_to_stage_events(self) -> bool:
        """Subscribe to stage completion events.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.stage_events import StageEvent, get_event_bus

            bus = get_event_bus()

            # Subscribe to selfplay stage events
            bus.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_stage_complete)
            bus.subscribe(
                StageEvent.GPU_SELFPLAY_COMPLETE, self._on_gpu_selfplay_stage_complete
            )
            bus.subscribe(
                StageEvent.CANONICAL_SELFPLAY_COMPLETE,
                self._on_canonical_selfplay_stage_complete,
            )

            logger.info("[SelfplayOrchestrator] Subscribed to stage events")
            return True

        except ImportError:
            logger.warning("[SelfplayOrchestrator] stage_events not available")
            return False
        except Exception as e:
            logger.error(f"[SelfplayOrchestrator] Failed to subscribe to stages: {e}")
            return False

    def _is_selfplay_task(self, task_type: str) -> bool:
        """Check if task type is a selfplay task."""
        selfplay_types = {
            "selfplay",
            "gpu_selfplay",
            "hybrid_selfplay",
            "canonical_selfplay",
            "background_selfplay",
        }
        return task_type.lower() in selfplay_types

    def _get_selfplay_type(self, task_type: str) -> SelfplayType:
        """Convert task type string to SelfplayType enum."""
        mapping = {
            "selfplay": SelfplayType.CANONICAL,
            "canonical_selfplay": SelfplayType.CANONICAL,
            "gpu_selfplay": SelfplayType.GPU_ACCELERATED,
            "hybrid_selfplay": SelfplayType.HYBRID,
            "background_selfplay": SelfplayType.BACKGROUND,
        }
        return mapping.get(task_type.lower(), SelfplayType.CANONICAL)

    async def _on_task_spawned(self, event) -> None:
        """Handle TASK_SPAWNED event."""
        payload = event.payload
        task_type = payload.get("task_type", "")

        if not self._is_selfplay_task(task_type):
            return

        task_id = payload.get("task_id", "")
        node_id = payload.get("node_id", "")

        task = SelfplayTaskInfo(
            task_id=task_id,
            selfplay_type=self._get_selfplay_type(task_type),
            node_id=node_id,
            board_type=payload.get("board_type", "square8"),
            num_players=payload.get("num_players", 2),
            games_requested=payload.get("games_requested", 0),
            iteration=payload.get("iteration", 0),
        )

        self._active_tasks[task_id] = task
        logger.debug(
            f"[SelfplayOrchestrator] Tracking selfplay task {task_id} on {node_id}"
        )

    async def _on_task_completed(self, event) -> None:
        """Handle TASK_COMPLETED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        if task_id not in self._active_tasks:
            # Check if it's a selfplay task we missed
            task_type = payload.get("task_type", "")
            if self._is_selfplay_task(task_type):
                # Create a task info from completion data
                task = SelfplayTaskInfo(
                    task_id=task_id,
                    selfplay_type=self._get_selfplay_type(task_type),
                    node_id=payload.get("node_id", ""),
                    games_generated=payload.get("games_generated", 0),
                    success=True,
                )
                self._record_completion(task, payload)
            return

        task = self._active_tasks.pop(task_id)
        task.success = True
        task.end_time = time.time()
        task.games_generated = payload.get(
            "games_generated", payload.get("result", {}).get("games_generated", 0)
        )

        self._record_completion(task, payload)

        # Emit stage completion event
        await self._emit_selfplay_complete(task)

    async def _on_task_failed(self, event) -> None:
        """Handle TASK_FAILED event."""
        payload = event.payload
        task_id = payload.get("task_id", "")

        if task_id not in self._active_tasks:
            return

        task = self._active_tasks.pop(task_id)
        task.success = False
        task.end_time = time.time()
        task.error = payload.get("error", "Unknown error")

        self._record_completion(task, payload)

        logger.warning(
            f"[SelfplayOrchestrator] Selfplay task {task_id} failed: {task.error}"
        )

    def _record_completion(self, task: SelfplayTaskInfo, payload: Dict) -> None:
        """Record task completion in history and stats."""
        # Update statistics
        if task.success:
            self._total_games_generated += task.games_generated
            self._total_duration_seconds += task.duration

        # Add to history
        self._completed_history.append(task)

        # Trim history
        if len(self._completed_history) > self.max_history:
            self._completed_history = self._completed_history[-self.max_history :]

        # Notify callbacks
        for callback in self._completion_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"[SelfplayOrchestrator] Callback error: {e}")

        logger.info(
            f"[SelfplayOrchestrator] Selfplay {'completed' if task.success else 'failed'}: "
            f"{task.task_id} on {task.node_id}, games={task.games_generated}, "
            f"duration={task.duration:.1f}s"
        )

    async def _emit_selfplay_complete(self, task: SelfplayTaskInfo) -> bool:
        """Emit SELFPLAY_COMPLETE stage event."""
        try:
            from app.coordination.stage_events import (
                StageCompletionResult,
                StageEvent,
                get_event_bus,
            )
            from datetime import datetime

            # Determine stage event type based on selfplay type
            if task.selfplay_type == SelfplayType.GPU_ACCELERATED:
                event_type = StageEvent.GPU_SELFPLAY_COMPLETE
            elif task.selfplay_type == SelfplayType.CANONICAL:
                event_type = StageEvent.CANONICAL_SELFPLAY_COMPLETE
            else:
                event_type = StageEvent.SELFPLAY_COMPLETE

            result = StageCompletionResult(
                event=event_type,
                success=task.success,
                iteration=task.iteration,
                timestamp=datetime.now().isoformat(),
                board_type=task.board_type,
                num_players=task.num_players,
                games_generated=task.games_generated,
                error=task.error if not task.success else None,
                metadata={
                    "task_id": task.task_id,
                    "node_id": task.node_id,
                    "duration_seconds": task.duration,
                    "games_per_second": task.games_per_second,
                    "selfplay_type": task.selfplay_type.value,
                },
            )

            bus = get_event_bus()
            await bus.emit(result)

            logger.debug(f"[SelfplayOrchestrator] Emitted {event_type.value}")
            return True

        except ImportError:
            logger.debug("[SelfplayOrchestrator] stage_events not available")
            return False
        except Exception as e:
            logger.error(f"[SelfplayOrchestrator] Failed to emit stage event: {e}")
            return False

    async def _on_selfplay_stage_complete(self, result) -> None:
        """Handle SELFPLAY_COMPLETE stage event."""
        logger.debug(
            f"[SelfplayOrchestrator] Stage event: SELFPLAY_COMPLETE "
            f"success={result.success}, games={result.games_generated}"
        )

    async def _on_gpu_selfplay_stage_complete(self, result) -> None:
        """Handle GPU_SELFPLAY_COMPLETE stage event."""
        logger.debug(
            f"[SelfplayOrchestrator] Stage event: GPU_SELFPLAY_COMPLETE "
            f"success={result.success}, games={result.games_generated}"
        )

    async def _on_canonical_selfplay_stage_complete(self, result) -> None:
        """Handle CANONICAL_SELFPLAY_COMPLETE stage event."""
        logger.debug(
            f"[SelfplayOrchestrator] Stage event: CANONICAL_SELFPLAY_COMPLETE "
            f"success={result.success}, games={result.games_generated}"
        )

    def register_task(
        self,
        task_id: str,
        selfplay_type: SelfplayType,
        node_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        games_requested: int = 0,
        iteration: int = 0,
    ) -> SelfplayTaskInfo:
        """Manually register a selfplay task.

        Use this when spawning selfplay outside the task coordinator.

        Returns:
            The created SelfplayTaskInfo
        """
        task = SelfplayTaskInfo(
            task_id=task_id,
            selfplay_type=selfplay_type,
            node_id=node_id,
            board_type=board_type,
            num_players=num_players,
            games_requested=games_requested,
            iteration=iteration,
        )

        self._active_tasks[task_id] = task
        logger.debug(f"[SelfplayOrchestrator] Registered task {task_id}")
        return task

    async def complete_task(
        self,
        task_id: str,
        success: bool = True,
        games_generated: int = 0,
        error: str = "",
    ) -> Optional[SelfplayTaskInfo]:
        """Manually mark a selfplay task as complete.

        Use this when selfplay completes outside the task coordinator.

        Returns:
            The completed SelfplayTaskInfo, or None if not found
        """
        if task_id not in self._active_tasks:
            logger.warning(f"[SelfplayOrchestrator] Unknown task {task_id}")
            return None

        task = self._active_tasks.pop(task_id)
        task.success = success
        task.end_time = time.time()
        task.games_generated = games_generated
        task.error = error

        self._record_completion(task, {})

        # Emit stage completion event
        await self._emit_selfplay_complete(task)

        return task

    def on_completion(self, callback: Callable[[SelfplayTaskInfo], None]) -> None:
        """Register a callback for selfplay completions.

        Args:
            callback: Function to call when selfplay completes
        """
        self._completion_callbacks.append(callback)

    def get_active_tasks(self) -> List[SelfplayTaskInfo]:
        """Get all active selfplay tasks."""
        return list(self._active_tasks.values())

    def get_task(self, task_id: str) -> Optional[SelfplayTaskInfo]:
        """Get a specific task by ID."""
        return self._active_tasks.get(task_id)

    def get_history(self, limit: int = 50) -> List[SelfplayTaskInfo]:
        """Get recent task completion history."""
        return self._completed_history[-limit:]

    def get_stats(self) -> SelfplayStats:
        """Get aggregate selfplay statistics."""
        # Count by node
        by_node: Dict[str, int] = {}
        for task in self._active_tasks.values():
            by_node[task.node_id] = by_node.get(task.node_id, 0) + 1

        # Count by type
        by_type: Dict[str, int] = {}
        for task in self._active_tasks.values():
            type_key = task.selfplay_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        # Calculate averages
        completed = [t for t in self._completed_history if t.success]
        avg_games = (
            sum(t.games_generated for t in completed) / len(completed)
            if completed
            else 0.0
        )
        avg_gps = (
            sum(t.games_per_second for t in completed) / len(completed)
            if completed
            else 0.0
        )

        failed_count = sum(1 for t in self._completed_history if not t.success)

        return SelfplayStats(
            active_tasks=len(self._active_tasks),
            completed_tasks=len(completed),
            failed_tasks=failed_count,
            total_games_generated=self._total_games_generated,
            total_duration_seconds=self._total_duration_seconds,
            average_games_per_task=avg_games,
            average_games_per_second=avg_gps,
            by_node=by_node,
            by_type=by_type,
            last_activity_time=time.time(),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status for monitoring."""
        stats = self.get_stats()

        return {
            "active_tasks": stats.active_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "total_games_generated": stats.total_games_generated,
            "average_games_per_task": round(stats.average_games_per_task, 1),
            "average_games_per_second": round(stats.average_games_per_second, 2),
            "by_node": stats.by_node,
            "by_type": stats.by_type,
            "subscribed": self._subscribed,
            "history_size": len(self._completed_history),
        }

    def clear_history(self) -> int:
        """Clear task history. Returns number of entries cleared."""
        count = len(self._completed_history)
        self._completed_history.clear()
        return count


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_selfplay_orchestrator: Optional[SelfplayOrchestrator] = None


def get_selfplay_orchestrator() -> SelfplayOrchestrator:
    """Get the global SelfplayOrchestrator singleton."""
    global _selfplay_orchestrator
    if _selfplay_orchestrator is None:
        _selfplay_orchestrator = SelfplayOrchestrator()
    return _selfplay_orchestrator


def wire_selfplay_events() -> SelfplayOrchestrator:
    """Wire selfplay events to the orchestrator.

    This subscribes the orchestrator to task lifecycle events
    and stage completion events.

    Returns:
        The wired SelfplayOrchestrator instance
    """
    orchestrator = get_selfplay_orchestrator()
    orchestrator.subscribe_to_events()
    orchestrator.subscribe_to_stage_events()
    return orchestrator


async def emit_selfplay_completion(
    task_id: str,
    board_type: str,
    num_players: int,
    games_generated: int,
    success: bool = True,
    node_id: str = "",
    selfplay_type: str = "canonical",
    iteration: int = 0,
    error: str = "",
) -> bool:
    """Emit a selfplay completion event.

    Call this when selfplay finishes to notify downstream pipeline stages.

    Args:
        task_id: Task identifier
        board_type: Board type (e.g., "square8")
        num_players: Number of players
        games_generated: Number of games generated
        success: Whether selfplay succeeded
        node_id: Node that ran selfplay
        selfplay_type: Type of selfplay (canonical, gpu_selfplay, etc.)
        iteration: Pipeline iteration number
        error: Error message if failed

    Returns:
        True if event was emitted successfully
    """
    orchestrator = get_selfplay_orchestrator()

    # Check if task is already tracked
    task = orchestrator.get_task(task_id)

    if task is None:
        # Create and register the task
        type_enum = SelfplayType.CANONICAL
        if selfplay_type == "gpu_selfplay":
            type_enum = SelfplayType.GPU_ACCELERATED
        elif selfplay_type == "hybrid_selfplay":
            type_enum = SelfplayType.HYBRID

        task = orchestrator.register_task(
            task_id=task_id,
            selfplay_type=type_enum,
            node_id=node_id,
            board_type=board_type,
            num_players=num_players,
            games_requested=games_generated,
            iteration=iteration,
        )

    # Complete the task (this emits the stage event)
    result = await orchestrator.complete_task(
        task_id=task_id,
        success=success,
        games_generated=games_generated,
        error=error,
    )

    return result is not None


def get_selfplay_stats() -> SelfplayStats:
    """Convenience function to get selfplay statistics."""
    return get_selfplay_orchestrator().get_stats()


__all__ = [
    "SelfplayOrchestrator",
    "SelfplayType",
    "SelfplayTaskInfo",
    "SelfplayStats",
    "get_selfplay_orchestrator",
    "wire_selfplay_events",
    "emit_selfplay_completion",
    "get_selfplay_stats",
]

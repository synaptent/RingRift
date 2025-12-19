#!/usr/bin/env python3
"""Async Training Bridge for Coordination Integration.

Bridges the synchronous TrainingCoordinator with async event-driven pipeline,
enabling seamless integration between local training coordination and the
async P2P/event systems.

Usage:
    from app.coordination.async_training_bridge import (
        AsyncTrainingBridge,
        get_training_bridge,
        async_request_training,
        async_complete_training,
    )

    bridge = get_training_bridge()

    # Request training slot (async)
    job_id = await bridge.request_training_slot("square8", 2)

    # Training progress automatically emits events
    await bridge.update_progress(job_id, epochs=10, loss=0.05)

    # Complete training (emits TRAINING_COMPLETE event)
    await bridge.complete_training(job_id)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from app.coordination.training_coordinator import (
    TrainingCoordinator,
    TrainingJob,
    get_training_coordinator,
)
from app.coordination.stage_events import (
    StageEvent,
    StageCompletionResult,
    get_event_bus,
)

# Use centralized executor pool (December 2025)
from app.coordination.async_bridge_manager import (
    get_bridge_manager,
    get_shared_executor,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgressEvent:
    """Event emitted when training progress updates."""

    job_id: str
    board_type: str
    num_players: int
    epochs_completed: int
    best_val_loss: float
    current_elo: float


class AsyncTrainingBridge:
    """Async bridge between TrainingCoordinator and event-driven pipeline.

    Wraps synchronous TrainingCoordinator methods with async interfaces
    and automatically emits StageEvent events on state changes.

    .. note:: Coordination Integration (2025-12)
        This bridge connects:
        - TrainingCoordinator (sync, SQLite-backed cluster coordination)
        - StageEventBus (async event-driven pipeline orchestration)
        - P2PBackend (async REST API for remote training)

    Example:
        bridge = get_training_bridge()

        # Request training with automatic event emission
        job_id = await bridge.request_training_slot("square8", 2)

        # Progress updates emit events for monitoring
        await bridge.update_progress(job_id, epochs=50, loss=0.02)

        # Completion emits TRAINING_COMPLETE event
        await bridge.complete_training(job_id, final_elo=1650.0)
    """

    def __init__(
        self,
        coordinator: Optional[TrainingCoordinator] = None,
        emit_events: bool = True,
    ):
        """Initialize the async training bridge.

        Args:
            coordinator: TrainingCoordinator instance (uses global if None)
            emit_events: Whether to emit StageEvent events on state changes
        """
        self._coordinator = coordinator or get_training_coordinator()
        self._emit_events = emit_events
        self._progress_callbacks: list[Callable[[TrainingProgressEvent], None]] = []

    async def _run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Run synchronous function in shared bridge pool."""
        return await get_bridge_manager().run_sync(func, *args, **kwargs)

    async def can_start_training(self, board_type: str, num_players: int) -> bool:
        """Check if training can be started (async wrapper).

        Args:
            board_type: Board type (e.g., "square8")
            num_players: Number of players

        Returns:
            True if training slot is available
        """
        return await self._run_sync(
            self._coordinator.can_start_training,
            board_type,
            num_players
        )

    async def request_training_slot(
        self,
        board_type: str,
        num_players: int,
        model_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Request a training slot (async wrapper).

        Args:
            board_type: Board type
            num_players: Number of players
            model_version: Model version string
            metadata: Additional metadata

        Returns:
            job_id if slot acquired, None otherwise
        """
        job_id = await self._run_sync(
            self._coordinator.start_training,
            board_type,
            num_players,
            model_version,
            metadata
        )

        if job_id:
            logger.info(f"Acquired training slot: {job_id}")

        return job_id

    async def update_progress(
        self,
        job_id: str,
        epochs_completed: int = 0,
        best_val_loss: float = float("inf"),
        current_elo: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update training progress (async wrapper).

        Args:
            job_id: The job ID
            epochs_completed: Number of epochs completed
            best_val_loss: Best validation loss so far
            current_elo: Current Elo rating
            metadata: Additional metadata

        Returns:
            True if update successful
        """
        success = await self._run_sync(
            self._coordinator.update_progress,
            job_id,
            epochs_completed,
            best_val_loss,
            current_elo,
            metadata
        )

        if success and self._progress_callbacks:
            # Get job info for event
            job = await self.get_job_by_id(job_id)
            if job:
                event = TrainingProgressEvent(
                    job_id=job_id,
                    board_type=job.board_type,
                    num_players=job.num_players,
                    epochs_completed=epochs_completed,
                    best_val_loss=best_val_loss,
                    current_elo=current_elo,
                )
                for callback in self._progress_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

        return success

    async def complete_training(
        self,
        job_id: str,
        status: str = "completed",
        final_val_loss: Optional[float] = None,
        final_elo: Optional[float] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """Complete training and emit event (async wrapper).

        Args:
            job_id: The job ID
            status: Final status (completed, failed)
            final_val_loss: Final validation loss
            final_elo: Final Elo rating
            model_path: Path to trained model

        Returns:
            True if completed successfully
        """
        # Get job info before completion
        job = await self.get_job_by_id(job_id)
        board_type = job.board_type if job else "unknown"
        num_players = job.num_players if job else 0

        success = await self._run_sync(
            self._coordinator.complete_training,
            job_id,
            status,
            final_val_loss,
            final_elo
        )

        if success and self._emit_events:
            # Emit TRAINING_COMPLETE event
            result = StageCompletionResult(
                event=StageEvent.TRAINING_COMPLETE,
                success=(status == "completed"),
                iteration=0,
                timestamp=datetime.now().isoformat(),
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
                val_loss=final_val_loss,
                elo_delta=final_elo - 1500.0 if final_elo else None,
                metadata={
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "final_val_loss": final_val_loss,
                    "final_elo": final_elo,
                    "status": status,
                },
            )

            event_bus = get_event_bus()
            await event_bus.emit(result)
            logger.info(f"Emitted TRAINING_COMPLETE for {job_id}")

        return success

    async def get_job_by_id(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID."""
        # Parse board_type and num_players from job_id
        # Format: {board_type}_{num_players}p_{timestamp}_{pid}
        try:
            parts = job_id.split("_")
            if len(parts) >= 2:
                board_type = parts[0]
                num_players = int(parts[1].rstrip("p"))
                return await self._run_sync(
                    self._coordinator.get_job,
                    board_type,
                    num_players
                )
        except Exception as e:
            logger.warning(f"Could not parse job_id {job_id}: {e}")
        return None

    async def get_active_jobs(self) -> list[TrainingJob]:
        """Get all active training jobs."""
        return await self._run_sync(self._coordinator.get_active_jobs)

    async def get_training_status(self) -> Dict[str, Any]:
        """Get cluster-wide training status."""
        return await self._run_sync(self._coordinator.get_status)

    def on_progress(self, callback: Callable[[TrainingProgressEvent], None]) -> None:
        """Register a callback for progress updates.

        Args:
            callback: Function to call on progress updates
        """
        self._progress_callbacks.append(callback)

    def off_progress(self, callback: Callable[[TrainingProgressEvent], None]) -> None:
        """Unregister a progress callback.

        Args:
            callback: Function to remove
        """
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)


# Global singleton
_bridge: Optional[AsyncTrainingBridge] = None


def get_training_bridge(
    coordinator: Optional[TrainingCoordinator] = None,
    emit_events: bool = True,
) -> AsyncTrainingBridge:
    """Get the global async training bridge.

    Args:
        coordinator: TrainingCoordinator instance (uses global if None)
        emit_events: Whether to emit events on state changes

    Returns:
        AsyncTrainingBridge singleton
    """
    global _bridge
    if _bridge is None:
        _bridge = AsyncTrainingBridge(coordinator, emit_events)
    return _bridge


def reset_training_bridge() -> None:
    """Reset the global training bridge (for testing)."""
    global _bridge
    _bridge = None


# Convenience async functions


async def async_can_train(board_type: str, num_players: int) -> bool:
    """Check if training can start (async)."""
    return await get_training_bridge().can_start_training(board_type, num_players)


async def async_request_training(
    board_type: str,
    num_players: int,
    model_version: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Request a training slot (async)."""
    return await get_training_bridge().request_training_slot(
        board_type, num_players, model_version, metadata
    )


async def async_update_progress(
    job_id: str,
    epochs_completed: int = 0,
    best_val_loss: float = float("inf"),
    current_elo: float = 0.0,
) -> bool:
    """Update training progress (async)."""
    return await get_training_bridge().update_progress(
        job_id, epochs_completed, best_val_loss, current_elo
    )


async def async_complete_training(
    job_id: str,
    status: str = "completed",
    final_val_loss: Optional[float] = None,
    final_elo: Optional[float] = None,
    model_path: Optional[str] = None,
) -> bool:
    """Complete training and emit event (async)."""
    return await get_training_bridge().complete_training(
        job_id, status, final_val_loss, final_elo, model_path
    )


async def async_get_training_status() -> Dict[str, Any]:
    """Get cluster-wide training status (async)."""
    return await get_training_bridge().get_training_status()


__all__ = [
    # Main class
    "AsyncTrainingBridge",
    "TrainingProgressEvent",
    # Global access
    "get_training_bridge",
    "reset_training_bridge",
    # Convenience functions
    "async_can_train",
    "async_request_training",
    "async_update_progress",
    "async_complete_training",
    "async_get_training_status",
]

"""Centralized Event Emitters for RingRift AI.

This module provides typed emit functions for all event types, eliminating
the need for each module to re-implement event emission logic.

Usage:
    from app.coordination.event_emitters import (
        emit_training_started,
        emit_training_complete,
        emit_selfplay_complete,
        emit_sync_complete,
        emit_quality_updated,
    )

    # Emit training complete event
    await emit_training_complete(
        job_id="square8_2p_123",
        board_type="square8",
        num_players=2,
        final_loss=0.05,
        final_elo=1650.0,
    )

Benefits:
- Single source of truth for event emission
- Type-safe event payloads
- Automatic fallback when event bus unavailable
- Consistent logging and error handling
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Event Bus Imports (with fallback)
# =============================================================================

try:
    from app.coordination.stage_events import (
        StageEvent,
        StageCompletionResult,
        get_event_bus as get_stage_bus,
    )
    HAS_STAGE_EVENTS = True
except ImportError:
    HAS_STAGE_EVENTS = False
    StageEvent = None
    StageCompletionResult = None

    def get_stage_bus():
        return None

try:
    from app.distributed.data_events import (
        DataEvent,
        DataEventType,
        get_event_bus as get_data_bus,
    )
    HAS_DATA_EVENTS = True
except ImportError:
    HAS_DATA_EVENTS = False
    DataEvent = None
    DataEventType = None

    def get_data_bus():
        return None


# =============================================================================
# Helper Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


async def _emit_stage_event(
    event: "StageEvent",
    result: "StageCompletionResult",
) -> bool:
    """Emit a StageEvent with proper error handling.

    Args:
        event: The StageEvent type
        result: The completion result

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        logger.debug(f"Stage events not available, skipping {event}")
        return False

    try:
        bus = get_stage_bus()
        if bus is None:
            return False

        await bus.emit(result)
        logger.debug(f"Emitted {result.event.value}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit {event}: {e}")
        return False


def _emit_stage_event_sync(
    event: "StageEvent",
    result: "StageCompletionResult",
) -> bool:
    """Emit a StageEvent synchronously (for use in sync contexts).

    Args:
        event: The StageEvent type
        result: The completion result

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    try:
        bus = get_stage_bus()
        if bus is None:
            return False

        # Try async emit first
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(bus.emit(result))
            return True
        except RuntimeError:
            # No event loop - try sync emit if available
            if hasattr(bus, 'emit_sync'):
                bus.emit_sync(result)
                return True
            return False

    except Exception as e:
        logger.debug(f"Failed to emit {event} sync: {e}")
        return False


# =============================================================================
# Training Events
# =============================================================================

async def emit_training_started(
    job_id: str,
    board_type: str,
    num_players: int,
    model_version: str = "",
    node_name: str = "",
    **metadata,
) -> bool:
    """Emit TRAINING_STARTED event.

    Args:
        job_id: Training job ID
        board_type: Board type (e.g., "square8")
        num_players: Number of players
        model_version: Model version string
        node_name: Node running the training
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result = StageCompletionResult(
        event=StageEvent.TRAINING_STARTED,
        success=True,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        metadata={
            "job_id": job_id,
            "model_version": model_version,
            "node_name": node_name,
            **metadata,
        },
    )

    return await _emit_stage_event(StageEvent.TRAINING_STARTED, result)


async def emit_training_complete(
    job_id: str,
    board_type: str,
    num_players: int,
    success: bool = True,
    final_loss: Optional[float] = None,
    final_elo: Optional[float] = None,
    model_path: Optional[str] = None,
    epochs_completed: int = 0,
    **metadata,
) -> bool:
    """Emit TRAINING_COMPLETE or TRAINING_FAILED event.

    Args:
        job_id: Training job ID
        board_type: Board type
        num_players: Number of players
        success: Whether training succeeded
        final_loss: Final validation loss
        final_elo: Final Elo rating
        model_path: Path to trained model
        epochs_completed: Number of epochs completed
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    event = StageEvent.TRAINING_COMPLETE if success else StageEvent.TRAINING_FAILED

    result = StageCompletionResult(
        event=event,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        val_loss=final_loss,
        elo_delta=final_elo - 1500.0 if final_elo else None,
        metadata={
            "job_id": job_id,
            "final_loss": final_loss,
            "final_elo": final_elo,
            "epochs_completed": epochs_completed,
            **metadata,
        },
    )

    return await _emit_stage_event(event, result)


def emit_training_complete_sync(
    job_id: str,
    board_type: str,
    num_players: int,
    success: bool = True,
    **kwargs,
) -> bool:
    """Synchronous version of emit_training_complete."""
    if not HAS_STAGE_EVENTS:
        return False

    event = StageEvent.TRAINING_COMPLETE if success else StageEvent.TRAINING_FAILED

    result = StageCompletionResult(
        event=event,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        model_path=kwargs.get("model_path"),
        val_loss=kwargs.get("final_loss"),
        elo_delta=kwargs.get("final_elo", 0) - 1500.0 if kwargs.get("final_elo") else None,
        metadata={"job_id": job_id, **kwargs},
    )

    return _emit_stage_event_sync(event, result)


# =============================================================================
# Selfplay Events
# =============================================================================

async def emit_selfplay_complete(
    task_id: str,
    board_type: str,
    num_players: int,
    games_generated: int,
    success: bool = True,
    node_id: str = "",
    duration_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit SELFPLAY_COMPLETE event.

    Args:
        task_id: Task ID
        board_type: Board type
        num_players: Number of players
        games_generated: Number of games generated
        success: Whether selfplay succeeded
        node_id: Node that ran selfplay
        duration_seconds: Duration of selfplay
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result = StageCompletionResult(
        event=StageEvent.SELFPLAY_COMPLETE,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        games_generated=games_generated,
        metadata={
            "task_id": task_id,
            "node_id": node_id,
            "duration_seconds": duration_seconds,
            **metadata,
        },
    )

    return await _emit_stage_event(StageEvent.SELFPLAY_COMPLETE, result)


# =============================================================================
# Evaluation Events
# =============================================================================

async def emit_evaluation_complete(
    model_id: str,
    board_type: str,
    num_players: int,
    success: bool = True,
    win_rate: Optional[float] = None,
    elo_delta: Optional[float] = None,
    games_played: int = 0,
    **metadata,
) -> bool:
    """Emit EVALUATION_COMPLETE event.

    Args:
        model_id: Model ID being evaluated
        board_type: Board type
        num_players: Number of players
        success: Whether evaluation succeeded
        win_rate: Win rate achieved
        elo_delta: Elo change
        games_played: Number of games played
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result = StageCompletionResult(
        event=StageEvent.EVALUATION_COMPLETE,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        win_rate=win_rate,
        elo_delta=elo_delta,
        metadata={
            "model_id": model_id,
            "games_played": games_played,
            **metadata,
        },
    )

    return await _emit_stage_event(StageEvent.EVALUATION_COMPLETE, result)


# =============================================================================
# Promotion Events
# =============================================================================

async def emit_promotion_complete(
    model_id: str,
    board_type: str,
    num_players: int,
    promotion_type: str = "production",
    elo_improvement: Optional[float] = None,
    model_path: Optional[str] = None,
    **metadata,
) -> bool:
    """Emit PROMOTION_COMPLETE event.

    Args:
        model_id: Model ID promoted
        board_type: Board type
        num_players: Number of players
        promotion_type: Type of promotion (production, champion)
        elo_improvement: Elo improvement over previous
        model_path: Path to promoted model
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result = StageCompletionResult(
        event=StageEvent.PROMOTION_COMPLETE,
        success=True,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        model_path=model_path,
        elo_delta=elo_improvement,
        metadata={
            "model_id": model_id,
            "promotion_type": promotion_type,
            **metadata,
        },
    )

    return await _emit_stage_event(StageEvent.PROMOTION_COMPLETE, result)


# =============================================================================
# Sync Events
# =============================================================================

async def emit_sync_complete(
    sync_type: str,
    items_synced: int,
    success: bool = True,
    duration_seconds: float = 0.0,
    source: str = "",
    **metadata,
) -> bool:
    """Emit SYNC_COMPLETE event.

    Args:
        sync_type: Type of sync (data, model, elo, registry)
        items_synced: Number of items synced
        success: Whether sync succeeded
        duration_seconds: Duration of sync
        source: Source of sync
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result = StageCompletionResult(
        event=StageEvent.SYNC_COMPLETE,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        games_generated=items_synced,
        metadata={
            "sync_type": sync_type,
            "items_synced": items_synced,
            "duration_seconds": duration_seconds,
            "source": source,
            **metadata,
        },
    )

    return await _emit_stage_event(StageEvent.SYNC_COMPLETE, result)


# =============================================================================
# Quality Events
# =============================================================================

async def emit_quality_updated(
    board_type: str,
    num_players: int,
    avg_quality: float,
    total_games: int,
    high_quality_games: int,
    **metadata,
) -> bool:
    """Emit quality score updated event via DataEventBus.

    Args:
        board_type: Board type
        num_players: Number of players
        avg_quality: Average quality score
        total_games: Total games in dataset
        high_quality_games: Number of high-quality games
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_DATA_EVENTS:
        return False

    try:
        bus = get_data_bus()
        if bus is None:
            return False

        event = DataEvent(
            event_type=DataEventType.QUALITY_SCORE_UPDATED,
            payload={
                "board_type": board_type,
                "num_players": num_players,
                "avg_quality": avg_quality,
                "total_games": total_games,
                "high_quality_games": high_quality_games,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug("Emitted quality_score_updated event")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit quality event: {e}")
        return False


# =============================================================================
# Task Events (Generic)
# =============================================================================

async def emit_task_complete(
    task_id: str,
    task_type: str,
    success: bool = True,
    node_id: str = "",
    duration_seconds: float = 0.0,
    result_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """Emit task completion event.

    Maps task_type to appropriate StageEvent:
    - selfplay → SELFPLAY_COMPLETE
    - training → TRAINING_COMPLETE/FAILED
    - evaluation → EVALUATION_COMPLETE
    - sync → SYNC_COMPLETE

    Args:
        task_id: Task ID
        task_type: Type of task (selfplay, training, evaluation, sync)
        success: Whether task succeeded
        node_id: Node that ran the task
        duration_seconds: Duration of task
        result_data: Task-specific result data

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result_data = result_data or {}

    # Map task type to StageEvent
    event_map = {
        "selfplay": StageEvent.SELFPLAY_COMPLETE,
        "gpu_selfplay": StageEvent.GPU_SELFPLAY_COMPLETE,
        "training": StageEvent.TRAINING_COMPLETE if success else StageEvent.TRAINING_FAILED,
        "evaluation": StageEvent.EVALUATION_COMPLETE,
        "sync": StageEvent.SYNC_COMPLETE,
        "data_sync": StageEvent.SYNC_COMPLETE,
        "npz_export": StageEvent.NPZ_EXPORT_COMPLETE,
    }

    event = event_map.get(task_type.lower())
    if not event:
        logger.debug(f"No StageEvent mapping for task type: {task_type}")
        return False

    result = StageCompletionResult(
        event=event,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        games_generated=result_data.get("games_generated", 0),
        model_path=result_data.get("model_path"),
        elo_delta=result_data.get("elo_delta"),
        win_rate=result_data.get("win_rate"),
        metadata={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "duration_seconds": duration_seconds,
            **result_data,
        },
    )

    return await _emit_stage_event(event, result)


__all__ = [
    # Training events
    "emit_training_started",
    "emit_training_complete",
    "emit_training_complete_sync",
    # Selfplay events
    "emit_selfplay_complete",
    # Evaluation events
    "emit_evaluation_complete",
    # Promotion events
    "emit_promotion_complete",
    # Sync events
    "emit_sync_complete",
    # Quality events
    "emit_quality_updated",
    # Generic task events
    "emit_task_complete",
]

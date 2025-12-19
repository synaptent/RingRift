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

# Unified event router (preferred for cross-system routing)
try:
    from app.coordination.event_router import (
        get_router as get_event_router,
        EventSource,
    )
    HAS_EVENT_ROUTER = True
except ImportError:
    HAS_EVENT_ROUTER = False

    def get_event_router():
        return None


# Configuration: whether to use unified router or direct bus access
# Set to True to route all events through the unified router
USE_UNIFIED_ROUTER = True


# =============================================================================
# Helper Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


async def _emit_via_router(
    event_type: str,
    payload: Dict[str, Any],
    source: str = "event_emitters",
) -> bool:
    """Emit an event via the unified event router.

    Args:
        event_type: Event type string (or DataEventType value)
        payload: Event payload data
        source: Source component name

    Returns:
        True if emitted successfully
    """
    if not HAS_EVENT_ROUTER or not USE_UNIFIED_ROUTER:
        return False

    try:
        router = get_event_router()
        if router is None:
            return False

        await router.publish(
            event_type=event_type,
            payload=payload,
            source=source,
        )
        logger.debug(f"Emitted {event_type} via unified router")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit {event_type} via router: {e}")
        return False


def _emit_via_router_sync(
    event_type: str,
    payload: Dict[str, Any],
    source: str = "event_emitters",
) -> bool:
    """Emit an event synchronously via the unified router.

    Args:
        event_type: Event type string
        payload: Event payload data
        source: Source component name

    Returns:
        True if emitted successfully
    """
    if not HAS_EVENT_ROUTER or not USE_UNIFIED_ROUTER:
        return False

    try:
        router = get_event_router()
        if router is None:
            return False

        router.publish_sync(
            event_type=event_type,
            payload=payload,
            source=source,
        )
        logger.debug(f"Emitted {event_type} sync via unified router")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit {event_type} sync via router: {e}")
        return False


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
    # Try unified router first if enabled
    if USE_UNIFIED_ROUTER and HAS_EVENT_ROUTER:
        payload = {
            "event": event.value if hasattr(event, 'value') else str(event),
            "success": result.success if hasattr(result, 'success') else True,
            "config": result.config if hasattr(result, 'config') else "",
            "metrics": result.metrics if hasattr(result, 'metrics') else {},
        }
        if await _emit_via_router(event.value if hasattr(event, 'value') else str(event), payload, "stage_event"):
            return True

    # Fallback to direct stage bus
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
    # Try unified router first if enabled
    if USE_UNIFIED_ROUTER and HAS_EVENT_ROUTER:
        payload = {
            "event": event.value if hasattr(event, 'value') else str(event),
            "success": result.success if hasattr(result, 'success') else True,
            "config": result.config if hasattr(result, 'config') else "",
            "metrics": result.metrics if hasattr(result, 'metrics') else {},
        }
        if _emit_via_router_sync(event.value if hasattr(event, 'value') else str(event), payload, "stage_event"):
            return True

    # Fallback to direct stage bus
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
    selfplay_type: str = "standard",  # "standard", "gpu_accelerated", "canonical"
    iteration: int = 0,
    error: Optional[str] = None,
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
        selfplay_type: Type of selfplay ("standard", "gpu_accelerated", "canonical")
        iteration: Current iteration
        error: Error message if failed
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    # Map selfplay_type to appropriate event
    if selfplay_type == "gpu_accelerated":
        event = StageEvent.GPU_SELFPLAY_COMPLETE
    elif selfplay_type == "canonical":
        event = StageEvent.CANONICAL_SELFPLAY_COMPLETE
    else:
        event = StageEvent.SELFPLAY_COMPLETE

    result = StageCompletionResult(
        event=event,
        success=success,
        iteration=iteration,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        games_generated=games_generated,
        error=error if not success else None,
        metadata={
            "task_id": task_id,
            "node_id": node_id,
            "duration_seconds": duration_seconds,
            "selfplay_type": selfplay_type,
            **metadata,
        },
    )

    return await _emit_stage_event(event, result)


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
    iteration: int = 0,
    components: Optional[list] = None,
    errors: Optional[list] = None,
    **metadata,
) -> bool:
    """Emit SYNC_COMPLETE event.

    Args:
        sync_type: Type of sync (data, model, elo, registry)
        items_synced: Number of items synced
        success: Whether sync succeeded
        duration_seconds: Duration of sync
        source: Source of sync
        iteration: Sync iteration counter
        components: List of synced component names
        errors: List of error messages if any
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    result = StageCompletionResult(
        event=StageEvent.SYNC_COMPLETE,
        success=success,
        iteration=iteration,
        timestamp=_get_timestamp(),
        games_generated=items_synced,
        metadata={
            "sync_type": sync_type,
            "items_synced": items_synced,
            "duration_seconds": duration_seconds,
            "source": source,
            "components": components or [],
            "errors": errors or [],
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


# =============================================================================
# Optimization Events (December 2025)
# =============================================================================

async def emit_optimization_triggered(
    optimization_type: str,  # "cmaes" or "nas"
    run_id: str,
    reason: str,
    parameters_searched: int = 0,
    search_space: Optional[Dict[str, Any]] = None,
    generations: int = 0,
    population_size: int = 0,
    **metadata,
) -> bool:
    """Emit CMAES_TRIGGERED or NAS_TRIGGERED event.

    Args:
        optimization_type: Type of optimization ("cmaes" or "nas")
        run_id: Unique run identifier
        reason: Reason for triggering optimization
        parameters_searched: Number of parameters being optimized
        search_space: Search space configuration
        generations: Number of generations (for evolutionary methods)
        population_size: Population size
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

        event_type = (
            DataEventType.CMAES_TRIGGERED
            if optimization_type.lower() == "cmaes"
            else DataEventType.NAS_TRIGGERED
        )

        event = DataEvent(
            event_type=event_type,
            payload={
                "run_id": run_id,
                "reason": reason,
                "parameters_searched": parameters_searched,
                "search_space": search_space or {},
                "generations": generations,
                "population_size": population_size,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted {event_type.value} event")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit optimization event: {e}")
        return False


# =============================================================================
# Metrics Events (December 2025)
# =============================================================================

async def emit_plateau_detected(
    metric_name: str,
    current_value: float,
    best_value: float,
    epochs_since_improvement: int,
    plateau_type: str = "metric",  # "loss", "elo", "metric"
    **metadata,
) -> bool:
    """Emit PLATEAU_DETECTED event.

    Args:
        metric_name: Name of the metric that plateaued
        current_value: Current metric value
        best_value: Best value seen
        epochs_since_improvement: Epochs since last improvement
        plateau_type: Type of plateau
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
            event_type=DataEventType.PLATEAU_DETECTED,
            payload={
                "metric_name": metric_name,
                "plateau_type": plateau_type,
                "current_value": current_value,
                "best_value": best_value,
                "epochs_since_improvement": epochs_since_improvement,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug("Emitted plateau_detected event")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit plateau event: {e}")
        return False


async def emit_regression_detected(
    metric_name: str,
    current_value: float,
    previous_value: float,
    severity: str = "minor",  # "minor", "moderate", "severe", "critical"
    **metadata,
) -> bool:
    """Emit REGRESSION_DETECTED event.

    Args:
        metric_name: Name of the metric that regressed
        current_value: Current metric value
        previous_value: Previous/best value
        severity: Severity level of regression
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
            event_type=DataEventType.REGRESSION_DETECTED,
            payload={
                "metric_name": metric_name,
                "current_value": current_value,
                "previous_value": previous_value,
                "severity": severity,
                "regression_amount": previous_value - current_value,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted regression_detected event (severity={severity})")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit regression event: {e}")
        return False


# =============================================================================
# Backpressure Events (December 2025)
# =============================================================================

async def emit_backpressure_activated(
    node_id: str,
    level: str,  # "low", "medium", "high", "critical"
    reason: str,
    resource_type: str = "",
    utilization: float = 0.0,
    **metadata,
) -> bool:
    """Emit BACKPRESSURE_ACTIVATED event.

    Args:
        node_id: Node experiencing backpressure
        level: Backpressure level
        reason: Reason for activation
        resource_type: Resource causing backpressure (gpu, memory, disk)
        utilization: Current utilization percentage
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
            event_type=DataEventType.BACKPRESSURE_ACTIVATED,
            payload={
                "node_id": node_id,
                "level": level,
                "reason": reason,
                "resource_type": resource_type,
                "utilization": utilization,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted backpressure_activated event for {node_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit backpressure event: {e}")
        return False


async def emit_backpressure_released(
    node_id: str,
    previous_level: str = "",
    duration_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit BACKPRESSURE_RELEASED event.

    Args:
        node_id: Node where backpressure was released
        previous_level: Previous backpressure level
        duration_seconds: How long backpressure was active
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
            event_type=DataEventType.BACKPRESSURE_RELEASED,
            payload={
                "node_id": node_id,
                "previous_level": previous_level,
                "duration_seconds": duration_seconds,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted backpressure_released event for {node_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit backpressure release event: {e}")
        return False


# =============================================================================
# Cache Events (December 2025)
# =============================================================================

async def emit_cache_invalidated(
    invalidation_type: str,  # "model" or "node"
    target_id: str,
    count: int,
    affected_nodes: Optional[list] = None,
    affected_models: Optional[list] = None,
    **metadata,
) -> bool:
    """Emit CACHE_INVALIDATED event.

    Args:
        invalidation_type: Type of invalidation ("model" or "node")
        target_id: Model ID or node ID invalidated
        count: Number of cache entries invalidated
        affected_nodes: List of affected node IDs
        affected_models: List of affected model IDs
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
            event_type=DataEventType.CACHE_INVALIDATED,
            payload={
                "invalidation_type": invalidation_type,
                "target_id": target_id,
                "count": count,
                "affected_nodes": affected_nodes or [],
                "affected_models": affected_models or [],
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted cache_invalidated event: {invalidation_type}={target_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit cache invalidation event: {e}")
        return False


# =============================================================================
# Host/Node Events (December 2025)
# =============================================================================

async def emit_host_online(
    node_id: str,
    host_type: str = "",
    capabilities: Optional[Dict[str, Any]] = None,
    **metadata,
) -> bool:
    """Emit HOST_ONLINE event.

    Args:
        node_id: Node coming online
        host_type: Type of host (gh200, cpu, etc.)
        capabilities: Host capabilities dict
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
            event_type=DataEventType.HOST_ONLINE,
            payload={
                "node_id": node_id,
                "host_id": node_id,  # Alias for compatibility
                "host_type": host_type,
                "capabilities": capabilities or {},
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted host_online event for {node_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit host online event: {e}")
        return False


async def emit_host_offline(
    node_id: str,
    reason: str = "",
    **metadata,
) -> bool:
    """Emit HOST_OFFLINE event.

    Args:
        node_id: Node going offline
        reason: Reason for going offline
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
            event_type=DataEventType.HOST_OFFLINE,
            payload={
                "node_id": node_id,
                "host_id": node_id,
                "reason": reason,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted host_offline event for {node_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit host offline event: {e}")
        return False


async def emit_node_recovered(
    node_id: str,
    recovery_type: str = "automatic",
    offline_duration_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit NODE_RECOVERED event.

    Args:
        node_id: Node that recovered
        recovery_type: Type of recovery (automatic, manual)
        offline_duration_seconds: How long node was offline
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
            event_type=DataEventType.NODE_RECOVERED,
            payload={
                "node_id": node_id,
                "host_id": node_id,
                "recovery_type": recovery_type,
                "offline_duration_seconds": offline_duration_seconds,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.debug(f"Emitted node_recovered event for {node_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit node recovered event: {e}")
        return False


# =============================================================================
# Error Recovery & Resilience Events (December 2025)
# =============================================================================

async def emit_training_rollback_needed(
    model_id: str,
    reason: str,
    checkpoint_path: Optional[str] = None,
    severity: str = "moderate",
    **metadata,
) -> bool:
    """Emit TRAINING_ROLLBACK_NEEDED event.

    Args:
        model_id: Model that needs rollback
        reason: Why rollback is needed
        checkpoint_path: Path to checkpoint to rollback to
        severity: Severity level (minor, moderate, severe, critical)
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
            event_type=DataEventType.TRAINING_ROLLBACK_NEEDED,
            payload={
                "model_id": model_id,
                "reason": reason,
                "checkpoint_path": checkpoint_path,
                "severity": severity,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.warning(f"Emitted training_rollback_needed for {model_id}: {reason}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit rollback event: {e}")
        return False


async def emit_handler_failed(
    handler_name: str,
    event_type: str,
    error: str,
    coordinator: str = "",
    **metadata,
) -> bool:
    """Emit HANDLER_FAILED event when an event handler throws an exception.

    Args:
        handler_name: Name of the failed handler
        event_type: Event type being handled
        error: Error message
        coordinator: Coordinator that owns the handler
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
            event_type=DataEventType.HANDLER_FAILED,
            payload={
                "handler_name": handler_name,
                "event_type": event_type,
                "error": error,
                "coordinator": coordinator,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.error(f"Emitted handler_failed: {handler_name} on {event_type}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit handler_failed event: {e}")
        return False


async def emit_handler_timeout(
    handler_name: str,
    event_type: str,
    timeout_seconds: float,
    coordinator: str = "",
    **metadata,
) -> bool:
    """Emit HANDLER_TIMEOUT event when an event handler times out.

    Args:
        handler_name: Name of the timed out handler
        event_type: Event type being handled
        timeout_seconds: Timeout duration
        coordinator: Coordinator that owns the handler
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
            event_type=DataEventType.HANDLER_TIMEOUT,
            payload={
                "handler_name": handler_name,
                "event_type": event_type,
                "timeout_seconds": timeout_seconds,
                "coordinator": coordinator,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.warning(f"Emitted handler_timeout: {handler_name} after {timeout_seconds}s")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit handler_timeout event: {e}")
        return False


async def emit_coordinator_health_degraded(
    coordinator_name: str,
    reason: str,
    health_score: float = 0.0,
    issues: Optional[list] = None,
    **metadata,
) -> bool:
    """Emit COORDINATOR_HEALTH_DEGRADED event.

    Args:
        coordinator_name: Name of the degraded coordinator
        reason: Why health is degraded
        health_score: Health score 0.0-1.0
        issues: List of specific issues
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
            event_type=DataEventType.COORDINATOR_HEALTH_DEGRADED,
            payload={
                "coordinator_name": coordinator_name,
                "reason": reason,
                "health_score": health_score,
                "issues": issues or [],
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.warning(f"Emitted coordinator_health_degraded: {coordinator_name}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit health degraded event: {e}")
        return False


async def emit_coordinator_shutdown(
    coordinator_name: str,
    reason: str = "graceful",
    remaining_tasks: int = 0,
    state_snapshot: Optional[Dict[str, Any]] = None,
    **metadata,
) -> bool:
    """Emit COORDINATOR_SHUTDOWN event for graceful shutdown.

    Args:
        coordinator_name: Name of the shutting down coordinator
        reason: Shutdown reason (graceful, error, forced)
        remaining_tasks: Number of tasks still pending
        state_snapshot: Final state before shutdown
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
            event_type=DataEventType.COORDINATOR_SHUTDOWN,
            payload={
                "coordinator_name": coordinator_name,
                "reason": reason,
                "remaining_tasks": remaining_tasks,
                "state_snapshot": state_snapshot or {},
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.info(f"Emitted coordinator_shutdown: {coordinator_name}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit shutdown event: {e}")
        return False


async def emit_coordinator_heartbeat(
    coordinator_name: str,
    health_score: float = 1.0,
    active_handlers: int = 0,
    events_processed: int = 0,
    **metadata,
) -> bool:
    """Emit COORDINATOR_HEARTBEAT for liveness detection.

    Args:
        coordinator_name: Name of the coordinator
        health_score: Current health 0.0-1.0
        active_handlers: Number of handlers currently running
        events_processed: Total events processed since startup
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
            event_type=DataEventType.COORDINATOR_HEARTBEAT,
            payload={
                "coordinator_name": coordinator_name,
                "health_score": health_score,
                "active_handlers": active_handlers,
                "events_processed": events_processed,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        return True

    except Exception as e:
        logger.debug(f"Failed to emit heartbeat event: {e}")
        return False


async def emit_task_abandoned(
    task_id: str,
    task_type: str,
    node_id: str,
    reason: str,
    **metadata,
) -> bool:
    """Emit TASK_ABANDONED event for intentionally abandoned tasks.

    Args:
        task_id: Abandoned task ID
        task_type: Type of task
        node_id: Node where task was running
        reason: Why task was abandoned
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
            event_type=DataEventType.TASK_ABANDONED,
            payload={
                "task_id": task_id,
                "task_type": task_type,
                "node_id": node_id,
                "reason": reason,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.info(f"Emitted task_abandoned: {task_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit task_abandoned event: {e}")
        return False


async def emit_task_orphaned(
    task_id: str,
    task_type: str,
    node_id: str,
    last_heartbeat: float,
    reason: str,
    **metadata,
) -> bool:
    """Emit TASK_ORPHANED event for tasks that lost their parent worker.

    This event enables cross-coordinator cleanup coordination when tasks
    become orphaned due to worker failure or network issues.

    Args:
        task_id: Orphaned task ID
        task_type: Type of task (selfplay, training, etc.)
        node_id: Last known node where task was running
        last_heartbeat: Timestamp of last heartbeat
        reason: Why task is considered orphaned
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
            event_type=DataEventType.TASK_ORPHANED,
            payload={
                "task_id": task_id,
                "task_type": task_type,
                "node_id": node_id,
                "last_heartbeat": last_heartbeat,
                "reason": reason,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.info(f"Emitted task_orphaned: {task_id} ({reason})")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit task_orphaned event: {e}")
        return False


async def emit_model_corrupted(
    model_id: str,
    model_path: str,
    corruption_type: str,
    **metadata,
) -> bool:
    """Emit MODEL_CORRUPTED event when model file corruption is detected.

    Args:
        model_id: Corrupted model ID
        model_path: Path to corrupted model
        corruption_type: Type of corruption (checksum, format, missing)
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
            event_type=DataEventType.MODEL_CORRUPTED,
            payload={
                "model_id": model_id,
                "model_path": model_path,
                "corruption_type": corruption_type,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.error(f"Emitted model_corrupted: {model_id}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit model_corrupted event: {e}")
        return False


async def emit_training_rollback_completed(
    model_id: str,
    checkpoint_path: str,
    rollback_from: str,
    reason: str,
    **metadata,
) -> bool:
    """Emit TRAINING_ROLLBACK_COMPLETED event.

    Args:
        model_id: Model that was rolled back
        checkpoint_path: Path to checkpoint that was restored
        rollback_from: Previous model version that was replaced
        reason: Why rollback was performed
        **metadata: Additional event metadata

    Returns:
        True if event was emitted successfully
    """
    if not HAS_DATA_EVENTS:
        return False

    try:
        bus = get_data_bus()
        if bus is None:
            return False

        event = DataEvent(
            event_type=DataEventType.TRAINING_ROLLBACK_COMPLETED,
            payload={
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "rollback_from": rollback_from,
                "reason": reason,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.info(f"Emitted training_rollback_completed: {model_id} from {rollback_from}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit rollback_completed event: {e}")
        return False


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: dict,
    new_weights: dict,
    reason: str,
    trigger: str = "automatic",
    **metadata,
) -> bool:
    """Emit CURRICULUM_REBALANCED event.

    Args:
        config: Board configuration that was rebalanced
        old_weights: Previous curriculum weights
        new_weights: New curriculum weights
        reason: Why rebalancing occurred
        trigger: What triggered rebalancing (automatic, manual, elo_change)
        **metadata: Additional event metadata

    Returns:
        True if event was emitted successfully
    """
    if not HAS_DATA_EVENTS:
        return False

    try:
        bus = get_data_bus()
        if bus is None:
            return False

        event = DataEvent(
            event_type=DataEventType.CURRICULUM_REBALANCED,
            payload={
                "config": config,
                "old_weights": old_weights,
                "new_weights": new_weights,
                "reason": reason,
                "trigger": trigger,
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.info(f"Emitted curriculum_rebalanced for {config}: {reason}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit curriculum_rebalanced event: {e}")
        return False


async def emit_training_triggered(
    config: str,
    job_id: str,
    trigger_reason: str,
    game_count: int = 0,
    threshold: int = 0,
    priority: str = "normal",
    **metadata,
) -> bool:
    """Emit event when training is triggered (before it starts).

    This is distinct from TRAINING_STARTED which fires when training actually begins.
    TRAINING_TRIGGERED fires when the decision to train is made.

    Args:
        config: Board configuration
        job_id: Training job ID
        trigger_reason: Why training was triggered (threshold, manual, scheduled)
        game_count: Number of games available for training
        threshold: Threshold that triggered training
        priority: Job priority (low, normal, high)
        **metadata: Additional event metadata

    Returns:
        True if event was emitted successfully
    """
    if not HAS_DATA_EVENTS:
        return False

    try:
        bus = get_data_bus()
        if bus is None:
            return False

        # Use TRAINING_THRESHOLD_REACHED as the closest existing event type
        event = DataEvent(
            event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
            payload={
                "config": config,
                "job_id": job_id,
                "trigger_reason": trigger_reason,
                "games": game_count,
                "threshold": threshold,
                "priority": priority,
                "event_subtype": "training_triggered",
                "timestamp": _get_timestamp(),
                **metadata,
            },
            source="event_emitters",
        )

        await bus.publish(event)
        logger.info(f"Emitted training_triggered for {config}: {trigger_reason}")
        return True

    except Exception as e:
        logger.debug(f"Failed to emit training_triggered event: {e}")
        return False


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
    # Optimization events (December 2025)
    "emit_optimization_triggered",
    # Metrics events (December 2025)
    "emit_plateau_detected",
    "emit_regression_detected",
    # Backpressure events (December 2025)
    "emit_backpressure_activated",
    "emit_backpressure_released",
    # Cache events (December 2025)
    "emit_cache_invalidated",
    # Host/Node events (December 2025)
    "emit_host_online",
    "emit_host_offline",
    "emit_node_recovered",
    # Error Recovery & Resilience events (December 2025)
    "emit_training_rollback_needed",
    "emit_training_rollback_completed",
    "emit_handler_failed",
    "emit_handler_timeout",
    "emit_coordinator_health_degraded",
    "emit_coordinator_shutdown",
    "emit_coordinator_heartbeat",
    "emit_task_abandoned",
    "emit_task_orphaned",
    "emit_model_corrupted",
    # Curriculum events (December 2025)
    "emit_curriculum_rebalanced",
    "emit_training_triggered",
]

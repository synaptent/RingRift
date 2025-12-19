"""Event Bus Helper Module - Consolidated event handling utilities.

This module provides safe, reusable event bus functions to eliminate duplicate
try/except import patterns found across 17+ scripts in the codebase.

Instead of this pattern in every script:
    try:
        from app.distributed.data_events import (
            DataEventType, DataEvent, get_event_bus, emit_model_promoted, ...
        )
        HAS_EVENT_BUS = True
    except ImportError:
        HAS_EVENT_BUS = False

Use this:
    from app.distributed.event_helpers import (
        get_event_bus_safe, emit_event_safe, has_event_bus,
        emit_model_promoted_safe, emit_training_completed_safe, ...
    )

Usage:
    # Check if event bus is available
    if has_event_bus():
        bus = get_event_bus_safe()
        await bus.publish(event)

    # Or use safe wrappers that handle unavailability gracefully
    await emit_model_promoted_safe(model_id, config, elo, elo_gain, source)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Try to import the actual event bus components
_HAS_EVENT_BUS = False
_DataEventType = None
_DataEvent = None
_EventBus = None
_get_event_bus = None
_emit_model_promoted = None
_emit_training_completed = None
_emit_evaluation_completed = None
_emit_error = None
_emit_elo_updated = None

try:
    from app.distributed.data_events import (
        DataEventType,
        DataEvent,
        EventBus,
        get_event_bus,
        emit_model_promoted,
        emit_training_completed,
        emit_evaluation_completed,
        emit_error,
        emit_elo_updated,
    )
    _HAS_EVENT_BUS = True
    _DataEventType = DataEventType
    _DataEvent = DataEvent
    _EventBus = EventBus
    _get_event_bus = get_event_bus
    _emit_model_promoted = emit_model_promoted
    _emit_training_completed = emit_training_completed
    _emit_evaluation_completed = emit_evaluation_completed
    _emit_error = emit_error
    _emit_elo_updated = emit_elo_updated
except ImportError:
    pass


def has_event_bus() -> bool:
    """Check if the event bus module is available.

    Returns:
        True if app.distributed.data_events is importable, False otherwise.
    """
    return _HAS_EVENT_BUS


def get_event_bus_safe() -> Optional[Any]:
    """Get the event bus instance if available.

    Returns:
        EventBus instance or None if not available.
    """
    if not _HAS_EVENT_BUS or _get_event_bus is None:
        return None
    try:
        return _get_event_bus()
    except Exception as e:
        logger.debug(f"Failed to get event bus: {e}")
        return None


def get_event_types():
    """Get the DataEventType enum if available.

    Returns:
        DataEventType enum or None if not available.
    """
    return _DataEventType


def create_event(
    event_type: str,
    payload: Dict[str, Any],
    source: str = ""
) -> Optional[Any]:
    """Create a DataEvent if the event bus is available.

    Args:
        event_type: Event type name (will look up in DataEventType enum)
        payload: Event payload dictionary
        source: Source component name

    Returns:
        DataEvent instance or None if not available.
    """
    if not _HAS_EVENT_BUS or _DataEvent is None or _DataEventType is None:
        return None

    try:
        # Get event type from enum
        et = getattr(_DataEventType, event_type, None)
        if et is None:
            logger.warning(f"Unknown event type: {event_type}")
            return None
        return _DataEvent(event_type=et, payload=payload, source=source)
    except Exception as e:
        logger.debug(f"Failed to create event: {e}")
        return None


async def emit_event_safe(
    event_type: str,
    payload: Dict[str, Any],
    source: str = ""
) -> bool:
    """Safely emit an event, handling unavailable event bus gracefully.

    Args:
        event_type: Event type name (will look up in DataEventType enum)
        payload: Event payload dictionary
        source: Source component name

    Returns:
        True if event was emitted successfully, False otherwise.
    """
    bus = get_event_bus_safe()
    if bus is None:
        logger.debug(f"Event bus unavailable, dropping event: {event_type}")
        return False

    event = create_event(event_type, payload, source)
    if event is None:
        return False

    try:
        await bus.publish(event)
        return True
    except Exception as e:
        logger.warning(f"Failed to emit event {event_type}: {e}")
        return False


def subscribe_safe(
    event_type: str,
    handler: Callable,
    bus: Optional[Any] = None
) -> bool:
    """Safely subscribe to an event type.

    Args:
        event_type: Event type name
        handler: Async handler function
        bus: Optional event bus instance (will get default if None)

    Returns:
        True if subscription succeeded, False otherwise.
    """
    if bus is None:
        bus = get_event_bus_safe()
    if bus is None:
        logger.debug(f"Event bus unavailable, cannot subscribe to: {event_type}")
        return False

    if _DataEventType is None:
        return False

    try:
        et = getattr(_DataEventType, event_type, None)
        if et is None:
            logger.warning(f"Unknown event type for subscription: {event_type}")
            return False
        bus.subscribe(et, handler)
        return True
    except Exception as e:
        logger.warning(f"Failed to subscribe to {event_type}: {e}")
        return False


# =============================================================================
# Safe wrappers for common emit functions
# =============================================================================

async def emit_model_promoted_safe(
    model_id: str,
    config: str,
    elo: float,
    elo_gain: float = 0.0,
    source: str = ""
) -> bool:
    """Safely emit MODEL_PROMOTED event.

    Args:
        model_id: ID of promoted model
        config: Configuration key (e.g., "square8_2p")
        elo: Current Elo rating
        elo_gain: Elo gain from previous best
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    if not _HAS_EVENT_BUS or _emit_model_promoted is None:
        logger.debug(f"Event bus unavailable, skipping MODEL_PROMOTED: {model_id}")
        return False

    try:
        await _emit_model_promoted(model_id, config, elo, elo_gain, source=source)
        return True
    except Exception as e:
        logger.warning(f"Failed to emit MODEL_PROMOTED: {e}")
        return False


async def emit_training_completed_safe(
    config: str,
    model_path: str,
    loss: float,
    samples: int,
    duration: float = 0.0,
    source: str = ""
) -> bool:
    """Safely emit TRAINING_COMPLETED event.

    Args:
        config: Configuration key (e.g., "square8_2p")
        model_path: Path to trained model
        loss: Final training loss
        samples: Number of training samples used
        duration: Training duration in seconds
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    if not _HAS_EVENT_BUS or _emit_training_completed is None:
        logger.debug(f"Event bus unavailable, skipping TRAINING_COMPLETED: {config}")
        return False

    try:
        await _emit_training_completed(config, model_path, loss, samples, duration, source=source)
        return True
    except Exception as e:
        logger.warning(f"Failed to emit TRAINING_COMPLETED: {e}")
        return False


async def emit_evaluation_completed_safe(
    config: str,
    elo: float,
    games: int,
    win_rate: float,
    source: str = ""
) -> bool:
    """Safely emit EVALUATION_COMPLETED event.

    Args:
        config: Configuration key (e.g., "square8_2p")
        elo: Elo rating estimate
        games: Number of games played
        win_rate: Win rate (0.0-1.0)
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    if not _HAS_EVENT_BUS or _emit_evaluation_completed is None:
        logger.debug(f"Event bus unavailable, skipping EVALUATION_COMPLETED: {config}")
        return False

    try:
        await _emit_evaluation_completed(config, elo, games, win_rate, source=source)
        return True
    except Exception as e:
        logger.warning(f"Failed to emit EVALUATION_COMPLETED: {e}")
        return False


async def emit_error_safe(
    component: str,
    error: str,
    source: str = ""
) -> bool:
    """Safely emit ERROR event.

    Args:
        component: Component where error occurred
        error: Error message
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    if not _HAS_EVENT_BUS or _emit_error is None:
        logger.debug(f"Event bus unavailable, skipping ERROR: {component}")
        return False

    try:
        await _emit_error(component, error, source=source)
        return True
    except Exception as e:
        logger.warning(f"Failed to emit ERROR: {e}")
        return False


async def emit_elo_updated_safe(
    config: str,
    model_id: str,
    new_elo: float,
    old_elo: float,
    games_played: int,
    source: str = ""
) -> bool:
    """Safely emit ELO_UPDATED event.

    Args:
        config: Configuration key (e.g., "square8_2p")
        model_id: Model identifier
        new_elo: New Elo rating
        old_elo: Previous Elo rating
        games_played: Number of games played
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    if not _HAS_EVENT_BUS or _emit_elo_updated is None:
        logger.debug(f"Event bus unavailable, skipping ELO_UPDATED: {model_id}")
        return False

    try:
        await _emit_elo_updated(config, model_id, new_elo, old_elo, games_played, source=source)
        return True
    except Exception as e:
        logger.warning(f"Failed to emit ELO_UPDATED: {e}")
        return False


async def emit_new_games_safe(
    host: str,
    new_games: int,
    config: str = "",
    source: str = ""
) -> bool:
    """Safely emit NEW_GAMES_AVAILABLE event.

    Args:
        host: Host where games originated
        new_games: Number of new games
        config: Configuration key (optional)
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "NEW_GAMES_AVAILABLE",
        {"host": host, "new_games": new_games, "config": config},
        source
    )


async def emit_training_started_safe(
    config: str,
    samples: int,
    source: str = ""
) -> bool:
    """Safely emit TRAINING_STARTED event.

    Args:
        config: Configuration key
        samples: Number of training samples
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "TRAINING_STARTED",
        {"config": config, "samples": samples},
        source
    )


async def emit_training_failed_safe(
    config: str,
    error: str,
    duration: float = 0.0,
    source: str = ""
) -> bool:
    """Safely emit TRAINING_FAILED event.

    Args:
        config: Configuration key
        error: Error message
        duration: Duration before failure
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "TRAINING_FAILED",
        {"config": config, "error": error, "duration": duration},
        source
    )


# =============================================================================
# Quality event helpers (December 2025)
# =============================================================================

async def emit_quality_score_updated_safe(
    game_id: str,
    old_score: float,
    new_score: float,
    config: str = "",
    source: str = ""
) -> bool:
    """Safely emit QUALITY_SCORE_UPDATED event.

    Args:
        game_id: ID of the game whose quality was updated
        old_score: Previous quality score (0-1)
        new_score: New quality score (0-1)
        config: Configuration key (optional)
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "QUALITY_SCORE_UPDATED",
        {
            "game_id": game_id,
            "old_score": old_score,
            "new_score": new_score,
            "config": config,
        },
        source
    )


async def emit_quality_distribution_changed_safe(
    config: str,
    avg_quality: float,
    high_quality_ratio: float,
    low_quality_ratio: float,
    total_games: int,
    source: str = ""
) -> bool:
    """Safely emit QUALITY_DISTRIBUTION_CHANGED event.

    Args:
        config: Configuration key (e.g., "square8_2p")
        avg_quality: Average quality score (0-1)
        high_quality_ratio: Ratio of high-quality games (0-1)
        low_quality_ratio: Ratio of low-quality games (0-1)
        total_games: Total number of games
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "QUALITY_DISTRIBUTION_CHANGED",
        {
            "config": config,
            "avg_quality": avg_quality,
            "high_quality_ratio": high_quality_ratio,
            "low_quality_ratio": low_quality_ratio,
            "total_games": total_games,
        },
        source
    )


async def emit_high_quality_data_available_safe(
    config: str,
    high_quality_count: int,
    avg_quality: float,
    source: str = ""
) -> bool:
    """Safely emit HIGH_QUALITY_DATA_AVAILABLE event.

    Indicates that enough high-quality data is available for training.

    Args:
        config: Configuration key
        high_quality_count: Number of high-quality games
        avg_quality: Average quality score of high-quality games
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "HIGH_QUALITY_DATA_AVAILABLE",
        {
            "config": config,
            "high_quality_count": high_quality_count,
            "avg_quality": avg_quality,
        },
        source
    )


async def emit_low_quality_data_warning_safe(
    config: str,
    low_quality_count: int,
    low_quality_ratio: float,
    avg_quality: float,
    source: str = ""
) -> bool:
    """Safely emit LOW_QUALITY_DATA_WARNING event.

    Warns that data quality is below acceptable threshold.

    Args:
        config: Configuration key
        low_quality_count: Number of low-quality games
        low_quality_ratio: Ratio of low-quality games (0-1)
        avg_quality: Current average quality
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "LOW_QUALITY_DATA_WARNING",
        {
            "config": config,
            "low_quality_count": low_quality_count,
            "low_quality_ratio": low_quality_ratio,
            "avg_quality": avg_quality,
        },
        source
    )


# =============================================================================
# Tier promotion event helpers (December 2025)
# =============================================================================

async def emit_tier_promotion_safe(
    config: str,
    old_tier: str,
    new_tier: str,
    model_id: str = "",
    win_rate: float = 0.0,
    elo: float = 0.0,
    games_played: int = 0,
    source: str = ""
) -> bool:
    """Safely emit TIER_PROMOTION event for difficulty ladder progression.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_tier: Previous tier (e.g., "D4")
        new_tier: New tier after promotion (e.g., "D5")
        model_id: ID of the model being promoted
        win_rate: Win rate that triggered promotion
        elo: Current Elo rating
        games_played: Number of games played at current tier
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    return await emit_event_safe(
        "TIER_PROMOTION",
        {
            "config": config,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "model_id": model_id,
            "win_rate": win_rate,
            "elo": elo,
            "games_played": games_played,
        },
        source
    )


# =============================================================================
# Sync wrappers (for non-async contexts)
# =============================================================================

def emit_sync(
    event_type: str,
    payload: Dict[str, Any],
    source: str = ""
) -> bool:
    """Synchronous wrapper for emitting events.

    Creates a new event loop if needed. Use sparingly - prefer async version.

    Args:
        event_type: Event type name
        payload: Event payload dictionary
        source: Source component name

    Returns:
        True if emitted successfully, False otherwise.
    """
    try:
        asyncio.get_running_loop()
        # Already in async context - schedule as task
        asyncio.create_task(emit_event_safe(event_type, payload, source))
        return True
    except RuntimeError:
        # No running loop - create one
        try:
            return asyncio.run(emit_event_safe(event_type, payload, source))
        except Exception as e:
            logger.debug(f"Failed to emit event synchronously: {e}")
            return False


# =============================================================================
# Re-exports for convenience
# =============================================================================

# These allow direct use of the types when event bus is available
DataEventType = _DataEventType
DataEvent = _DataEvent
EventBus = _EventBus

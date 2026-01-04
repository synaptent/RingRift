"""Centralized Event Emitters for RingRift AI.

.. deprecated:: December 2025
    This module is deprecated. Use ``app.coordination.event_router`` directly:

    from app.coordination.event_router import get_event_bus, DataEvent, DataEventType

    bus = get_event_bus()
    await bus.publish(DataEvent(
        event_type=DataEventType.TRAINING_COMPLETED,
        payload={"job_id": "...", "board_type": "...", ...},
        source="my_component",
    ))

    This module will be removed in Q2 2026.

This module provides typed emit functions for all event types, eliminating
the need for each module to re-implement event emission logic.

Usage (DEPRECATED):
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
import time
import warnings
from datetime import datetime
from typing import Any

from app.coordination.event_utils import make_config_key

logger = logging.getLogger(__name__)

# Emit deprecation warning on import (December 2025)
warnings.warn(
    "app.coordination.event_emitters is deprecated. "
    "Use app.coordination.event_router directly (get_event_bus(), DataEvent, DataEventType). "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Event Bus Imports (with atomic state to prevent race conditions)
# =============================================================================


class _EventState:
    """Atomic state for event availability.

    December 27, 2025: Created to fix COORDINATOR_HEALTHY race condition.
    Ensures HAS_DATA_EVENTS and DataEventType are always consistent.
    Previously, fire-and-forget tasks could see HAS_DATA_EVENTS=True but
    DataEventType=None due to module-level variable inconsistency.
    """

    def __init__(self) -> None:
        self.stage_available = False
        self.data_available = False
        self.router_available = False
        self.event_type = None
        self.data_event = None
        self.stage_event = None
        self.stage_result = None
        self.get_data_bus = None
        self.get_stage_bus = None
        self.get_router = None


_state = _EventState()

# Single import block - no backward compat second block that causes inconsistency
try:
    from app.coordination.event_router import (
        StageCompletionResult,
        StageEvent,
        get_stage_event_bus as _get_stage_bus,
        DataEvent,
        DataEventType,
        get_event_bus as _get_data_bus,
    )
    _state.stage_available = True
    _state.data_available = True
    _state.event_type = DataEventType
    _state.data_event = DataEvent
    _state.stage_event = StageEvent
    _state.stage_result = StageCompletionResult
    _state.get_data_bus = _get_data_bus
    _state.get_stage_bus = _get_stage_bus
except ImportError:
    # Leave _state with defaults (all False/None)
    StageEvent = None
    StageCompletionResult = None
    DataEvent = None
    DataEventType = None

    def _get_stage_bus():
        return None

    def _get_data_bus():
        return None

# Unified event router
try:
    from app.coordination.event_router import get_router as _get_router
    _state.router_available = True
    _state.get_router = _get_router
except ImportError:
    pass

# Backward-compat module-level aliases (for existing code)
# These are now derived from atomic state, so they're always consistent
HAS_STAGE_EVENTS = _state.stage_available
HAS_DATA_EVENTS = _state.data_available
HAS_EVENT_ROUTER = _state.router_available


def get_stage_bus() -> Any | None:
    """Get stage event bus (backward-compat wrapper)."""
    if _state.get_stage_bus:
        return _state.get_stage_bus()
    return None


def get_data_bus() -> Any | None:
    """Get data event bus (backward-compat wrapper)."""
    if _state.get_data_bus:
        return _state.get_data_bus()
    return None


def get_event_router() -> Any | None:
    """Get unified event router (backward-compat wrapper)."""
    if _state.get_router:
        return _state.get_router()
    return None


# Configuration: whether to use unified router or direct bus access
# Set to True to route all events through the unified router
USE_UNIFIED_ROUTER = True


# =============================================================================
# Helper Functions
# =============================================================================

async def _emit_data_event(
    event_type: DataEventType,
    payload: dict[str, Any],
    source: str = "event_emitters",
    log_message: str | None = None,
    log_level: str = "debug",
) -> bool:
    """Emit a DataEvent with standardized error handling.

    Consolidates the repeated try/except/bus.publish pattern used by
    30+ DataEvent emitter functions, saving ~300 LOC of boilerplate.

    December 2025: Created during code consolidation initiative.

    Args:
        event_type: The DataEventType enum value
        payload: Event payload dict (timestamp auto-added if missing)
        source: Event source identifier
        log_message: Optional log message (uses event_type.value if None)
        log_level: Log level for success ("debug", "info", "warning")

    Returns:
        True if emitted successfully, False otherwise
    """
    if not HAS_DATA_EVENTS:
        return False

    try:
        bus = get_data_bus()
        if bus is None:
            return False

        # Auto-add timestamp if not present
        if "timestamp" not in payload:
            payload["timestamp"] = _get_timestamp()

        event = DataEvent(
            event_type=event_type,
            payload=payload,
            source=source,
        )

        await bus.publish(event)

        msg = log_message or f"Emitted {event_type.value}"
        if log_level == "info":
            logger.info(msg)
        elif log_level == "warning":
            logger.warning(msg)
        else:
            logger.debug(msg)

        return True

    except asyncio.CancelledError:
        # December 29, 2025: Let task cancellation propagate
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
        # December 29, 2025: Narrowed from bare except Exception
        # - AttributeError: Bus not available or method doesn't exist
        # - TypeError: Wrong payload type or arguments
        # - ValueError: Invalid event type
        # - RuntimeError: No event loop, bus not initialized
        # - OSError: Network/IO issues for cross-process events
        logger.debug(f"Failed to emit {event_type.value}: {e}")
        return False


def _emit_data_event_sync(
    event_type: DataEventType,
    payload: dict[str, Any],
    source: str = "event_emitters",
    log_message: str | None = None,
) -> bool:
    """Synchronous version of _emit_data_event for use in sync contexts.

    Uses fire_and_forget to schedule the async emission.

    Args:
        event_type: The DataEventType enum value
        payload: Event payload dict
        source: Event source identifier
        log_message: Optional log message

    Returns:
        True if scheduled successfully, False otherwise
    """
    if not HAS_DATA_EVENTS:
        return False

    try:
        from app.utils.async_utils import fire_and_forget
        import asyncio

        asyncio.get_running_loop()

        # Add timestamp if missing
        if "timestamp" not in payload:
            payload["timestamp"] = _get_timestamp()

        async def _emit():
            bus = get_data_bus()
            if bus is None:
                return
            event = DataEvent(
                event_type=event_type,
                payload=payload,
                source=source,
            )
            await bus.publish(event)
            msg = log_message or f"Emitted {event_type.value}"
            logger.debug(msg)

        fire_and_forget(_emit(), name=f"emit_{event_type.value}")
        return True

    except RuntimeError:
        # No event loop - cannot emit sync
        return False
    except (AttributeError, TypeError, ValueError, ImportError) as e:
        # December 29, 2025: Narrowed from bare except Exception
        logger.debug(f"Failed to schedule {event_type.value} emission: {e}")
        return False


# =============================================================================
# Retry-Enabled Emission (December 2025)
# =============================================================================

# Critical events that should be retried on failure
CRITICAL_EVENT_TYPES = {
    "training_started",
    "training_completed",
    "model_promoted",
    "consolidation_complete",
    "data_sync_completed",
    "evaluation_completed",
    "regression_detected",
    "promotion_failed",
}


async def _emit_data_event_with_retry(
    event_type: DataEventType,
    payload: dict[str, Any],
    source: str = "event_emitters",
    log_message: str | None = None,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> bool:
    """Emit a DataEvent with exponential backoff retry for critical events.

    December 2025: Added to ensure critical pipeline events aren't lost due
    to transient failures (network issues, bus not ready, etc.).

    Args:
        event_type: The DataEventType enum value
        payload: Event payload dict (timestamp auto-added if missing)
        source: Event source identifier
        log_message: Optional log message (uses event_type.value if None)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 0.5)

    Returns:
        True if emitted successfully, False if all retries exhausted
    """
    if not HAS_DATA_EVENTS:
        return False

    # Auto-add timestamp if not present
    if "timestamp" not in payload:
        payload["timestamp"] = _get_timestamp()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            bus = get_data_bus()
            if bus is None:
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return False

            event = DataEvent(
                event_type=event_type,
                payload=payload,
                source=source,
            )

            await bus.publish(event)

            msg = log_message or f"Emitted {event_type.value}"
            if attempt > 0:
                logger.info(f"{msg} (after {attempt} retries)")
            else:
                logger.debug(msg)

            return True

        except asyncio.CancelledError:
            # December 29, 2025: Let task cancellation propagate
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            # December 29, 2025: Narrowed from bare except Exception
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.debug(
                    f"Retry {attempt + 1}/{max_retries} for {event_type.value} "
                    f"after {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    f"Failed to emit {event_type.value} after {max_retries} retries: {last_error}"
                )

    return False


def is_critical_event(event_type: DataEventType | str) -> bool:
    """Check if an event type is critical and should use retry logic.

    Args:
        event_type: Event type (enum or string value)

    Returns:
        True if the event is critical
    """
    if hasattr(event_type, "value"):
        event_str = event_type.value
    else:
        event_str = str(event_type)
    return event_str in CRITICAL_EVENT_TYPES


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


async def _emit_via_router(
    event_type: str,
    payload: dict[str, Any],
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

    except asyncio.CancelledError:
        # December 29, 2025: Let task cancellation propagate
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        # December 29, 2025: Narrowed from bare except Exception
        logger.debug(f"Failed to emit {event_type} via router: {e}")
        return False


def _emit_via_router_sync(
    event_type: str,
    payload: dict[str, Any],
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

    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        # December 29, 2025: Narrowed from bare except Exception
        logger.debug(f"Failed to emit {event_type} sync via router: {e}")
        return False


async def _emit_stage_event(
    event: StageEvent,
    result: StageCompletionResult,
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
        # December 29, 2025: Fixed payload construction to use result.to_dict()
        # and add derived fields expected by subscribers (config_key, games_added)
        if hasattr(result, 'to_dict'):
            payload = result.to_dict()
        else:
            # Fallback for old-style results
            payload = {
                "event": event.value if hasattr(event, 'value') else str(event),
                "success": result.success if hasattr(result, 'success') else True,
            }

        # Add derived fields for backward compatibility with subscribers
        # Subscribers expect config_key and games_added, but StageCompletionResult
        # provides board_type, num_players, and games_generated
        if hasattr(result, 'board_type') and hasattr(result, 'num_players'):
            payload["config_key"] = f"{result.board_type}_{result.num_players}p"
        if hasattr(result, 'games_generated'):
            payload["games_added"] = result.games_generated

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

    except asyncio.CancelledError:
        # December 29, 2025: Let task cancellation propagate
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
        # December 29, 2025: Narrowed from bare except Exception
        logger.debug(f"Failed to emit {event}: {e}")
        return False


# Critical stage events that should use retry logic
CRITICAL_STAGE_EVENTS = {
    "training_complete",
    "training_failed",
    "evaluation_complete",
    "promotion_complete",
    "promotion_failed",
}


async def _emit_stage_event_with_retry(
    event: StageEvent,
    result: StageCompletionResult,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> bool:
    """Emit a StageEvent with exponential backoff retry.

    December 2025: Added for critical pipeline stage events.

    Args:
        event: The StageEvent type
        result: The completion result
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        True if emitted successfully
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            # Try unified router first if enabled
            if USE_UNIFIED_ROUTER and HAS_EVENT_ROUTER:
                # December 29, 2025: Fixed payload construction to use result.to_dict()
                # and add derived fields expected by subscribers (config_key, games_added)
                if hasattr(result, 'to_dict'):
                    payload = result.to_dict()
                else:
                    # Fallback for old-style results
                    payload = {
                        "event": event.value if hasattr(event, 'value') else str(event),
                        "success": result.success if hasattr(result, 'success') else True,
                    }

                # Add derived fields for backward compatibility with subscribers
                if hasattr(result, 'board_type') and hasattr(result, 'num_players'):
                    payload["config_key"] = f"{result.board_type}_{result.num_players}p"
                if hasattr(result, 'games_generated'):
                    payload["games_added"] = result.games_generated

                if await _emit_via_router(event.value if hasattr(event, 'value') else str(event), payload, "stage_event"):
                    if attempt > 0:
                        logger.info(f"Emitted {event} (after {attempt} retries)")
                    return True

            # Fallback to direct stage bus
            if not HAS_STAGE_EVENTS:
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return False

            bus = get_stage_bus()
            if bus is None:
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return False

            await bus.emit(result)
            if attempt > 0:
                logger.info(f"Emitted {result.event.value} (after {attempt} retries)")
            else:
                logger.debug(f"Emitted {result.event.value}")
            return True

        except asyncio.CancelledError:
            # December 29, 2025: Let task cancellation propagate
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            # December 29, 2025: Narrowed from bare except Exception
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.debug(
                    f"Retry {attempt + 1}/{max_retries} for {event} after {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(f"Failed to emit {event} after {max_retries} retries: {last_error}")

    return False


def _emit_stage_event_sync(
    event: StageEvent,
    result: StageCompletionResult,
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
            from app.utils.async_utils import fire_and_forget
            asyncio.get_running_loop()
            fire_and_forget(bus.emit(result), name=f"emit_{event}")
            return True
        except RuntimeError:
            # No event loop - try sync emit if available
            if hasattr(bus, 'emit_sync'):
                bus.emit_sync(result)
                return True
            return False

    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        # December 29, 2025: Narrowed from bare except Exception
        # - AttributeError: Bus not available or method doesn't exist
        # - TypeError: Wrong argument types
        # - ValueError: Invalid event type
        # - RuntimeError: Event loop issues
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
    final_loss: float | None = None,
    final_elo: float | None = None,
    model_path: str | None = None,
    epochs_completed: int = 0,
    **metadata,
) -> bool:
    """Emit TRAINING_COMPLETE or TRAINING_FAILED event.

    This is a critical event and uses retry logic to ensure delivery.

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

    # Use retry for this critical event (December 2025)
    return await _emit_stage_event_with_retry(event, result)


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


def emit_training_started_sync(
    job_id: str,
    board_type: str,
    num_players: int,
    model_version: str = "",
    node_name: str = "",
    **metadata,
) -> bool:
    """Synchronous version of emit_training_started.

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

    return _emit_stage_event_sync(StageEvent.TRAINING_STARTED, result)


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
    error: str | None = None,
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
    win_rate: float | None = None,
    elo_delta: float | None = None,
    games_played: int = 0,
    **metadata,
) -> bool:
    """Emit EVALUATION_COMPLETE event.

    This is a critical event and uses retry logic to ensure delivery.

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

    # Use retry for this critical event (December 2025)
    return await _emit_stage_event_with_retry(StageEvent.EVALUATION_COMPLETE, result)


# =============================================================================
# Promotion Events
# =============================================================================

async def emit_promotion_complete(
    model_id: str,
    board_type: str,
    num_players: int,
    promotion_type: str = "production",
    elo_improvement: float | None = None,
    model_path: str | None = None,
    **metadata,
) -> bool:
    """Emit PROMOTION_COMPLETE event.

    This is a critical event and uses retry logic to ensure delivery.

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

    # Use retry for this critical event (December 2025)
    return await _emit_stage_event_with_retry(StageEvent.PROMOTION_COMPLETE, result)


def emit_promotion_complete_sync(
    model_id: str,
    board_type: str,
    num_players: int,
    promotion_type: str = "production",
    elo_improvement: float | None = None,
    model_path: str | None = None,
    **metadata,
) -> bool:
    """Synchronous version of emit_promotion_complete.

    Use this in sync contexts like CLI scripts (e.g., auto_promote.py).

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

    return _emit_stage_event_sync(StageEvent.PROMOTION_COMPLETE, result)


# =============================================================================
# Export Events
# =============================================================================

async def emit_npz_export_complete(
    board_type: str,
    num_players: int,
    samples_exported: int,
    games_exported: int,
    output_path: str,
    success: bool = True,
    duration_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit NPZ_EXPORT_COMPLETE event to trigger training.

    This event enables automatic training triggering when export completes.

    Args:
        board_type: Board type exported
        num_players: Number of players
        samples_exported: Number of training samples exported
        games_exported: Number of games exported
        output_path: Path to the NPZ file
        success: Whether export succeeded
        duration_seconds: Duration of export
        **metadata: Additional metadata (feature_version, encoder, etc.)

    Returns:
        True if emitted successfully
    """
    if not HAS_STAGE_EVENTS:
        return False

    config_key = make_config_key(board_type, num_players)

    result = StageCompletionResult(
        event=StageEvent.NPZ_EXPORT_COMPLETE,
        success=success,
        iteration=0,
        timestamp=_get_timestamp(),
        board_type=board_type,
        num_players=num_players,
        games_generated=games_exported,
        metadata={
            "config_key": config_key,
            "samples_exported": samples_exported,
            "games_exported": games_exported,
            "output_path": output_path,
            "duration_seconds": duration_seconds,
            **metadata,
        },
    )

    return await _emit_stage_event(StageEvent.NPZ_EXPORT_COMPLETE, result)


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
    components: list | None = None,
    errors: list | None = None,
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


async def emit_sync_failure_critical(
    consecutive_failures: int,
    last_success: float | None = None,
    source: str = "sync_coordinator",
    **metadata,
) -> bool:
    """Emit SYNC_FAILURE_CRITICAL event when sync fails multiple times.

    December 2025: Added to trigger recovery when background sync repeatedly fails.
    This event should trigger:
    - Alert to operators
    - Potential failover to alternate sync mechanism
    - Circuit breaker activation

    Args:
        consecutive_failures: Number of consecutive sync failures
        last_success: Timestamp of last successful sync (Unix epoch)
        source: Component emitting this event
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    if not HAS_DATA_EVENTS:
        return False

    import time
    time_since_success = None
    if last_success:
        time_since_success = time.time() - last_success

    return await _emit_data_event(
        DataEventType.SYNC_FAILURE_CRITICAL,
        {
            "consecutive_failures": consecutive_failures,
            "last_success": last_success,
            "time_since_success_seconds": time_since_success,
            "source": source,
            **metadata,
        },
    )


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
    """Emit quality score updated event via DataEventBus."""
    if not HAS_DATA_EVENTS:
        return False
    return await _emit_data_event(
        DataEventType.QUALITY_SCORE_UPDATED,
        {
            "board_type": board_type,
            "num_players": num_players,
            "avg_quality": avg_quality,
            "total_games": total_games,
            "high_quality_games": high_quality_games,
            **metadata,
        },
        log_message="Emitted quality_score_updated event",
    )


async def emit_game_quality_score(
    game_id: str,
    quality_score: float,
    quality_category: str,
    training_weight: float,
    game_length: int = 0,
    is_decisive: bool = True,
    source: str = "",
    **metadata,
) -> bool:
    """Emit per-game quality score event via DataEventBus."""
    # Defensive check: DataEventType may be None during import failures
    if not HAS_DATA_EVENTS or DataEventType is None:
        return False
    return await _emit_data_event(
        DataEventType.QUALITY_SCORE_UPDATED,
        {
            "game_id": game_id,
            "quality_score": quality_score,
            "quality_category": quality_category,
            "training_weight": training_weight,
            "game_length": game_length,
            "is_decisive": is_decisive,
            "source": source,
            "is_per_game": True,
            **metadata,
        },
        source=source or "unified_quality",
        log_message=f"Emitted game quality score event for {game_id}",
    )


async def emit_quality_penalty_applied(
    config_key: str,
    penalty: float,
    reason: str,
    current_weight: float = 1.0,
    source: str = "",
    **metadata,
) -> bool:
    """Emit QUALITY_PENALTY_APPLIED event when quality penalty is applied to a config.

    This event triggers the curriculum feedback loop to reduce selfplay rate
    for configs with degraded quality, closing the loop:
    quality degradation → penalty → reduced selfplay → better quality data.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        penalty: Penalty factor applied (0.0-1.0, where 1.0 = full penalty)
        reason: Why penalty was applied (e.g., "low_quality_score", "high_error_rate")
        current_weight: Current curriculum weight after penalty
        source: Module/function that triggered the penalty
        **metadata: Additional metadata

    Returns:
        True if emitted successfully

    Example:
        await emit_quality_penalty_applied(
            config_key="hex8_2p",
            penalty=0.15,
            reason="quality_score_below_threshold",
            current_weight=0.85,
        )
    """
    return await _emit_data_event(
        DataEventType.QUALITY_PENALTY_APPLIED,
        {
            "config_key": config_key,
            "penalty": penalty,
            "reason": reason,
            "current_weight": current_weight,
            "timestamp": _get_timestamp(),
            **metadata,
        },
        source=source or "quality_monitor",
        log_message=f"Emitted quality_penalty_applied for {config_key}: penalty={penalty:.2f}, reason={reason}",
        log_level="warning",
    )


# =============================================================================
# Data Sync Events
# =============================================================================

async def emit_new_games(
    host: str,
    new_games: int,
    total_games: int,
    source: str = "",
) -> bool:
    """Emit NEW_GAMES_AVAILABLE event via DataEventBus."""
    return await _emit_data_event(
        DataEventType.NEW_GAMES_AVAILABLE,
        {"host": host, "new_games": new_games, "total_games": total_games},
        source=source or "event_emitters",
        log_message=f"Emitted new_games_available event: {new_games} from {host}",
    )


# =============================================================================
# Task Events (Generic)
# =============================================================================

async def emit_task_complete(
    task_id: str,
    task_type: str,
    success: bool = True,
    node_id: str = "",
    duration_seconds: float = 0.0,
    result_data: dict[str, Any] | None = None,
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


async def emit_task_spawned(
    task_id: str,
    task_type: str,
    node_id: str,
    config: dict[str, Any] | None = None,
    **metadata,
) -> bool:
    """Emit TASK_SPAWNED event when a task starts execution.

    Args:
        task_id: Unique task identifier
        task_type: Type of task (selfplay, training, evaluation, sync)
        node_id: Node executing the task
        config: Task configuration
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.TASK_SPAWNED,
        {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "config": config or {},
            "started_at": _get_timestamp(),
            **metadata,
        },
        log_message=f"Emitted task_spawned: {task_id} ({task_type}) on {node_id}",
        log_level="info",
    )


async def emit_task_cancelled(
    task_id: str,
    task_type: str,
    node_id: str,
    reason: str,
    requested_by: str = "user",
    **metadata,
) -> bool:
    """Emit TASK_CANCELLED event for user-initiated task cancellations.

    Different from TASK_ABANDONED which is for system-initiated abandonments.

    Args:
        task_id: Task being cancelled
        task_type: Type of task
        node_id: Node running the task
        reason: Why task is being cancelled
        requested_by: Who requested cancellation (user, coordinator, etc.)
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.TASK_CANCELLED,
        {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "reason": reason,
            "requested_by": requested_by,
            "cancelled_at": _get_timestamp(),
            **metadata,
        },
        log_message=f"Emitted task_cancelled: {task_id} ({reason})",
        log_level="info",
    )


async def emit_task_heartbeat(
    task_id: str,
    task_type: str,
    node_id: str,
    progress_percent: float = 0.0,
    games_completed: int = 0,
    samples_generated: int = 0,
    elapsed_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit TASK_HEARTBEAT event for periodic task progress updates.

    Args:
        task_id: Task identifier
        task_type: Type of task
        node_id: Node running the task
        progress_percent: Estimated completion percentage (0-100)
        games_completed: Number of games completed (for selfplay)
        samples_generated: Number of samples generated
        elapsed_seconds: Time elapsed since task start
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.TASK_HEARTBEAT,
        {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "progress_percent": progress_percent,
            "games_completed": games_completed,
            "samples_generated": samples_generated,
            "elapsed_seconds": elapsed_seconds,
            "heartbeat_at": _get_timestamp(),
            **metadata,
        },
        log_message=f"Emitted task_heartbeat: {task_id} ({progress_percent:.1f}%)",
        log_level="debug",
    )


async def emit_task_failed(
    task_id: str,
    task_type: str,
    node_id: str,
    error: str,
    error_type: str = "unknown",
    traceback: str = "",
    retryable: bool = False,
    **metadata,
) -> bool:
    """Emit TASK_FAILED event when a task fails during execution.

    Args:
        task_id: Failed task identifier
        task_type: Type of task
        node_id: Node where task failed
        error: Error message
        error_type: Classification of error (timeout, oom, crash, etc.)
        traceback: Full traceback if available
        retryable: Whether this task could be retried
        **metadata: Additional metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.TASK_FAILED,
        {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "error": error,
            "error_type": error_type,
            "traceback": traceback,
            "retryable": retryable,
            "failed_at": _get_timestamp(),
            **metadata,
        },
        log_message=f"Emitted task_failed: {task_id} ({error_type})",
        log_level="warning",
    )


# =============================================================================
# Optimization Events (December 2025)
# =============================================================================

async def emit_optimization_triggered(
    optimization_type: str,  # "cmaes" or "nas"
    run_id: str,
    reason: str,
    parameters_searched: int = 0,
    search_space: dict[str, Any] | None = None,
    generations: int = 0,
    population_size: int = 0,
    **metadata,
) -> bool:
    """Emit CMAES_TRIGGERED or NAS_TRIGGERED event."""
    if not HAS_DATA_EVENTS:
        return False
    event_type = (
        DataEventType.CMAES_TRIGGERED
        if optimization_type.lower() == "cmaes"
        else DataEventType.NAS_TRIGGERED
    )
    return await _emit_data_event(
        event_type,
        {
            "run_id": run_id,
            "reason": reason,
            "parameters_searched": parameters_searched,
            "search_space": search_space or {},
            "generations": generations,
            "population_size": population_size,
            **metadata,
        },
    )


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
    """Emit PLATEAU_DETECTED event."""
    if not HAS_DATA_EVENTS:
        return False
    return await _emit_data_event(
        DataEventType.PLATEAU_DETECTED,
        {
            "metric_name": metric_name,
            "plateau_type": plateau_type,
            "current_value": current_value,
            "best_value": best_value,
            "epochs_since_improvement": epochs_since_improvement,
            **metadata,
        },
        log_message="Emitted plateau_detected event",
    )


async def emit_hyperparameter_updated(
    config: str,
    param_name: str,
    old_value: Any,
    new_value: Any,
    optimizer: str = "manual",  # "cmaes", "nas", "manual", "adaptive"
    **metadata,
) -> bool:
    """Emit HYPERPARAMETER_UPDATED event."""
    return await _emit_data_event(
        DataEventType.HYPERPARAMETER_UPDATED,
        {
            "config": config,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "optimizer": optimizer,
            **metadata,
        },
        log_message=f"Emitted hyperparameter_updated event for {param_name}",
    )


async def emit_regression_detected(
    metric_name: str,
    current_value: float,
    previous_value: float,
    severity: str = "minor",  # "minor", "moderate", "severe", "critical"
    **metadata,
) -> bool:
    """Emit REGRESSION_DETECTED event."""
    return await _emit_data_event(
        DataEventType.REGRESSION_DETECTED,
        {
            "metric_name": metric_name,
            "current_value": current_value,
            "previous_value": previous_value,
            "severity": severity,
            "regression_amount": previous_value - current_value,
            **metadata,
        },
        log_message=f"Emitted regression_detected event (severity={severity})",
    )


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
    """Emit BACKPRESSURE_ACTIVATED event."""
    if not HAS_DATA_EVENTS:
        return False
    return await _emit_data_event(
        DataEventType.BACKPRESSURE_ACTIVATED,
        {
            "node_id": node_id,
            "level": level,
            "reason": reason,
            "resource_type": resource_type,
            "utilization": utilization,
            **metadata,
        },
        log_message=f"Emitted backpressure_activated event for {node_id}",
    )


async def emit_backpressure_released(
    node_id: str,
    previous_level: str = "",
    duration_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit BACKPRESSURE_RELEASED event."""
    return await _emit_data_event(
        DataEventType.BACKPRESSURE_RELEASED,
        {
            "node_id": node_id,
            "previous_level": previous_level,
            "duration_seconds": duration_seconds,
            **metadata,
        },
        log_message=f"Emitted backpressure_released event for {node_id}",
    )


# =============================================================================
# Cache Events (December 2025)
# =============================================================================

async def emit_cache_invalidated(
    invalidation_type: str,  # "model" or "node"
    target_id: str,
    count: int,
    affected_nodes: list | None = None,
    affected_models: list | None = None,
    **metadata,
) -> bool:
    """Emit CACHE_INVALIDATED event."""
    if not HAS_DATA_EVENTS:
        return False
    return await _emit_data_event(
        DataEventType.CACHE_INVALIDATED,
        {
            "invalidation_type": invalidation_type,
            "target_id": target_id,
            "count": count,
            "affected_nodes": affected_nodes or [],
            "affected_models": affected_models or [],
            **metadata,
        },
        log_message=f"Emitted cache_invalidated event: {invalidation_type}={target_id}",
    )


# =============================================================================
# Host/Node Events (December 2025)
# =============================================================================

async def emit_host_online(
    node_id: str,
    host_type: str = "",
    capabilities: dict[str, Any] | None = None,
    **metadata,
) -> bool:
    """Emit HOST_ONLINE event."""
    if not HAS_DATA_EVENTS:
        return False
    return await _emit_data_event(
        DataEventType.HOST_ONLINE,
        {
            "node_id": node_id,
            "host_id": node_id,  # Alias for compatibility
            "host_type": host_type,
            "capabilities": capabilities or {},
            **metadata,
        },
        log_message=f"Emitted host_online event for {node_id}",
    )


async def emit_host_offline(
    node_id: str,
    reason: str = "",
    **metadata,
) -> bool:
    """Emit HOST_OFFLINE event."""
    return await _emit_data_event(
        DataEventType.HOST_OFFLINE,
        {"node_id": node_id, "host_id": node_id, "reason": reason, **metadata},
        log_message=f"Emitted host_offline event for {node_id}",
    )


async def emit_node_recovered(
    node_id: str,
    recovery_type: str = "automatic",
    offline_duration_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit NODE_RECOVERED event."""
    return await _emit_data_event(
        DataEventType.NODE_RECOVERED,
        {
            "node_id": node_id,
            "host_id": node_id,
            "recovery_type": recovery_type,
            "offline_duration_seconds": offline_duration_seconds,
            **metadata,
        },
        log_message=f"Emitted node_recovered event for {node_id}",
    )


async def emit_node_activated(
    node_id: str,
    activation_type: str = "selfplay",
    config_key: str = "",
    **metadata,
) -> bool:
    """Emit NODE_ACTIVATED event.

    December 2025: Emitted when a node is activated by cluster watchdog
    or other activation mechanisms.

    Args:
        node_id: Identifier of the activated node
        activation_type: Type of activation (selfplay, training, etc.)
        config_key: Board configuration activated (e.g., "hex8_2p")
        **metadata: Additional metadata
    """
    return await _emit_data_event(
        DataEventType.NODE_ACTIVATED,
        {
            "node_id": node_id,
            "host_id": node_id,
            "activation_type": activation_type,
            "config_key": config_key,
            **metadata,
        },
        log_message=f"Emitted node_activated event for {node_id}",
    )


async def emit_node_suspect(
    node_id: str,
    last_seen: float | None = None,
    seconds_since_heartbeat: float = 0.0,
    **metadata,
) -> bool:
    """Emit NODE_SUSPECT event when a node enters SUSPECT state.

    December 2025: Added for peer state transition tracking.
    SUSPECT is a grace period between ALIVE and DEAD to reduce false positives.

    Args:
        node_id: Identifier of the suspect node
        last_seen: Timestamp when node was last seen
        seconds_since_heartbeat: Seconds since last heartbeat
        **metadata: Additional metadata
    """
    return await _emit_data_event(
        DataEventType.NODE_SUSPECT,
        {
            "node_id": node_id,
            "host_id": node_id,
            "last_seen": last_seen,
            "seconds_since_heartbeat": seconds_since_heartbeat,
            **metadata,
        },
        log_message=f"Emitted node_suspect event for {node_id} ({seconds_since_heartbeat:.0f}s since heartbeat)",
        log_level="warning",
    )


async def emit_node_retired(
    node_id: str,
    reason: str = "manual",
    last_seen: float | None = None,
    total_uptime_seconds: float = 0.0,
    **metadata,
) -> bool:
    """Emit NODE_RETIRED event when a node is retired from the cluster.

    December 2025: Added for peer state transition tracking.
    Retired nodes are excluded from job allocation but may be recovered later.

    Args:
        node_id: Identifier of the retired node
        reason: Why the node was retired (manual, timeout, error, capacity)
        last_seen: Timestamp when node was last seen
        total_uptime_seconds: Total uptime before retirement
        **metadata: Additional metadata
    """
    return await _emit_data_event(
        DataEventType.NODE_RETIRED,
        {
            "node_id": node_id,
            "host_id": node_id,
            "reason": reason,
            "last_seen": last_seen,
            "total_uptime_seconds": total_uptime_seconds,
            **metadata,
        },
        log_message=f"Emitted node_retired event for {node_id} (reason: {reason})",
        log_level="info",
    )


# =============================================================================
# Error Recovery & Resilience Events (December 2025)
# =============================================================================

async def emit_training_rollback_needed(
    model_id: str,
    reason: str,
    checkpoint_path: str | None = None,
    severity: str = "moderate",
    **metadata,
) -> bool:
    """Emit TRAINING_ROLLBACK_NEEDED event."""
    return await _emit_data_event(
        DataEventType.TRAINING_ROLLBACK_NEEDED,
        {
            "model_id": model_id,
            "reason": reason,
            "checkpoint_path": checkpoint_path,
            "severity": severity,
            **metadata,
        },
        log_message=f"Emitted training_rollback_needed for {model_id}: {reason}",
        log_level="warning",
    )


async def emit_handler_failed(
    handler_name: str,
    event_type: str,
    error: str,
    coordinator: str = "",
    **metadata,
) -> bool:
    """Emit HANDLER_FAILED event when an event handler throws an exception."""
    # Note: Using log_level="warning" since error() was too noisy
    return await _emit_data_event(
        DataEventType.HANDLER_FAILED,
        {
            "handler_name": handler_name,
            "event_type": event_type,
            "error": error,
            "coordinator": coordinator,
            **metadata,
        },
        log_message=f"Emitted handler_failed: {handler_name} on {event_type}",
        log_level="warning",
    )


async def emit_handler_timeout(
    handler_name: str,
    event_type: str,
    timeout_seconds: float,
    coordinator: str = "",
    **metadata,
) -> bool:
    """Emit HANDLER_TIMEOUT event when an event handler times out."""
    return await _emit_data_event(
        DataEventType.HANDLER_TIMEOUT,
        {
            "handler_name": handler_name,
            "event_type": event_type,
            "timeout_seconds": timeout_seconds,
            "coordinator": coordinator,
            **metadata,
        },
        log_message=f"Emitted handler_timeout: {handler_name} after {timeout_seconds}s",
        log_level="warning",
    )


async def emit_coordinator_health_degraded(
    coordinator_name: str,
    reason: str,
    health_score: float = 0.0,
    issues: list | None = None,
    **metadata,
) -> bool:
    """Emit COORDINATOR_HEALTH_DEGRADED event."""
    return await _emit_data_event(
        DataEventType.COORDINATOR_HEALTH_DEGRADED,
        {
            "coordinator_name": coordinator_name,
            "reason": reason,
            "health_score": health_score,
            "issues": issues or [],
            **metadata,
        },
        log_message=f"Emitted coordinator_health_degraded: {coordinator_name}",
        log_level="warning",
    )


async def emit_coordinator_shutdown(
    coordinator_name: str,
    reason: str = "graceful",
    remaining_tasks: int = 0,
    state_snapshot: dict[str, Any] | None = None,
    **metadata,
) -> bool:
    """Emit COORDINATOR_SHUTDOWN event for graceful shutdown."""
    return await _emit_data_event(
        DataEventType.COORDINATOR_SHUTDOWN,
        {
            "coordinator_name": coordinator_name,
            "reason": reason,
            "remaining_tasks": remaining_tasks,
            "state_snapshot": state_snapshot or {},
            **metadata,
        },
        log_message=f"Emitted coordinator_shutdown: {coordinator_name}",
        log_level="info",
    )


async def emit_coordinator_heartbeat(
    coordinator_name: str,
    health_score: float = 1.0,
    active_handlers: int = 0,
    events_processed: int = 0,
    **metadata,
) -> bool:
    """Emit COORDINATOR_HEARTBEAT for liveness detection."""
    return await _emit_data_event(
        DataEventType.COORDINATOR_HEARTBEAT,
        {
            "coordinator_name": coordinator_name,
            "health_score": health_score,
            "active_handlers": active_handlers,
            "events_processed": events_processed,
            **metadata,
        },
    )


async def emit_coordinator_healthy(
    coordinator_name: str,
    *,
    health_score: float = 1.0,
    uptime_seconds: float = 0.0,
    subscribed: bool = True,
    source: str = "",
) -> bool:
    """Emit COORDINATOR_HEALTHY event when coordinator initializes successfully.

    December 2025: Added for coordination_bootstrap integration to track
    successful coordinator initialization and recovery.

    Args:
        coordinator_name: Name of the coordinator
        health_score: Health score (0.0-1.0), defaults to 1.0 for newly initialized
        uptime_seconds: Seconds since initialization
        subscribed: Whether coordinator is subscribed to its events
        source: Event source identifier

    Returns:
        True if emitted successfully
    """
    # December 27, 2025: Guard using atomic state to prevent race condition
    if not _state.data_available or _state.event_type is None:
        return False
    return await _emit_data_event(
        _state.event_type.COORDINATOR_HEALTHY,
        {
            "coordinator_name": coordinator_name,
            "health_score": health_score,
            "uptime_seconds": uptime_seconds,
            "subscribed": subscribed,
        },
        source=source or "event_emitters",
        log_message=f"Emitted coordinator_healthy: {coordinator_name}",
        log_level="info",
    )


async def emit_coordinator_unhealthy(
    coordinator_name: str,
    *,
    reason: str = "initialization_failed",
    error: str | None = None,
    health_score: float = 0.0,
    source: str = "",
) -> bool:
    """Emit COORDINATOR_UNHEALTHY event when coordinator fails to initialize.

    December 2025: Added for coordination_bootstrap integration to track
    coordinator initialization failures and health degradation.

    Args:
        coordinator_name: Name of the coordinator
        reason: Reason for unhealthy status
        error: Error message if available
        health_score: Health score (0.0-1.0)
        source: Event source identifier

    Returns:
        True if emitted successfully
    """
    # December 27, 2025: Guard using atomic state to prevent race condition
    if not _state.data_available or _state.event_type is None:
        return False
    return await _emit_data_event(
        _state.event_type.COORDINATOR_UNHEALTHY,
        {
            "coordinator_name": coordinator_name,
            "reason": reason,
            "error": error or "",
            "health_score": health_score,
        },
        source=source or "event_emitters",
        log_message=f"Emitted coordinator_unhealthy: {coordinator_name} - {reason}",
        log_level="warning",
    )


async def emit_task_abandoned(
    task_id: str,
    task_type: str,
    node_id: str,
    reason: str,
    **metadata,
) -> bool:
    """Emit TASK_ABANDONED event for intentionally abandoned tasks."""
    return await _emit_data_event(
        DataEventType.TASK_ABANDONED,
        {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "reason": reason,
            **metadata,
        },
        log_message=f"Emitted task_abandoned: {task_id}",
        log_level="info",
    )


async def emit_task_orphaned(
    task_id: str,
    task_type: str,
    node_id: str,
    last_heartbeat: float,
    reason: str,
    **metadata,
) -> bool:
    """Emit TASK_ORPHANED event for tasks that lost their parent worker."""
    return await _emit_data_event(
        DataEventType.TASK_ORPHANED,
        {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "last_heartbeat": last_heartbeat,
            "reason": reason,
            **metadata,
        },
        log_message=f"Emitted task_orphaned: {task_id} ({reason})",
        log_level="info",
    )


async def emit_model_corrupted(
    model_id: str,
    model_path: str,
    corruption_type: str,
    **metadata,
) -> bool:
    """Emit MODEL_CORRUPTED event when model file corruption is detected."""
    return await _emit_data_event(
        DataEventType.MODEL_CORRUPTED,
        {
            "model_id": model_id,
            "model_path": model_path,
            "corruption_type": corruption_type,
            **metadata,
        },
        log_message=f"Emitted model_corrupted: {model_id}",
        log_level="warning",
    )


async def emit_training_rollback_completed(
    model_id: str,
    checkpoint_path: str,
    rollback_from: str,
    reason: str,
    **metadata,
) -> bool:
    """Emit TRAINING_ROLLBACK_COMPLETED event."""
    return await _emit_data_event(
        DataEventType.TRAINING_ROLLBACK_COMPLETED,
        {
            "model_id": model_id,
            "checkpoint_path": checkpoint_path,
            "rollback_from": rollback_from,
            "reason": reason,
            **metadata,
        },
        log_message=f"Emitted training_rollback_completed: {model_id} from {rollback_from}",
        log_level="info",
    )


async def emit_curriculum_updated(
    config_key: str,
    new_weight: float,
    trigger: str = "automatic",
    all_weights: dict[str, float] | None = None,
    **metadata,
) -> bool:
    """Emit CURRICULUM_REBALANCED event for a single config update.

    This is a convenience wrapper for emit_curriculum_rebalanced when
    updating a single config's weight. Used by the curriculum feedback
    system to notify the selfplay orchestrator of weight changes.

    December 2025: Added for Phase 1 self-improvement feedback loop.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        new_weight: New weight for this config
        trigger: What triggered the update (promotion, elo_change, plateau, etc.)
        all_weights: Optional dict of all current weights
        **metadata: Additional event metadata

    Returns:
        True if event was emitted successfully

    Example:
        await emit_curriculum_updated("square8_2p", 1.3, trigger="promotion")
    """
    return await emit_curriculum_rebalanced(
        config=config_key,
        old_weights={},  # Old weights often not available for single updates
        new_weights=all_weights or {config_key: new_weight},
        reason=f"config_update_{config_key}",
        trigger=trigger,
        config_key=config_key,
        new_weight=new_weight,
        **metadata,
    )


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: dict,
    new_weights: dict,
    reason: str,
    trigger: str = "automatic",
    **metadata,
) -> bool:
    """Emit CURRICULUM_REBALANCED event."""
    return await _emit_data_event(
        DataEventType.CURRICULUM_REBALANCED,
        {
            "config": config,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "reason": reason,
            "trigger": trigger,
            **metadata,
        },
        log_message=f"Emitted curriculum_rebalanced for {config}: {reason}",
        log_level="info",
    )


async def emit_training_triggered(
    config: str,
    job_id: str,
    trigger_reason: str,
    game_count: int = 0,
    threshold: int = 0,
    priority: str = "normal",
    **metadata,
) -> bool:
    """Emit event when training is triggered (before it starts)."""
    return await _emit_data_event(
        DataEventType.TRAINING_THRESHOLD_REACHED,
        {
            "config": config,
            "job_id": job_id,
            "trigger_reason": trigger_reason,
            "games": game_count,
            "threshold": threshold,
            "priority": priority,
            "event_subtype": "training_triggered",
            **metadata,
        },
        log_message=f"Emitted training_triggered for {config}: {trigger_reason}",
        log_level="info",
    )


# =============================================================================
# Cluster Health Events (December 2025)
# =============================================================================
# These emitters consolidate the try/except boilerplate from:
# - cluster_watchdog_daemon.py
# - unified_node_health_daemon.py
# - node_recovery_daemon.py
# - unified_health_manager.py


async def emit_node_unhealthy(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    gpu_utilization: float | None = None,
    disk_used_percent: float | None = None,
    consecutive_failures: int = 0,
    source: str = "",
) -> bool:
    """Emit NODE_UNHEALTHY event when a node is detected as unhealthy."""
    return await _emit_data_event(
        DataEventType.NODE_UNHEALTHY,
        {
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "gpu_utilization": gpu_utilization,
            "disk_used_percent": disk_used_percent,
            "consecutive_failures": consecutive_failures,
        },
        source=source or "event_emitters",
        log_message=f"Emitted node_unhealthy for {node_id}: {reason}",
        log_level="warning",
    )


async def emit_health_check_passed(
    node_id: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    latency_ms: float | None = None,
    source: str = "",
) -> bool:
    """Emit HEALTH_CHECK_PASSED event after successful health check."""
    return await _emit_data_event(
        DataEventType.HEALTH_CHECK_PASSED,
        {
            "node_id": node_id,
            "node_ip": node_ip,
            "check_type": check_type,
            "latency_ms": latency_ms,
        },
        source=source or "event_emitters",
        log_message=f"Emitted health_check_passed for {node_id}",
    )


async def emit_health_check_failed(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    error: str = "",
    source: str = "",
) -> bool:
    """Emit HEALTH_CHECK_FAILED event after failed health check."""
    return await _emit_data_event(
        DataEventType.HEALTH_CHECK_FAILED,
        {
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "check_type": check_type,
            "error": error,
        },
        source=source or "event_emitters",
        log_message=f"Emitted health_check_failed for {node_id}: {reason}",
        log_level="warning",
    )


async def emit_p2p_cluster_healthy(
    healthy_nodes: int,
    node_count: int,
    *,
    source: str = "",
) -> bool:
    """Emit P2P_CLUSTER_HEALTHY event when cluster becomes healthy."""
    return await _emit_data_event(
        DataEventType.P2P_CLUSTER_HEALTHY,
        {"healthy": True, "healthy_nodes": healthy_nodes, "node_count": node_count},
        source=source or "event_emitters",
        log_message=f"Emitted p2p_cluster_healthy: {healthy_nodes}/{node_count} nodes",
        log_level="info",
    )


async def emit_p2p_cluster_unhealthy(
    healthy_nodes: int,
    node_count: int,
    *,
    alerts: list[str] | None = None,
    source: str = "",
) -> bool:
    """Emit P2P_CLUSTER_UNHEALTHY event when cluster becomes unhealthy."""
    return await _emit_data_event(
        DataEventType.P2P_CLUSTER_UNHEALTHY,
        {
            "healthy": False,
            "healthy_nodes": healthy_nodes,
            "node_count": node_count,
            "alerts": alerts or [],
        },
        source=source or "event_emitters",
        log_message=f"Emitted p2p_cluster_unhealthy: {healthy_nodes}/{node_count} nodes",
        log_level="warning",
    )


async def emit_split_brain_detected(
    leaders_seen: list[str],
    *,
    severity: str = "warning",
    voter_count: int = 0,
    resolution_action: str = "step_down",
    source: str = "",
) -> bool:
    """Emit SPLIT_BRAIN_DETECTED event when multiple leaders are detected.

    December 2025: Critical for cluster coordination - indicates P2P split-brain
    condition where multiple nodes believe they are the leader. This triggers:
    - AlertManager: Send critical alert
    - UnifiedHealthManager: Track cluster degradation
    - LeadershipCoordinator: Initiate resolution

    Args:
        leaders_seen: List of node IDs claiming leadership
        severity: "warning" (2 leaders) or "critical" (3+ leaders)
        voter_count: Number of voter nodes in quorum
        resolution_action: Action taken (step_down, force_election, wait)
        source: Source component emitting the event
    """
    return await _emit_data_event(
        DataEventType.SPLIT_BRAIN_DETECTED,
        {
            "leaders_seen": leaders_seen,
            "leader_count": len(leaders_seen),
            "severity": severity,
            "voter_count": voter_count,
            "resolution_action": resolution_action,
        },
        source=source or "event_emitters",
        log_message=f"Emitted split_brain_detected: {len(leaders_seen)} leaders ({severity})",
        log_level="error" if severity == "critical" else "warning",
    )


async def emit_p2p_node_dead(
    node_id: str,
    *,
    reason: str = "timeout",
    last_seen: float | None = None,
    offline_duration_seconds: float = 0.0,
    source: str = "",
) -> bool:
    """Emit P2P_NODE_DEAD event when a node is confirmed dead.

    December 2025: Distinct from HOST_OFFLINE - this indicates a node that
    has been confirmed dead (not just temporarily offline) and requires
    work reassignment. Subscribers use this for:
    - SelfplayScheduler: Mark node as unavailable for selfplay allocation
    - UnifiedQueuePopulator: Reassign jobs from dead node
    """
    return await _emit_data_event(
        DataEventType.P2P_NODE_DEAD,
        {
            "node_id": node_id,
            "reason": reason,
            "last_seen": last_seen or time.time(),
            "offline_duration_seconds": offline_duration_seconds,
        },
        source=source or "event_emitters",
        log_message=f"Emitted p2p_node_dead for {node_id}: {reason}",
        log_level="warning",
    )


# =============================================================================
# Replication Repair Events (December 2025)
# =============================================================================


async def emit_repair_completed(
    game_id: str,
    source_nodes: list[str],
    target_nodes: list[str],
    duration_seconds: float,
    new_replica_count: int,
    source: str = "",
    **metadata,
) -> bool:
    """Emit REPAIR_COMPLETED event when a replication repair succeeds.

    December 2025: Wired into unified_replication_daemon.py for pipeline
    coordination of successful repair operations.

    Args:
        game_id: ID of the game that was repaired
        source_nodes: Nodes that provided the data
        target_nodes: Nodes that received the data
        duration_seconds: How long the repair took
        new_replica_count: Current replica count after repair
        source: Event source identifier
        **metadata: Additional event metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.REPAIR_COMPLETED,
        {
            "game_id": game_id,
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "duration_seconds": duration_seconds,
            "new_replica_count": new_replica_count,
            **metadata,
        },
        source=source or "unified_replication_daemon",
        log_message=f"Emitted repair_completed for {game_id}",
        log_level="info",
    )


async def emit_repair_failed(
    game_id: str,
    source_nodes: list[str],
    target_nodes: list[str],
    error: str,
    duration_seconds: float = 0.0,
    current_replica_count: int = 0,
    source: str = "",
    **metadata,
) -> bool:
    """Emit REPAIR_FAILED event when a replication repair fails.

    December 2025: Wired into unified_replication_daemon.py for pipeline
    coordination of failed repair operations.

    Args:
        game_id: ID of the game that failed repair
        source_nodes: Nodes that were supposed to provide data
        target_nodes: Nodes that were supposed to receive data
        error: Error message describing failure
        duration_seconds: How long before failure
        current_replica_count: Current replica count (unchanged)
        source: Event source identifier
        **metadata: Additional event metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.REPAIR_FAILED,
        {
            "game_id": game_id,
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "error": error,
            "duration_seconds": duration_seconds,
            "current_replica_count": current_replica_count,
            **metadata,
        },
        source=source or "unified_replication_daemon",
        log_message=f"Emitted repair_failed for {game_id}: {error}",
        log_level="warning",
    )


async def emit_generic_event(
    event_type: DataEventType,
    payload: dict,
    source: str = "",
    log_message: str = "",
    log_level: str = "debug",
) -> bool:
    """Emit a generic event with custom payload.

    December 2025: Added for flexible event emission from daemons that
    need to emit events not covered by typed emitters.

    Args:
        event_type: The DataEventType to emit
        payload: Event payload dictionary
        source: Event source identifier
        log_message: Optional log message
        log_level: Logging level ("debug", "info", "warning")

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        event_type,
        payload,
        source=source or "generic",
        log_message=log_message or f"Emitted {event_type.name}",
        log_level=log_level,
    )


__all__ = [
    # Retry-enabled emission (December 2025)
    "CRITICAL_EVENT_TYPES",
    "CRITICAL_STAGE_EVENTS",
    "is_critical_event",
    # Backpressure events (December 2025)
    "emit_backpressure_activated",
    "emit_backpressure_released",
    # Cache events (December 2025)
    "emit_cache_invalidated",
    "emit_coordinator_health_degraded",
    "emit_coordinator_healthy",
    "emit_coordinator_heartbeat",
    "emit_coordinator_shutdown",
    "emit_coordinator_unhealthy",
    # Curriculum events (December 2025)
    "emit_curriculum_rebalanced",
    "emit_curriculum_updated",
    # Evaluation events
    "emit_evaluation_complete",
    "emit_handler_failed",
    "emit_handler_timeout",
    # Hyperparameter events (December 2025)
    "emit_hyperparameter_updated",
    "emit_host_offline",
    # Host/Node events (December 2025)
    "emit_host_online",
    "emit_model_corrupted",
    # New games events (December 2025)
    "emit_new_games",
    "emit_node_recovered",
    "emit_node_retired",
    "emit_node_suspect",
    "emit_node_unhealthy",
    # Generic event emission (December 2025)
    "emit_generic_event",
    # Cluster health events (December 2025)
    "emit_health_check_passed",
    "emit_health_check_failed",
    "emit_p2p_cluster_healthy",
    "emit_p2p_cluster_unhealthy",
    "emit_p2p_node_dead",
    # Optimization events (December 2025)
    "emit_optimization_triggered",
    # Metrics events (December 2025)
    "emit_plateau_detected",
    # Promotion events
    "emit_promotion_complete",
    "emit_promotion_complete_sync",
    # Quality events
    "emit_quality_updated",
    "emit_game_quality_score",
    "emit_regression_detected",
    # Replication repair events (December 2025)
    "emit_repair_completed",
    "emit_repair_failed",
    # Selfplay events
    "emit_selfplay_complete",
    # Sync events
    "emit_sync_complete",
    "emit_sync_failure_critical",
    "emit_task_abandoned",
    # Generic task events
    "emit_task_complete",
    "emit_task_orphaned",
    "emit_training_complete",
    "emit_training_complete_sync",
    "emit_training_rollback_completed",
    # Error Recovery & Resilience events (December 2025)
    "emit_training_rollback_needed",
    # Training events
    "emit_training_started",
    "emit_training_started_sync",
    "emit_training_triggered",
]

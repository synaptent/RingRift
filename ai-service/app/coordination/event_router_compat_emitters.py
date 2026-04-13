"""Compatibility emitters for ``event_router``.

These helpers keep legacy imports stable while the main router module stays
focused on routing and subscription behavior.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime
from typing import Any

from app.coordination.event_utils import make_config_key
from app.core.async_context import fire_and_forget

logger = logging.getLogger(__name__)


async def _publish(event_type: str, payload: dict[str, Any], source: str) -> Any:
    from app.coordination.event_router import publish

    return await publish(event_type=event_type, payload=payload, source=source)


def _get_runtime_loop() -> asyncio.AbstractEventLoop | None:
    from app.coordination.event_router import _get_router_runtime_loop

    return _get_router_runtime_loop()


def safe_emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    source: str = "unknown",
    log_on_failure: bool = True,
) -> bool:
    """Safely emit an event without raising exceptions.

    Wraps event emission in try-catch to prevent event failures from crashing
    the caller. Works in both sync and async contexts.

    This helper eliminates the repetitive try/except boilerplate:
        try:
            emit_something(...)
        except Exception as e:
            logger.debug(f"Event failed: {e}")

    Instead use:
        safe_emit_event("MY_EVENT", {"key": "value"}, source="my_component")

    Args:
        event_type: Event type string (e.g., "TRAINING_COMPLETED")
        payload: Event payload dict (default: empty dict)
        source: Source identifier for logging
        log_on_failure: Whether to log failures (default: True)

    Returns:
        True if event was emitted successfully, False otherwise

    December 2025: Added to reduce 960+ LOC of boilerplate across codebase.
    """
    if payload is None:
        payload = {}

    try:
        # Try to get running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Async context - fire and forget
            emit_coro = _publish(event_type=event_type, payload=payload, source=source)
            try:
                fire_and_forget(emit_coro, name=f"safe_emit_{event_type}")
            except Exception:
                emit_coro.close()
                raise
        else:
            # Sync context: schedule on known running router loop when possible.
            # This avoids cross-loop lock errors when called from worker threads.
            main_loop = _get_runtime_loop()

            if main_loop is not None:
                # Schedule onto a running loop from this sync thread.
                emit_coro = _publish(
                    event_type=event_type,
                    payload=payload,
                    source=source,
                )
                try:
                    future = asyncio.run_coroutine_threadsafe(emit_coro, main_loop)
                except Exception:
                    emit_coro.close()
                    raise

                # Ensure background failures are visible even if caller ignores return.
                def _log_emit_failure(done_future: concurrent.futures.Future) -> None:
                    try:
                        exc = done_future.exception()
                        if exc is not None and log_on_failure:
                            logger.warning(
                                f"[{source}] Event {event_type} emission failed: {exc}"
                            )
                    except (concurrent.futures.CancelledError, RuntimeError):
                        if log_on_failure:
                            logger.warning(
                                f"[{source}] Event {event_type} emission was cancelled"
                            )

                future.add_done_callback(_log_emit_failure)
            else:
                # No running loop anywhere — safe to create one
                asyncio.run(
                    _publish(event_type=event_type, payload=payload, source=source)
                )

        return True

    except (ImportError, RuntimeError, OSError, AttributeError) as e:
        if log_on_failure:
            logger.warning(f"[{source}] Event {event_type} emission failed: {e}")
        return False
    except Exception as e:
        # Catch-all for unexpected errors to ensure caller never crashes
        if log_on_failure:
            logger.warning(f"[{source}] Event {event_type} emission error: {e}")
        return False


def emit_event(
    event_type: str | Any,  # Can be string or DataEventType enum
    payload: dict[str, Any] | None = None,
    source: str = "unknown",
) -> bool:
    """Emit an event to the event system.

    Convenience wrapper around safe_emit_event that handles enum types.

    Args:
        event_type: Event type string or DataEventType enum
        payload: Event payload dict (default: empty dict)
        source: Source identifier for logging

    Returns:
        True if event was emitted successfully, False otherwise
    """
    # Convert enum to string if needed
    event_type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
    return safe_emit_event(event_type_str, payload, source)


async def emit_training_started(
    config_key: str,
    node_name: str = "",
    **extra_payload
) -> None:
    """Emit TRAINING_STARTED event to all systems."""
    await _publish(
        event_type="TRAINING_STARTED",
        payload={
            "config_key": config_key,
            "node_name": node_name,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="training",
    )


async def emit_training_completed(
    config_key: str,
    model_id: str,
    val_loss: float = 0.0,
    epochs: int = 0,
    **extra_payload
) -> None:
    """Emit TRAINING_COMPLETED event to all systems."""
    await _publish(
        event_type="TRAINING_COMPLETED",
        payload={
            "config_key": config_key,
            "model_id": model_id,
            "val_loss": val_loss,
            "epochs": epochs,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="training",
    )


async def emit_training_failed(
    config_key: str,
    error: str,
    **extra_payload
) -> None:
    """Emit TRAINING_FAILED event to all systems."""
    await _publish(
        event_type="TRAINING_FAILED",
        payload={
            "config_key": config_key,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="training",
    )


async def emit_evaluation_started(
    model_path: str,
    board_type: str,
    num_players: int,
    config_key: str | None = None,
    **extra_payload
) -> None:
    """Emit EVALUATION_STARTED event to all systems.

    December 30, 2025: Added for Gap #3 integration fix.
    Enables metrics tracking and coordination when evaluation begins.
    """
    if config_key is None:
        config_key = make_config_key(board_type, num_players)

    payload = {
        "model_path": model_path,
        "board_type": board_type,
        "num_players": num_players,
        "config_key": config_key,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(extra_payload)

    await _publish(
        event_type="EVALUATION_STARTED",
        payload=payload,
        source="evaluation",
    )


async def emit_evaluation_completed(
    model_id: str | None = None,
    elo: float | None = None,
    win_rate: float = 0.0,
    games_played: int = 0,
    # December 30, 2025: Added explicit parameters for evaluation_daemon compatibility
    model_path: str | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    opponent_results: dict | None = None,
    harness_results: dict | None = None,
    best_harness: str | None = None,
    best_elo: float | None = None,
    composite_participant_ids: list | None = None,
    is_multi_harness: bool = False,
    # December 30, 2025: Architecture for multi-architecture training support
    architecture: str | None = None,
    **extra_payload
) -> None:
    """Emit EVALUATION_COMPLETED event to all systems.

    December 30, 2025: Extended with multi-harness evaluation support and
    multi-architecture training tracking.
    - harness_results: Dict of harness_name -> {elo, win_rate, games_played, composite_participant_id}
    - composite_participant_ids: List of composite IDs for (model, harness) combinations
    - is_multi_harness: True if evaluated under multiple harnesses
    - architecture: Model architecture (v2, v3, v4, v5, v5_heavy, etc.)
    """
    # Use model_path as model_id if not provided
    effective_model_id = model_id or model_path or "unknown"
    # Use best_elo if available, otherwise passed elo
    effective_elo = best_elo if best_elo is not None else (elo or 0.0)

    payload = {
        "model_id": effective_model_id,
        "model_path": model_path,
        "elo": effective_elo,
        "win_rate": win_rate,
        "games_played": games_played,
        "timestamp": datetime.now().isoformat(),
    }

    # Add optional fields if provided
    if board_type is not None:
        payload["board_type"] = board_type
    if num_players is not None:
        payload["num_players"] = num_players
    # Feb 24, 2026: config_key is required by auto_promotion_daemon
    if board_type is not None and num_players is not None and "config_key" not in payload:
        payload["config_key"] = f"{board_type}_{num_players}p"
    if opponent_results is not None:
        payload["opponent_results"] = opponent_results
    if harness_results is not None:
        payload["harness_results"] = harness_results
    if best_harness is not None:
        payload["best_harness"] = best_harness
    if best_elo is not None:
        payload["best_elo"] = best_elo
    if composite_participant_ids is not None:
        payload["composite_participant_ids"] = composite_participant_ids
    if is_multi_harness:
        payload["is_multi_harness"] = is_multi_harness
    # December 30, 2025: Add architecture for multi-architecture support
    if architecture is not None:
        payload["architecture"] = architecture

    # Add any extra payload
    payload.update(extra_payload)

    await _publish(
        event_type="EVALUATION_COMPLETED",
        payload=payload,
        source="evaluation",
    )


async def emit_sync_completed(
    sync_type: str,
    files_synced: int = 0,
    bytes_transferred: int = 0,
    **extra_payload
) -> None:
    """Emit DATA_SYNC_COMPLETED event to all systems."""
    await _publish(
        event_type="DATA_SYNC_COMPLETED",
        payload={
            "sync_type": sync_type,
            "files_synced": files_synced,
            "bytes_transferred": bytes_transferred,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="sync",
    )


async def emit_model_promoted(
    model_id: str,
    tier: str = "production",
    elo: float = 0.0,
    **extra_payload
) -> None:
    """Emit MODEL_PROMOTED event to all systems."""
    await _publish(
        event_type="MODEL_PROMOTED",
        payload={
            "model_id": model_id,
            "tier": tier,
            "elo": elo,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="promotion",
    )


async def emit_selfplay_batch_completed(
    config_key: str,
    games_generated: int,
    duration_seconds: float = 0.0,
    **extra_payload
) -> None:
    """Emit SELFPLAY_BATCH_COMPLETE event to all systems."""
    await _publish(
        event_type="SELFPLAY_BATCH_COMPLETE",
        payload={
            "config_key": config_key,
            "games_generated": games_generated,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="selfplay",
    )


def emit_training_started_sync(
    config_key: str,
    node_name: str = "",
    **extra_payload
) -> None:
    """Sync version of emit_training_started for non-async contexts."""
    try:
        # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
        asyncio.get_running_loop()
        fire_and_forget(
            emit_training_started(config_key, node_name, **extra_payload),
            name=f"emit_training_started_{config_key}",
        )
    except RuntimeError:
        # No running loop - create one with asyncio.run()
        asyncio.run(emit_training_started(config_key, node_name, **extra_payload))


def emit_training_completed_sync(
    config_key: str,
    model_id: str,
    val_loss: float = 0.0,
    epochs: int = 0,
    **extra_payload
) -> None:
    """Sync version of emit_training_completed."""
    try:
        # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
        asyncio.get_running_loop()
        fire_and_forget(
            emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload),
            name=f"emit_training_completed_{config_key}",
        )
    except RuntimeError:
        # No running loop - create one with asyncio.run()
        asyncio.run(emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload))

"""Consolidated event emission helpers with optional logging.

Jan 4, 2026 - Phase 1 of Code Consolidation (Session 17.5).

Problem: 270+ emit_event() calls across coordination layer with 30+ files
having identical try/except wrapper patterns. 15+ files have logging+emit boilerplate.

Solution: Consolidated helper that combines event emission with optional logging,
reducing duplication and standardizing error handling.

Usage:
    from app.coordination.event_emission_helpers import safe_emit_event

    # Simple emission (same as safe_event_emitter.safe_emit_event)
    safe_emit_event("MY_EVENT", {"key": "value"})

    # With logging before emission
    safe_emit_event(
        "SYNC_STARTED",
        {"target": "node-1"},
        log_before="Starting sync to node-1",
    )

    # With logging after emission
    safe_emit_event(
        "SYNC_COMPLETED",
        {"files": 42},
        log_after="Sync completed successfully",
    )

    # Full logging context
    safe_emit_event(
        DataEventType.TRAINING_COMPLETED,
        {"model": "hex8_2p"},
        log_before="Finishing training job",
        log_after="Training completion event emitted",
        context="training_daemon",
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    log_before: str | None = None,
    log_after: str | None = None,
    context: str = "event_emission",
    source: str = "module",
    log_level: int = logging.INFO,
) -> bool:
    """Consolidated event emission with optional logging.

    Combines event emission with optional before/after logging messages,
    reducing boilerplate across coordination modules.

    Args:
        event_type: Event type string or DataEventType enum value
        payload: Optional event payload dict
        log_before: Optional message to log before emission (at log_level)
        log_after: Optional message to log after successful emission (at log_level)
        context: Context string for error messages (default: "event_emission")
        source: Source identifier for the event (default: "module")
        log_level: Logging level for before/after messages (default: INFO)

    Returns:
        True if event was scheduled successfully, False otherwise

    Example:
        >>> safe_emit_event(
        ...     "SYNC_COMPLETED",
        ...     {"files": 42, "duration": 5.3},
        ...     log_before="Completing sync operation",
        ...     context="auto_sync",
        ... )
        True
    """
    # Convert enum to string if needed
    event_type_str = (
        event_type.value if hasattr(event_type, "value") else str(event_type)
    )

    # Log before emission
    if log_before:
        logger.log(log_level, f"[{context}] {log_before}")

    try:
        # Lazy imports to avoid circular dependencies
        # Jan 5, 2026: Use EventRouter instead of EventBus directly.
        # EventRouter properly handles string event types and converts to DataEventType enum.
        # Previously used EventBus which caused "'str' object has no attribute 'value'" error
        # when DataEvent was created with string instead of DataEventType enum.
        from app.coordination.event_router import get_router

        router = get_router()
        if router is None:
            logger.debug(f"[{context}] Event router unavailable for {event_type_str}")
            return False

        # router.publish() is async - schedule it properly
        try:
            loop = asyncio.get_running_loop()
            # Schedule as fire-and-forget task
            # EventRouter.publish() handles string→enum conversion internally
            task = loop.create_task(
                router.publish(event_type_str, payload or {}, source)
            )
            # Add error callback to log failures without crashing
            task.add_done_callback(
                lambda t: (
                    logger.debug(
                        f"[{context}] Event {event_type_str} publish failed: {t.exception()}"
                    )
                    if t.exception()
                    else None
                )
            )

            # Log after successful scheduling
            if log_after:
                logger.log(log_level, f"[{context}] {log_after}")

            return True

        except RuntimeError:
            # No running event loop - can't schedule async publish
            logger.debug(
                f"[{context}] No event loop, {event_type_str} event not published"
            )
            return False

    except (AttributeError, ImportError, TypeError) as e:
        # AttributeError - router missing attribute
        # ImportError - module unavailable during shutdown
        # TypeError - wrong signature
        logger.debug(f"[{context}] Event emission failed for {event_type_str}: {e}")
        return False


async def safe_emit_event_async(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    log_before: str | None = None,
    log_after: str | None = None,
    context: str = "event_emission",
    source: str = "module",
    log_level: int = logging.INFO,
) -> bool:
    """Async version of safe_emit_event.

    For use in async contexts where you want to await the emission result.

    Args:
        event_type: Event type string or DataEventType enum value
        payload: Optional event payload dict
        log_before: Optional message to log before emission
        log_after: Optional message to log after successful emission
        context: Context string for error messages
        source: Source identifier for the event
        log_level: Logging level for before/after messages

    Returns:
        True if event was emitted successfully, False otherwise
    """
    try:
        return await asyncio.to_thread(
            safe_emit_event,
            event_type,
            payload,
            log_before=log_before,
            log_after=log_after,
            context=context,
            source=source,
            log_level=log_level,
        )
    except RuntimeError:
        # No event loop available - fall back to sync
        return safe_emit_event(
            event_type,
            payload,
            log_before=log_before,
            log_after=log_after,
            context=context,
            source=source,
            log_level=log_level,
        )


def emit_with_logging(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    log_message: str | None = None,
    context: str = "event_emission",
    source: str = "module",
) -> bool:
    """Convenience wrapper that logs the event type being emitted.

    Simpler alternative when you just want to log what event is being emitted.

    Args:
        event_type: Event type string or DataEventType enum value
        payload: Optional event payload dict
        log_message: Custom log message (default: "Emitting {event_type}")
        context: Context string for error messages
        source: Source identifier for the event

    Returns:
        True if event was scheduled successfully, False otherwise

    Example:
        >>> emit_with_logging(
        ...     "TRAINING_STARTED",
        ...     {"config_key": "hex8_2p"},
        ...     context="training_trigger",
        ... )
        # Logs: [training_trigger] Emitting TRAINING_STARTED
        True
    """
    event_type_str = (
        event_type.value if hasattr(event_type, "value") else str(event_type)
    )
    default_message = f"Emitting {event_type_str}"
    return safe_emit_event(
        event_type,
        payload,
        log_before=log_message or default_message,
        context=context,
        source=source,
    )


__all__ = [
    "safe_emit_event",
    "safe_emit_event_async",
    "emit_with_logging",
]

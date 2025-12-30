#!/usr/bin/env python3
"""Unified event handler decorator for coordination layer.

This module provides a decorator that standardizes event handling patterns,
reducing boilerplate and ensuring consistent extraction, validation, and
error handling across all event handlers.

Usage:
    from app.coordination.event_handler_decorator import event_handler, EventContext

    class MyDaemon:
        @event_handler(required=["config_key"], optional=["model_path", "elo"])
        def _on_training_complete(
            self, ctx: EventContext, config_key: str, model_path: str | None, elo: float
        ) -> None:
            # Handler receives validated, typed parameters
            # ctx provides access to raw payload, metadata, and helpers
            pass

        @event_handler(extract="training_completed")
        async def _on_training_done(self, ctx: EventContext, data: TrainingEventData) -> None:
            # Uses batch extraction for common event types
            if data.is_valid:
                await self.process(data)

December 2025: Created as part of event handler pattern standardization.
Consolidates 5 patterns across 70+ handlers into 1 unified approach.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ParamSpec,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)

# Type variables for decorator
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Event Context - Provides access to event data and helpers
# =============================================================================


@dataclass
class EventContext:
    """Context object passed to event handlers with extracted data and helpers.

    Provides a standardized interface for accessing event payload data,
    metadata, and utility methods.

    Attributes:
        raw_event: The original event object/dict as received
        payload: Normalized payload (dict), handles .payload/.metadata/dict
        event_type: Event type string if available
        source: Event source if available
        timestamp: Event timestamp (defaults to now)
        extraction_errors: List of field extraction errors (non-fatal)

    Example:
        @event_handler(required=["config_key"])
        def _on_event(self, ctx: EventContext, config_key: str) -> None:
            # Access raw payload for custom fields
            custom_field = ctx.get("custom_field", "default")

            # Check if extraction had issues
            if ctx.extraction_errors:
                logger.warning(f"Extraction issues: {ctx.extraction_errors}")
    """

    raw_event: Any
    payload: dict[str, Any]
    event_type: str = ""
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    extraction_errors: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field from payload with default.

        Args:
            key: Field name to extract
            default: Default value if missing

        Returns:
            Field value or default
        """
        return self.payload.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer field with type coercion.

        Args:
            key: Field name to extract
            default: Default value if missing or invalid

        Returns:
            Integer value or default
        """
        value = self.payload.get(key, default)
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float field with type coercion.

        Args:
            key: Field name to extract
            default: Default value if missing or invalid

        Returns:
            Float value or default
        """
        value = self.payload.get(key, default)
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean field with type coercion.

        Args:
            key: Field name to extract
            default: Default value if missing

        Returns:
            Boolean value or default
        """
        value = self.payload.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value) if value is not None else default

    def get_list(self, key: str, default: list | None = None) -> list:
        """Get a list field with type validation.

        Args:
            key: Field name to extract
            default: Default value if missing

        Returns:
            List value or default (empty list if None)
        """
        value = self.payload.get(key)
        if value is None:
            return default if default is not None else []
        if isinstance(value, list):
            return value
        return default if default is not None else []

    def get_dict(self, key: str, default: dict | None = None) -> dict:
        """Get a dict field with type validation.

        Args:
            key: Field name to extract
            default: Default value if missing

        Returns:
            Dict value or default (empty dict if None)
        """
        value = self.payload.get(key)
        if value is None:
            return default if default is not None else {}
        if isinstance(value, dict):
            return value
        return default if default is not None else {}

    @property
    def config_key(self) -> str:
        """Get config_key using standard extraction logic."""
        return _extract_config_key(self.payload)

    @property
    def board_type(self) -> str:
        """Get board_type from payload or parsed config_key."""
        if "board_type" in self.payload:
            return str(self.payload["board_type"])
        # Try parsing from config_key
        config_key = self.config_key
        if config_key:
            from app.coordination.config_key import ConfigKey

            parsed = ConfigKey.parse(config_key)
            if parsed:
                return parsed.board_type
        return ""

    @property
    def num_players(self) -> int:
        """Get num_players from payload or parsed config_key."""
        if "num_players" in self.payload:
            return self.get_int("num_players", 0)
        # Try parsing from config_key
        config_key = self.config_key
        if config_key:
            from app.coordination.config_key import ConfigKey

            parsed = ConfigKey.parse(config_key)
            if parsed:
                return parsed.num_players
        return 0

    @property
    def model_path(self) -> str | None:
        """Get model_path using standard extraction logic."""
        return _extract_model_path(self.payload)


# =============================================================================
# Field Extraction Helpers (internal)
# =============================================================================


def _normalize_payload(event: Any) -> dict[str, Any]:
    """Normalize event to payload dict.

    Handles multiple event formats:
    - Event object with .payload attribute
    - Event object with .metadata attribute
    - Dict event (pass through)
    - None/other (return empty dict)

    Args:
        event: Raw event object

    Returns:
        Normalized payload as dict
    """
    if hasattr(event, "payload") and event.payload is not None:
        payload = event.payload
        if isinstance(payload, dict):
            return payload
    if hasattr(event, "metadata") and event.metadata is not None:
        metadata = event.metadata
        if isinstance(metadata, dict):
            return metadata
    if isinstance(event, dict):
        return event
    return {}


def _extract_config_key(payload: dict[str, Any]) -> str:
    """Extract config_key from payload using standard aliases.

    Checks: "config_key", "config"

    Args:
        payload: Event payload dict

    Returns:
        Config key string or empty string
    """
    return payload.get("config_key") or payload.get("config") or ""


def _extract_model_path(payload: dict[str, Any]) -> str | None:
    """Extract model path from payload using standard aliases.

    Checks: "model_path", "checkpoint_path", "path"

    Args:
        payload: Event payload dict

    Returns:
        Model path string or None
    """
    return (
        payload.get("model_path")
        or payload.get("checkpoint_path")
        or payload.get("path")
    )


def _extract_event_type(event: Any, payload: dict[str, Any]) -> str:
    """Extract event type from event or payload.

    Args:
        event: Raw event object
        payload: Normalized payload

    Returns:
        Event type string or empty string
    """
    # Try event object attributes
    if hasattr(event, "event_type"):
        return str(event.event_type)
    if hasattr(event, "type"):
        return str(event.type)
    # Try payload
    return payload.get("event_type") or payload.get("type") or ""


def _extract_source(event: Any, payload: dict[str, Any]) -> str:
    """Extract event source from event or payload.

    Args:
        event: Raw event object
        payload: Normalized payload

    Returns:
        Source string or empty string
    """
    if hasattr(event, "source"):
        return str(event.source)
    return payload.get("source") or ""


# =============================================================================
# Field Specification Types
# =============================================================================


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a field to extract from event payload.

    Attributes:
        name: Field name in payload (may include aliases)
        aliases: Alternative field names to check
        default: Default value if not found
        required: Whether field is required (raises if missing)
        type_coerce: Type to coerce value to (int, float, bool, str)
    """

    name: str
    aliases: tuple[str, ...] = ()
    default: Any = None
    required: bool = False
    type_coerce: type | None = None

    def extract(self, payload: dict[str, Any]) -> tuple[Any, str | None]:
        """Extract field value from payload.

        Args:
            payload: Event payload dict

        Returns:
            Tuple of (value, error_message). Error is None if successful.
        """
        # Try primary name
        value = payload.get(self.name)

        # Try aliases
        if value is None:
            for alias in self.aliases:
                value = payload.get(alias)
                if value is not None:
                    break

        # Handle missing required field
        if value is None:
            if self.required:
                return None, f"Required field '{self.name}' not found"
            return self.default, None

        # Type coercion
        if self.type_coerce is not None:
            try:
                if self.type_coerce is bool:
                    if isinstance(value, bool):
                        return value, None
                    if isinstance(value, str):
                        return value.lower() in ("true", "1", "yes", "on"), None
                    return bool(value), None
                return self.type_coerce(value), None
            except (ValueError, TypeError) as e:
                return self.default, f"Type coercion failed for '{self.name}': {e}"

        return value, None


# Standard field specs for common fields
STANDARD_FIELDS: dict[str, FieldSpec] = {
    "config_key": FieldSpec(
        name="config_key",
        aliases=("config",),
        default="",
        type_coerce=str,
    ),
    "model_path": FieldSpec(
        name="model_path",
        aliases=("checkpoint_path", "path"),
        default=None,
        type_coerce=str,
    ),
    "board_type": FieldSpec(
        name="board_type",
        default="",
        type_coerce=str,
    ),
    "num_players": FieldSpec(
        name="num_players",
        default=0,
        type_coerce=int,
    ),
    "elo": FieldSpec(
        name="elo",
        aliases=("rating", "elo_rating"),
        default=0.0,
        type_coerce=float,
    ),
    "games_played": FieldSpec(
        name="games_played",
        aliases=("games", "game_count", "games_count"),
        default=0,
        type_coerce=int,
    ),
    "win_rate": FieldSpec(
        name="win_rate",
        aliases=("winrate",),
        default=0.0,
        type_coerce=float,
    ),
    "epochs": FieldSpec(
        name="epochs",
        aliases=("epoch", "epochs_completed"),
        default=0,
        type_coerce=int,
    ),
    "loss": FieldSpec(
        name="loss",
        aliases=("final_loss", "train_loss"),
        default=0.0,
        type_coerce=float,
    ),
    "source": FieldSpec(
        name="source",
        aliases=("source_node", "node"),
        default="",
        type_coerce=str,
    ),
    "severity": FieldSpec(
        name="severity",
        default="unknown",
        type_coerce=str,
    ),
    "reason": FieldSpec(
        name="reason",
        default="",
        type_coerce=str,
    ),
}


# =============================================================================
# Batch Extraction Types
# =============================================================================


# Mapping of extraction type to function
BATCH_EXTRACTORS: dict[str, str] = {
    "training_completed": "extract_training_data",
    "evaluation_completed": "extract_evaluation_data",
    "selfplay_completed": "_extract_selfplay_data",
    "sync_completed": "_extract_sync_data",
}


def _get_batch_extractor(extract_type: str) -> Callable[[dict], Any] | None:
    """Get batch extraction function by type.

    Args:
        extract_type: Type name (e.g., "training_completed")

    Returns:
        Extraction function or None if not found
    """
    if extract_type not in BATCH_EXTRACTORS:
        return None

    func_name = BATCH_EXTRACTORS[extract_type]

    # Try event_utils first (has dataclass extractors)
    try:
        from app.coordination import event_utils

        if hasattr(event_utils, func_name):
            return getattr(event_utils, func_name)
    except ImportError:
        pass

    # Try event_handler_utils
    try:
        from app.coordination import event_handler_utils

        if hasattr(event_handler_utils, func_name):
            return getattr(event_handler_utils, func_name)
    except ImportError:
        pass

    return None


# =============================================================================
# Decorator Implementation
# =============================================================================


@dataclass
class HandlerMetrics:
    """Metrics for event handler invocations.

    Attributes:
        calls: Total number of handler invocations
        errors: Number of errors encountered
        total_duration_ms: Total processing time in milliseconds
        last_call_time: Timestamp of last invocation
    """

    calls: int = 0
    errors: int = 0
    total_duration_ms: float = 0.0
    last_call_time: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per call in milliseconds."""
        return self.total_duration_ms / self.calls if self.calls > 0 else 0.0


# Global metrics registry
_handler_metrics: dict[str, HandlerMetrics] = {}


def get_handler_metrics(handler_name: str) -> HandlerMetrics | None:
    """Get metrics for a specific handler.

    Args:
        handler_name: Fully qualified handler name

    Returns:
        HandlerMetrics or None if not tracked
    """
    return _handler_metrics.get(handler_name)


def get_all_handler_metrics() -> dict[str, HandlerMetrics]:
    """Get metrics for all tracked handlers.

    Returns:
        Dict mapping handler name to metrics
    """
    return dict(_handler_metrics)


def reset_handler_metrics() -> None:
    """Reset all handler metrics. Mainly for testing."""
    _handler_metrics.clear()


class event_handler:
    """Decorator for standardized event handling.

    This decorator:
    1. Normalizes event payload (handles .payload, .metadata, dict)
    2. Creates EventContext with helpers
    3. Extracts required/optional fields with type coercion
    4. Handles errors gracefully with logging
    5. Tracks metrics (calls, errors, duration)

    Args:
        required: List of required field names (handler fails if missing)
        optional: List of optional field names (uses defaults if missing)
        extract: Batch extraction type ("training_completed", "evaluation_completed", etc.)
        on_error: Error handling strategy ("log", "raise", "ignore")
        track_metrics: Whether to track handler metrics

    Examples:
        # Basic usage with required/optional fields
        @event_handler(required=["config_key"], optional=["model_path"])
        def _on_event(self, ctx: EventContext, config_key: str, model_path: str | None):
            pass

        # Batch extraction for common event types
        @event_handler(extract="training_completed")
        def _on_training(self, ctx: EventContext, data: TrainingEventData):
            if data.is_valid:
                self.process(data)

        # Async handler
        @event_handler(required=["config_key"])
        async def _on_async_event(self, ctx: EventContext, config_key: str):
            await self.process_async(config_key)
    """

    def __init__(
        self,
        required: list[str] | None = None,
        optional: list[str] | None = None,
        extract: str | None = None,
        on_error: str = "log",
        track_metrics: bool = True,
    ) -> None:
        self.required = required or []
        self.optional = optional or []
        self.extract = extract
        self.on_error = on_error
        self.track_metrics = track_metrics

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        """Wrap the handler function."""
        is_async = inspect.iscoroutinefunction(func)
        handler_name = f"{func.__module__}.{func.__qualname__}"

        # Initialize metrics
        if self.track_metrics and handler_name not in _handler_metrics:
            _handler_metrics[handler_name] = HandlerMetrics()

        if is_async:
            return self._wrap_async(func, handler_name)  # type: ignore
        return self._wrap_sync(func, handler_name)  # type: ignore

    def _wrap_sync(
        self, func: Callable[..., R], handler_name: str
    ) -> Callable[..., R | None]:
        """Wrap synchronous handler."""

        @functools.wraps(func)
        def wrapper(self_or_event: Any, *args: Any, **kwargs: Any) -> R | None:
            start_time = time.time()

            # Determine if method (has self) or function
            if args:
                # Method: self_or_event is self, first arg is event
                instance = self_or_event
                event = args[0]
                remaining_args = args[1:]
            else:
                # Function or method with just event
                instance = None
                event = self_or_event
                remaining_args = ()

            try:
                # Build context and extract fields
                ctx, extracted = self._build_context_and_extract(event)

                if extracted is None:
                    # Extraction failed for required field
                    return None

                # Call handler with context and extracted fields
                if instance is not None:
                    result = func(instance, ctx, *extracted, *remaining_args, **kwargs)
                else:
                    result = func(ctx, *extracted, *remaining_args, **kwargs)

                return result

            except (AttributeError, KeyError, TypeError, ValueError) as e:
                self._handle_error(handler_name, e, event)
                return None
            finally:
                if self.track_metrics:
                    duration_ms = (time.time() - start_time) * 1000
                    metrics = _handler_metrics.get(handler_name)
                    if metrics:
                        metrics.calls += 1
                        metrics.total_duration_ms += duration_ms
                        metrics.last_call_time = time.time()

        return wrapper

    def _wrap_async(
        self, func: Callable[..., Awaitable[R]], handler_name: str
    ) -> Callable[..., Awaitable[R | None]]:
        """Wrap asynchronous handler."""

        @functools.wraps(func)
        async def wrapper(self_or_event: Any, *args: Any, **kwargs: Any) -> R | None:
            start_time = time.time()

            # Determine if method (has self) or function
            if args:
                instance = self_or_event
                event = args[0]
                remaining_args = args[1:]
            else:
                instance = None
                event = self_or_event
                remaining_args = ()

            try:
                # Build context and extract fields
                ctx, extracted = self._build_context_and_extract(event)

                if extracted is None:
                    return None

                # Call handler
                if instance is not None:
                    result = await func(
                        instance, ctx, *extracted, *remaining_args, **kwargs
                    )
                else:
                    result = await func(ctx, *extracted, *remaining_args, **kwargs)

                return result

            except (AttributeError, KeyError, TypeError, ValueError) as e:
                self._handle_error(handler_name, e, event)
                return None
            finally:
                if self.track_metrics:
                    duration_ms = (time.time() - start_time) * 1000
                    metrics = _handler_metrics.get(handler_name)
                    if metrics:
                        metrics.calls += 1
                        metrics.total_duration_ms += duration_ms
                        metrics.last_call_time = time.time()

        return wrapper

    def _build_context_and_extract(
        self, event: Any
    ) -> tuple[EventContext, tuple[Any, ...] | None]:
        """Build context and extract fields.

        Args:
            event: Raw event

        Returns:
            Tuple of (context, extracted_values) or (context, None) if required field missing
        """
        payload = _normalize_payload(event)
        extraction_errors: list[str] = []

        ctx = EventContext(
            raw_event=event,
            payload=payload,
            event_type=_extract_event_type(event, payload),
            source=_extract_source(event, payload),
            extraction_errors=extraction_errors,
        )

        # Batch extraction mode
        if self.extract:
            extractor = _get_batch_extractor(self.extract)
            if extractor:
                try:
                    data = extractor(payload)
                    return ctx, (data,)
                except Exception as e:
                    extraction_errors.append(f"Batch extraction failed: {e}")
                    return ctx, None
            else:
                extraction_errors.append(f"Unknown extraction type: {self.extract}")
                return ctx, None

        # Field-by-field extraction
        extracted: list[Any] = []

        # Extract required fields
        for field_name in self.required:
            spec = STANDARD_FIELDS.get(
                field_name,
                FieldSpec(name=field_name, required=True),
            )
            # Override required flag for fields in required list
            spec = FieldSpec(
                name=spec.name,
                aliases=spec.aliases,
                default=spec.default,
                required=True,
                type_coerce=spec.type_coerce,
            )
            value, error = spec.extract(payload)
            if error:
                extraction_errors.append(error)
                if self.on_error != "ignore":
                    logger.warning(f"Event handler extraction error: {error}")
                return ctx, None
            extracted.append(value)

        # Extract optional fields
        for field_name in self.optional:
            spec = STANDARD_FIELDS.get(
                field_name,
                FieldSpec(name=field_name),
            )
            value, error = spec.extract(payload)
            if error:
                extraction_errors.append(error)
            extracted.append(value)

        return ctx, tuple(extracted)

    def _handle_error(self, handler_name: str, error: Exception, event: Any) -> None:
        """Handle errors during event processing.

        Args:
            handler_name: Handler function name
            error: Exception that occurred
            event: Original event
        """
        if self.track_metrics:
            metrics = _handler_metrics.get(handler_name)
            if metrics:
                metrics.errors += 1

        if self.on_error == "raise":
            raise
        elif self.on_error == "log":
            logger.error(
                f"[{handler_name}] Error processing event: {error}",
                exc_info=True,
            )
        # "ignore" - do nothing


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_with_context(event: Any) -> EventContext:
    """Create EventContext from event without using decorator.

    Useful for handlers that need manual control but want
    the standardized context object.

    Args:
        event: Raw event object

    Returns:
        EventContext with normalized payload and helpers

    Example:
        def _on_custom_event(self, event):
            ctx = extract_with_context(event)
            config_key = ctx.config_key
            custom_field = ctx.get("custom_field")
    """
    payload = _normalize_payload(event)
    return EventContext(
        raw_event=event,
        payload=payload,
        event_type=_extract_event_type(event, payload),
        source=_extract_source(event, payload),
    )


def normalize_event_payload(event: Any) -> dict[str, Any]:
    """Normalize event to payload dict.

    Standalone function for use without the decorator.
    Handles .payload, .metadata, and dict events.

    Args:
        event: Raw event object

    Returns:
        Payload as dict
    """
    return _normalize_payload(event)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main decorator
    "event_handler",
    # Context class
    "EventContext",
    # Field specification
    "FieldSpec",
    "STANDARD_FIELDS",
    # Metrics
    "HandlerMetrics",
    "get_handler_metrics",
    "get_all_handler_metrics",
    "reset_handler_metrics",
    # Convenience functions
    "extract_with_context",
    "normalize_event_payload",
]

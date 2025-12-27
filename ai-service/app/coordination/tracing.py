"""Distributed Tracing for RingRift AI (December 2025).

Provides trace_id propagation across the distributed system to enable:
- Request correlation across services
- End-to-end latency tracking
- Cross-process debugging
- Event chain visualization

Usage:
    from app.coordination.tracing import (
        get_trace_id,
        set_trace_id,
        new_trace,
        with_trace,
        TraceContext,
    )

    # Start a new trace
    with new_trace("training_pipeline") as ctx:
        print(f"Trace ID: {ctx.trace_id}")
        # All operations in this block share the trace_id

    # Propagate trace across services
    trace_id = get_trace_id()  # Get current trace
    # ... pass trace_id in request headers or event payloads ...

    # Continue trace in another service
    set_trace_id(received_trace_id)
    # ... operations now part of the same trace ...

Event Integration:
    # Events automatically include trace_id when available
    from app.coordination.event_router import DataEvent

    event = DataEvent(
        event_type=DataEventType.TRAINING_COMPLETED,
        payload={"model_id": "square8_v42"},
        source="trainer",
        # trace_id automatically injected from context
    )
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# Context variable for trace propagation
_current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "current_trace", default=None
)


@dataclass
class TraceSpan:
    """A span within a trace representing a single operation."""
    span_id: str
    name: str
    trace_id: str
    parent_span_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"
    tags: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "events": self.events,
        }


@dataclass
class TraceContext:
    """Context for a distributed trace.

    A trace represents an end-to-end operation that may span multiple
    services and processes. Each trace has a unique trace_id that is
    propagated across all related operations.
    """
    trace_id: str
    name: str
    start_time: float = 0.0
    tags: dict[str, Any] = field(default_factory=dict)
    spans: list[TraceSpan] = field(default_factory=list)
    _current_span: TraceSpan | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()

    @classmethod
    def new(cls, name: str = "unnamed", **tags) -> TraceContext:
        """Create a new trace context."""
        return cls(
            trace_id=generate_trace_id(),
            name=name,
            start_time=time.time(),
            tags=tags,
        )

    @classmethod
    def from_trace_id(cls, trace_id: str, name: str = "continued") -> TraceContext:
        """Continue a trace from an existing trace_id."""
        return cls(
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
        )

    def start_span(self, name: str, **tags) -> TraceSpan:
        """Start a new span within this trace."""
        span = TraceSpan(
            span_id=generate_span_id(),
            name=name,
            trace_id=self.trace_id,
            parent_span_id=self._current_span.span_id if self._current_span else None,
            start_time=time.time(),
            tags=tags,
        )
        self.spans.append(span)
        self._current_span = span
        return span

    def end_span(self, status: str = "ok", **tags) -> None:
        """End the current span."""
        if self._current_span:
            self._current_span.end_time = time.time()
            self._current_span.duration_ms = (
                self._current_span.end_time - self._current_span.start_time
            ) * 1000
            self._current_span.status = status
            self._current_span.tags.update(tags)

            # Find parent span
            parent_id = self._current_span.parent_span_id
            if parent_id:
                for span in self.spans:
                    if span.span_id == parent_id:
                        self._current_span = span
                        return
            self._current_span = None

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the current span."""
        if self._current_span:
            self._current_span.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes,
            })

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the current span or trace."""
        if self._current_span:
            self._current_span.tags[key] = value
        else:
            self.tags[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "tags": self.tags,
            "spans": [s.to_dict() for s in self.spans],
        }

    def get_duration_ms(self) -> float:
        """Get total trace duration in milliseconds."""
        return (time.time() - self.start_time) * 1000


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace-{uuid.uuid4().hex[:16]}"


def generate_span_id() -> str:
    """Generate a unique span ID."""
    return f"span-{uuid.uuid4().hex[:8]}"


def get_trace_id() -> str | None:
    """Get the current trace ID from context.

    Returns:
        Current trace_id or None if no trace is active
    """
    ctx = _current_trace.get()
    return ctx.trace_id if ctx else None


def get_trace_context() -> TraceContext | None:
    """Get the current trace context.

    Returns:
        Current TraceContext or None if no trace is active
    """
    return _current_trace.get()


def set_trace_id(trace_id: str, name: str = "continued") -> TraceContext:
    """Set the trace ID for the current context.

    Use this to continue a trace received from another service.

    Args:
        trace_id: Trace ID to use
        name: Name for this segment of the trace

    Returns:
        New TraceContext with the given trace_id
    """
    ctx = TraceContext.from_trace_id(trace_id, name)
    _current_trace.set(ctx)
    return ctx


@contextmanager
def new_trace(name: str = "unnamed", **tags: Any) -> Generator[TraceContext, None, None]:
    """Context manager to start a new trace.

    Args:
        name: Name of the trace
        **tags: Additional tags for the trace

    Yields:
        TraceContext for the new trace
    """
    ctx = TraceContext.new(name, **tags)
    token = _current_trace.set(ctx)
    try:
        yield ctx
    finally:
        _current_trace.reset(token)
        # Log trace summary
        logger.debug(
            f"[Trace {ctx.trace_id}] {ctx.name} completed in {ctx.get_duration_ms():.2f}ms "
            f"({len(ctx.spans)} spans)"
        )


@contextmanager
def with_trace(
    trace_id: str, name: str = "continued", **tags: Any
) -> Generator[TraceContext, None, None]:
    """Context manager to continue an existing trace.

    Args:
        trace_id: Trace ID to continue
        name: Name for this segment
        **tags: Additional tags

    Yields:
        TraceContext for the continued trace
    """
    ctx = TraceContext.from_trace_id(trace_id, name)
    ctx.tags.update(tags)
    token = _current_trace.set(ctx)
    try:
        yield ctx
    finally:
        _current_trace.reset(token)


@contextmanager
def span(name: str, **tags: Any) -> Generator[TraceSpan | None, None, None]:
    """Context manager to create a span within the current trace.

    Args:
        name: Name of the span
        **tags: Additional tags

    Yields:
        TraceSpan for the new span
    """
    ctx = _current_trace.get()
    if ctx is None:
        # No active trace, create a temporary one
        with new_trace(f"auto-{name}") as ctx:
            ctx.start_span(name, **tags)
            try:
                yield ctx._current_span
            except Exception as e:
                ctx.end_span(status="error", error=str(e))
                raise
            else:
                ctx.end_span(status="ok")
    else:
        ctx.start_span(name, **tags)
        try:
            yield ctx._current_span
        except Exception as e:
            ctx.end_span(status="error", error=str(e))
            raise
        else:
            ctx.end_span(status="ok")


def traced(name: str | None = None):
    """Decorator to automatically trace a function.

    Args:
        name: Optional name for the span (defaults to function name)

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        def wrapper(*args, **kwargs):
            with span(span_name):
                return func(*args, **kwargs)

        async def async_wrapper(*args, **kwargs):
            with span(span_name):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# Integration with Events
# =============================================================================

def inject_trace_into_event(event: Any) -> None:
    """Inject current trace_id into an event payload.

    Call this before publishing events to propagate traces.
    """
    trace_id = get_trace_id()
    if trace_id and hasattr(event, 'payload') and isinstance(event.payload, dict):
        event.payload['trace_id'] = trace_id


def extract_trace_from_event(event: Any) -> str | None:
    """Extract trace_id from an event payload.

    Call this when receiving events to continue traces.

    Returns:
        trace_id if present in event, None otherwise
    """
    if hasattr(event, 'payload') and isinstance(event.payload, dict):
        return event.payload.get('trace_id')
    return None


def inject_trace_into_headers(headers: dict[str, str]) -> dict[str, str]:
    """Inject current trace_id into HTTP headers.

    Args:
        headers: Existing headers dict

    Returns:
        Headers dict with trace_id added
    """
    trace_id = get_trace_id()
    if trace_id:
        headers['X-Trace-Id'] = trace_id
    ctx = get_trace_context()
    if ctx and ctx._current_span:
        headers['X-Span-Id'] = ctx._current_span.span_id
    return headers


def extract_trace_from_headers(headers: dict[str, str]) -> str | None:
    """Extract trace_id from HTTP headers.

    Args:
        headers: HTTP headers dict

    Returns:
        trace_id if present, None otherwise
    """
    return headers.get('X-Trace-Id') or headers.get('x-trace-id')


# =============================================================================
# Trace Collection
# =============================================================================

class TraceCollector:
    """Collects and stores traces for analysis.

    In production, this would export to a tracing backend like Jaeger or Zipkin.
    For now, it stores traces in memory and optionally logs them.
    """

    # Class-level lock for thread-safe lazy initialization (Dec 2025 fix)
    _init_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, max_traces: int = 1000, log_traces: bool = False):
        self.max_traces = max_traces
        self.log_traces = log_traces
        self._traces: list[dict[str, Any]] = []
        self._lock: threading.Lock | None = None  # Lazy init threading.Lock

    def collect(self, trace: TraceContext) -> None:
        """Collect a completed trace."""
        # Thread-safe lazy initialization (Dec 2025: fixed race condition)
        if self._lock is None:
            with TraceCollector._init_lock:
                if self._lock is None:  # Double-check pattern
                    self._lock = threading.Lock()

        trace_dict = trace.to_dict()

        with self._lock:
            self._traces.append(trace_dict)
            if len(self._traces) > self.max_traces:
                self._traces = self._traces[-self.max_traces:]

        if self.log_traces:
            logger.info(f"[TraceCollector] {trace.trace_id}: {trace.name} ({trace.get_duration_ms():.2f}ms)")

    def get_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent traces."""
        return self._traces[-limit:]

    def find_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Find a trace by ID."""
        for trace in reversed(self._traces):
            if trace.get('trace_id') == trace_id:
                return trace
        return None

    def get_slow_traces(self, threshold_ms: float = 1000.0) -> list[dict[str, Any]]:
        """Get traces exceeding duration threshold."""
        result = []
        for trace in self._traces:
            spans = trace.get('spans', [])
            if spans:
                total_ms = sum(s.get('duration_ms', 0) for s in spans)
                if total_ms >= threshold_ms:
                    result.append(trace)
        return result


# Global trace collector
_collector: TraceCollector | None = None


def get_trace_collector() -> TraceCollector:
    """Get the global trace collector."""
    global _collector
    if _collector is None:
        _collector = TraceCollector(log_traces=False)
    return _collector


def collect_trace(trace: TraceContext) -> None:
    """Collect a trace using the global collector."""
    get_trace_collector().collect(trace)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "TraceCollector",
    "TraceContext",
    # Classes
    "TraceSpan",
    "collect_trace",
    "extract_trace_from_event",
    "extract_trace_from_headers",
    "generate_span_id",
    # Functions
    "generate_trace_id",
    "get_trace_collector",
    "get_trace_context",
    "get_trace_id",
    "inject_trace_into_event",
    "inject_trace_into_headers",
    "new_trace",
    "set_trace_id",
    "span",
    "traced",
    "with_trace",
]

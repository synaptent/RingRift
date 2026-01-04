"""Distributed tracing infrastructure for RingRift AI service.

Jan 2026 - Phase 3.2: OpenTelemetry-compatible distributed tracing.

This module provides distributed tracing that:
- Integrates with existing logging trace context (TraceContext)
- Supports OpenTelemetry exporters when available
- Falls back to logging-based tracing when OTel is not installed
- Provides decorators for async and sync function tracing

Usage:
    from app.observability.tracing import get_tracer, trace_async, configure_tracing

    # Configure once at startup
    configure_tracing(service_name="ringrift-training")

    # Get a tracer for your module
    tracer = get_tracer(__name__)

    # Use decorator for tracing
    @trace_async("training_run")
    async def run_training(config_key: str):
        # Automatically traced
        ...

    # Or manually create spans
    async def my_function():
        with tracer.start_span("my_operation") as span:
            span.set_attribute("config_key", "hex8_2p")
            ...

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., http://localhost:4317)
    OTEL_SERVICE_NAME: Service name for traces (default: ringrift-ai)
    RINGRIFT_TRACING_ENABLED: Enable/disable tracing (default: true)
    RINGRIFT_TRACING_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

# ParamSpec requires Python 3.10+, use typing_extensions for 3.9 compatibility
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

# Integration with existing trace context
from app.core.logging_config import (
    TraceContext,
    get_trace_id,
    set_trace_id,
    get_node_id,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
P = ParamSpec("P")
T = TypeVar("T")


class TracingState(Enum):
    """State of the tracing system."""

    NOT_CONFIGURED = "not_configured"
    CONFIGURED_OTEL = "configured_otel"
    CONFIGURED_LOGGING = "configured_logging"
    DISABLED = "disabled"


@dataclass
class TraceConfig:
    """Configuration for distributed tracing.

    Attributes:
        service_name: Name of this service in traces
        enabled: Whether tracing is enabled
        sample_rate: Fraction of traces to sample (0.0-1.0)
        otlp_endpoint: OpenTelemetry OTLP endpoint
        log_spans: Whether to log span start/end events
        propagate_to_logging: Whether to set logging trace context
    """

    service_name: str = "ringrift-ai"
    enabled: bool = True
    sample_rate: float = 1.0
    otlp_endpoint: str | None = None
    log_spans: bool = True
    propagate_to_logging: bool = True

    @classmethod
    def from_env(cls) -> "TraceConfig":
        """Create config from environment variables."""
        return cls(
            service_name=os.environ.get("OTEL_SERVICE_NAME", "ringrift-ai"),
            enabled=os.environ.get("RINGRIFT_TRACING_ENABLED", "true").lower()
            == "true",
            sample_rate=float(
                os.environ.get("RINGRIFT_TRACING_SAMPLE_RATE", "1.0")
            ),
            otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            log_spans=os.environ.get("RINGRIFT_TRACING_LOG_SPANS", "true").lower()
            == "true",
            propagate_to_logging=True,
        )


@dataclass
class Span:
    """A trace span representing an operation.

    Spans can be nested and track timing, attributes, and events.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    _tracer: "Tracer | None" = field(default=None, repr=False)

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on this span."""
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> "Span":
        """Add an event to this span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )
        return self

    def set_status(self, status: str, description: str | None = None) -> "Span":
        """Set the status of this span."""
        self.status = status
        if description:
            self.attributes["status_description"] = description
        return self

    def end(self) -> None:
        """End this span and record its duration."""
        self.end_time = time.time()
        if self._tracer:
            self._tracer._record_span(self)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def __enter__(self) -> "Span":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is not None:
            self.set_status("ERROR", str(exc_val))
            self.set_attribute("exception.type", exc_type.__name__)
            self.set_attribute("exception.message", str(exc_val))
        self.end()


class Tracer:
    """A tracer that creates spans for distributed tracing.

    Integrates with OpenTelemetry when available, falls back to
    logging-based tracing otherwise.
    """

    def __init__(
        self,
        name: str,
        config: TraceConfig,
        otel_tracer: Any = None,
    ):
        self.name = name
        self.config = config
        self._otel_tracer = otel_tracer
        self._span_counter = 0

    def start_span(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Name of the operation
            parent: Optional parent span
            attributes: Initial span attributes

        Returns:
            A new Span that should be used as a context manager
        """
        # Get or create trace ID
        trace_id = get_trace_id()
        if trace_id is None:
            trace_id = set_trace_id()

        # Generate span ID
        self._span_counter += 1
        span_id = f"{trace_id}-{self._span_counter:04d}"

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            attributes=attributes or {},
            _tracer=self,
        )

        # Add standard attributes
        span.set_attribute("service.name", self.config.service_name)
        span.set_attribute("tracer.name", self.name)
        node_id = get_node_id()
        if node_id:
            span.set_attribute("node.id", node_id)

        if self.config.log_spans:
            logger.debug(
                f"[SPAN START] {name} trace_id={trace_id} span_id={span_id}",
                extra={"trace_id": trace_id, "span_name": name},
            )

        return span

    def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        if self.config.log_spans:
            logger.debug(
                f"[SPAN END] {span.name} duration={span.duration_ms:.2f}ms "
                f"status={span.status} trace_id={span.trace_id}",
                extra={
                    "trace_id": span.trace_id,
                    "span_name": span.name,
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                },
            )

        # If OTel is configured, we'd export here
        # For now, spans are recorded via logging


# Global state
_tracing_state: TracingState = TracingState.NOT_CONFIGURED
_config: TraceConfig | None = None
_tracers: dict[str, Tracer] = {}


def configure_tracing(
    service_name: str | None = None,
    config: TraceConfig | None = None,
) -> TracingState:
    """Configure the distributed tracing system.

    Should be called once at application startup.

    Args:
        service_name: Override service name
        config: Full configuration object (overrides service_name)

    Returns:
        The resulting tracing state
    """
    global _tracing_state, _config, _tracers

    if config is None:
        config = TraceConfig.from_env()

    if service_name:
        config.service_name = service_name

    _config = config

    if not config.enabled:
        _tracing_state = TracingState.DISABLED
        logger.info("Distributed tracing disabled")
        return _tracing_state

    # Try to initialize OpenTelemetry
    otel_available = False
    if config.otlp_endpoint:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Configure OTel
            resource = Resource.create({"service.name": config.service_name})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            otel_trace.set_tracer_provider(provider)
            otel_available = True
            logger.info(
                f"OpenTelemetry configured with endpoint: {config.otlp_endpoint}"
            )
        except ImportError:
            logger.info(
                "OpenTelemetry not installed, using logging-based tracing. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )
        except Exception as e:
            logger.warning(f"Failed to configure OpenTelemetry: {e}")

    if otel_available:
        _tracing_state = TracingState.CONFIGURED_OTEL
    else:
        _tracing_state = TracingState.CONFIGURED_LOGGING
        logger.info("Using logging-based distributed tracing")

    # Clear any existing tracers
    _tracers.clear()

    return _tracing_state


def get_tracer(name: str) -> Tracer:
    """Get a tracer for the given module/component name.

    If tracing is not configured, returns a tracer with default config.

    Args:
        name: Module or component name (typically __name__)

    Returns:
        A Tracer instance
    """
    global _config, _tracers

    if name in _tracers:
        return _tracers[name]

    if _config is None:
        _config = TraceConfig.from_env()

    tracer = Tracer(name=name, config=_config)
    _tracers[name] = tracer
    return tracer


def trace_async(
    operation_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace an async function.

    Args:
        operation_name: Custom operation name (defaults to function name)
        attributes: Attributes to add to the span

    Example:
        @trace_async("process_training_batch")
        async def process_batch(batch_id: int):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        name = operation_name or func.__name__
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_span(name, attributes=attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_sync(
    operation_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace a synchronous function.

    Args:
        operation_name: Custom operation name (defaults to function name)
        attributes: Attributes to add to the span

    Example:
        @trace_sync("compute_features")
        def compute_features(game_state):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        name = operation_name or func.__name__
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_span(name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status("ERROR", str(e))
                    raise

        return wrapper  # type: ignore

    return decorator


@contextmanager
def trace_operation(
    name: str,
    tracer_name: str = "ringrift",
    attributes: dict[str, Any] | None = None,
):
    """Context manager for tracing an operation.

    Useful when you can't use a decorator.

    Args:
        name: Operation name
        tracer_name: Tracer to use
        attributes: Span attributes

    Example:
        with trace_operation("database_query", attributes={"table": "games"}):
            result = db.execute(query)
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_span(name, attributes=attributes) as span:
        yield span


def get_tracing_state() -> TracingState:
    """Get the current tracing state."""
    return _tracing_state


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject trace context into HTTP headers for distributed tracing.

    Args:
        headers: Existing headers dict to add to

    Returns:
        Headers with trace context added
    """
    trace_id = get_trace_id()
    if trace_id:
        headers["X-Trace-ID"] = trace_id
        headers["traceparent"] = f"00-{trace_id}-0000000000000000-01"

    node_id = get_node_id()
    if node_id:
        headers["X-Node-ID"] = node_id

    return headers


def extract_trace_context(headers: dict[str, str]) -> str | None:
    """Extract trace context from HTTP headers.

    Args:
        headers: HTTP headers dict

    Returns:
        Trace ID if present, None otherwise
    """
    # Check our custom header first
    trace_id = headers.get("X-Trace-ID") or headers.get("x-trace-id")
    if trace_id:
        return trace_id

    # Fall back to W3C traceparent
    traceparent = headers.get("traceparent") or headers.get("Traceparent")
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 2:
            return parts[1]

    return None

"""OpenTelemetry Distributed Tracing for RingRift AI.

Per Section 4.3 of the action plan, this module provides:
- Distributed tracing for AI decision paths
- Multiple exporter support (Jaeger, OTLP, Console)
- Easy-to-use decorators for instrumenting functions
- Integration with existing Prometheus metrics
- Graceful degradation when OpenTelemetry not installed

Usage:
    from app.tracing import tracer, setup_tracing, traced

    # Setup once at application start
    setup_tracing(
        service_name="ringrift-ai",
        exporter="jaeger",  # or "otlp", "console", "none"
    )

    # Instrument functions with decorator
    @traced("get_ai_move")
    def get_ai_move(state, difficulty):
        span = get_current_span()
        span.set_attribute("difficulty", difficulty)
        # ... implementation

    # Or use context manager
    with tracer.start_as_current_span("operation") as span:
        span.set_attribute("key", "value")
        # ... implementation

Environment Variables:
    OTEL_EXPORTER: Exporter type (jaeger, otlp, console, none)
    OTEL_SERVICE_NAME: Service name for tracing
    OTEL_JAEGER_AGENT_HOST: Jaeger agent host (default: localhost)
    OTEL_JAEGER_AGENT_PORT: Jaeger agent port (default: 6831)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    OTEL_TRACING_ENABLED: Enable/disable tracing (default: true)

Note:
    The Jaeger Thrift exporter is deprecated. For new deployments, use OTLP
    exporter with Jaeger's native OTLP endpoint (port 4317):
        OTEL_EXPORTER=otlp
        OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

__all__ = [
    # Core functions
    "is_tracing_enabled",
    "get_tracer",
    "get_current_span",
    "setup_tracing",
    "shutdown_tracing",
    # Decorators
    "traced",
    "traced_async",
    # Context propagation
    "inject_trace_context",
    "extract_trace_context",
    "trace_context_from_headers",
    # Helper functions
    "add_ai_move_attributes",
    "add_training_attributes",
    # Classes
    "NoOpSpan",
    "NoOpTracer",
    # Module-level tracer
    "tracer",
]

# Type variable for generic function signatures
F = TypeVar('F', bound=Callable[..., Any])

# Try to import OpenTelemetry modules
try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider, Span
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    trace = None
    TracerProvider = None
    Resource = None
    SERVICE_NAME = None
    Span = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    Status = None
    StatusCode = None
    SpanKind = None
    TraceContextTextMapPropagator = None
    Context = None

# Optional exporters
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    HAS_JAEGER = True
except ImportError:
    HAS_JAEGER = False
    JaegerExporter = None

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False
    OTLPSpanExporter = None


# =============================================================================
# Global State
# =============================================================================

_tracer_provider: Optional[Any] = None
_tracer: Optional[Any] = None
_tracing_enabled: bool = False


# =============================================================================
# Configuration
# =============================================================================

def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _tracing_enabled and HAS_OPENTELEMETRY


def get_tracer(name: str = "ringrift.ai") -> Any:
    """Get the tracer instance.

    Args:
        name: Tracer name (usually module path)

    Returns:
        OpenTelemetry tracer or NoOpTracer if tracing disabled
    """
    global _tracer
    if _tracer is None:
        if HAS_OPENTELEMETRY:
            _tracer = trace.get_tracer(name)
        else:
            _tracer = NoOpTracer()
    return _tracer


def get_current_span() -> Any:
    """Get the current active span.

    Returns:
        Current span or NoOpSpan if no active span
    """
    if HAS_OPENTELEMETRY and _tracing_enabled:
        return trace.get_current_span()
    return NoOpSpan()


# =============================================================================
# Setup Functions
# =============================================================================

def setup_tracing(
    service_name: str = "ringrift-ai",
    exporter: str = "none",
    jaeger_host: Optional[str] = None,
    jaeger_port: Optional[int] = None,
    otlp_endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for tracing
        exporter: Exporter type ("jaeger", "otlp", "console", "none")
        jaeger_host: Jaeger agent host (default from env or localhost)
        jaeger_port: Jaeger agent port (default from env or 6831)
        otlp_endpoint: OTLP endpoint URL (default from env)
        sample_rate: Fraction of traces to sample (0.0 to 1.0)

    Returns:
        True if tracing was successfully set up
    """
    global _tracer_provider, _tracer, _tracing_enabled

    # Check environment for disable flag
    env_enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower()
    if env_enabled in ("false", "0", "no"):
        logger.info("Tracing disabled via OTEL_TRACING_ENABLED")
        _tracing_enabled = False
        return False

    if not HAS_OPENTELEMETRY:
        logger.warning("OpenTelemetry not installed, tracing disabled")
        _tracing_enabled = False
        return False

    # Get exporter from environment if not specified
    exporter = os.getenv("OTEL_EXPORTER", exporter)
    service_name = os.getenv("OTEL_SERVICE_NAME", service_name)

    if exporter == "none":
        logger.info("Tracing exporter set to 'none', tracing disabled")
        _tracing_enabled = False
        return False

    try:
        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: service_name})

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Add span processor based on exporter type
        span_processor = _create_span_processor(
            exporter=exporter,
            jaeger_host=jaeger_host,
            jaeger_port=jaeger_port,
            otlp_endpoint=otlp_endpoint,
        )

        if span_processor is not None:
            _tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(_tracer_provider)
            _tracer = trace.get_tracer(service_name)
            _tracing_enabled = True

            logger.info(
                f"OpenTelemetry tracing initialized: "
                f"service={service_name}, exporter={exporter}"
            )
            return True
        else:
            logger.warning(f"Failed to create span processor for exporter: {exporter}")
            _tracing_enabled = False
            return False

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        _tracing_enabled = False
        return False


def _create_span_processor(
    exporter: str,
    jaeger_host: Optional[str],
    jaeger_port: Optional[int],
    otlp_endpoint: Optional[str],
) -> Optional[Any]:
    """Create span processor based on exporter type."""
    if exporter == "console":
        return BatchSpanProcessor(ConsoleSpanExporter())

    elif exporter == "jaeger":
        if not HAS_JAEGER:
            logger.warning("Jaeger exporter not available, falling back to console")
            return BatchSpanProcessor(ConsoleSpanExporter())

        host = jaeger_host or os.getenv("OTEL_JAEGER_AGENT_HOST", "localhost")
        port = jaeger_port or int(os.getenv("OTEL_JAEGER_AGENT_PORT", "6831"))

        jaeger_exporter = JaegerExporter(
            agent_host_name=host,
            agent_port=port,
        )
        return BatchSpanProcessor(jaeger_exporter)

    elif exporter == "otlp":
        if not HAS_OTLP:
            logger.warning("OTLP exporter not available, falling back to console")
            return BatchSpanProcessor(ConsoleSpanExporter())

        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            logger.warning("OTLP endpoint not configured, falling back to console")
            return BatchSpanProcessor(ConsoleSpanExporter())

        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        return BatchSpanProcessor(otlp_exporter)

    else:
        logger.warning(f"Unknown exporter type: {exporter}")
        return None


def shutdown_tracing():
    """Shutdown tracing and flush any pending spans."""
    global _tracer_provider, _tracer, _tracing_enabled

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down tracer: {e}")

    _tracer_provider = None
    _tracer = None
    _tracing_enabled = False


# =============================================================================
# Decorators
# =============================================================================

def traced(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Args:
        span_name: Name for the span (default: function name)
        attributes: Static attributes to add to span
        kind: Span kind (default: INTERNAL)

    Returns:
        Decorated function

    Usage:
        @traced("get_ai_move", attributes={"component": "ai"})
        def get_ai_move(state, difficulty):
            span = get_current_span()
            span.set_attribute("difficulty", difficulty)
            ...
    """
    def decorator(func: F) -> F:
        name = span_name or func.__name__
        span_kind = kind if kind is not None else (SpanKind.INTERNAL if HAS_OPENTELEMETRY else None)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            tracer = get_tracer()
            with tracer.start_as_current_span(name, kind=span_kind) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def traced_async(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None,
) -> Callable[[F], F]:
    """Decorator to trace an async function.

    Same as @traced but for async functions.
    """
    def decorator(func: F) -> F:
        name = span_name or func.__name__
        span_kind = kind if kind is not None else (SpanKind.INTERNAL if HAS_OPENTELEMETRY else None)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            tracer = get_tracer()
            with tracer.start_as_current_span(name, kind=span_kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Context Propagation
# =============================================================================

def inject_trace_context(carrier: Dict[str, str]) -> None:
    """Inject trace context into carrier for propagation.

    Args:
        carrier: Dict to inject trace headers into
    """
    if not is_tracing_enabled():
        return

    propagator = TraceContextTextMapPropagator()
    propagator.inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> Optional[Any]:
    """Extract trace context from carrier.

    Args:
        carrier: Dict containing trace headers

    Returns:
        OpenTelemetry context or None
    """
    if not is_tracing_enabled():
        return None

    propagator = TraceContextTextMapPropagator()
    return propagator.extract(carrier)


@contextmanager
def trace_context_from_headers(headers: Dict[str, str]):
    """Context manager to set trace context from HTTP headers.

    Args:
        headers: HTTP headers containing trace context

    Usage:
        with trace_context_from_headers(request.headers):
            # Code here runs with extracted trace context
            ...
    """
    if not is_tracing_enabled():
        yield
        return

    context = extract_trace_context(dict(headers))
    if context:
        token = trace.context.attach(context)
        try:
            yield
        finally:
            trace.context.detach(token)
    else:
        yield


# =============================================================================
# No-Op Classes for Graceful Degradation
# =============================================================================

class NoOpSpan:
    """No-operation span when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class NoOpTracer:
    """No-operation tracer when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        """Context manager that yields a NoOpSpan."""
        yield NoOpSpan()


# =============================================================================
# Convenience Functions for Common Operations
# =============================================================================

def add_ai_move_attributes(
    span: Any,
    board_type: str,
    difficulty: int,
    engine_type: str,
    simulations: int = 0,
    depth: int = 0,
    time_ms: float = 0.0,
) -> None:
    """Add standard AI move attributes to a span.

    Args:
        span: The span to add attributes to
        board_type: Type of game board
        difficulty: AI difficulty level
        engine_type: AI engine (mcts, minimax, heuristic)
        simulations: MCTS simulations (if applicable)
        depth: Search depth (if applicable)
        time_ms: Time taken in milliseconds
    """
    span.set_attribute("ai.board_type", board_type)
    span.set_attribute("ai.difficulty", difficulty)
    span.set_attribute("ai.engine_type", engine_type)
    span.set_attribute("ai.simulations", simulations)
    span.set_attribute("ai.depth", depth)
    span.set_attribute("ai.time_ms", time_ms)


def add_training_attributes(
    span: Any,
    epoch: int,
    batch_size: int,
    learning_rate: float,
    loss: float = 0.0,
) -> None:
    """Add standard training attributes to a span.

    Args:
        span: The span to add attributes to
        epoch: Current epoch number
        batch_size: Training batch size
        learning_rate: Current learning rate
        loss: Current loss value
    """
    span.set_attribute("training.epoch", epoch)
    span.set_attribute("training.batch_size", batch_size)
    span.set_attribute("training.learning_rate", learning_rate)
    span.set_attribute("training.loss", loss)


# =============================================================================
# Module-level Tracer
# =============================================================================

# Default tracer instance
tracer = get_tracer("ringrift.ai")

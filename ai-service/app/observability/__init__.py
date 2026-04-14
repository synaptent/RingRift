"""Observability infrastructure for RingRift AI service.

Jan 2026 - Phase 3.2: Distributed tracing and metrics.

This module provides:
- OpenTelemetry-compatible distributed tracing
- Prometheus metrics export
- Integration with existing logging trace context
"""

from app.observability.tracing import (
    configure_tracing,
    get_tracer,
    trace_async,
    trace_sync,
    TraceConfig,
    TracingState,
)

__all__ = [
    "configure_tracing",
    "get_tracer",
    "trace_async",
    "trace_sync",
    "TraceConfig",
    "TracingState",
]


def __dir__() -> list[str]:
    """Expose the intended observability surface for discoverability."""

    return sorted(set(globals()) | set(__all__))

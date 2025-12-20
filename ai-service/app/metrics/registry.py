"""Unified metrics registry for safe metric creation.

This module consolidates the _safe_metric() pattern that was previously
duplicated across:
- app/metrics/orchestrator.py
- app/metrics/coordinator.py
- app/training/train.py
- app/ai/decision_log.py

Usage:
    from app.metrics.registry import safe_metric
    from prometheus_client import Counter, Gauge, Histogram

    MY_COUNTER = safe_metric(
        Counter,
        'my_counter_total',
        'Description of my counter',
        labelnames=['label1', 'label2'],
    )

The safe_metric function prevents duplicate metric registration errors
by returning existing metrics if they've already been registered.

December 2025: Centralized metric registration.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, TypeVar

from prometheus_client import (
    REGISTRY,
    Counter,
    Enum,
    Gauge,
    Histogram,
    Info,
    Summary,
)

logger = logging.getLogger(__name__)

# Type var for metric classes
M = TypeVar('M', Counter, Gauge, Histogram, Summary, Info, Enum)

# Thread-safe lock for metric registration
_registry_lock = threading.RLock()

# Track metrics we've registered for debugging
_registered_metrics: dict[str, str] = {}  # name -> type


def safe_metric(
    metric_class: type[M],
    name: str,
    doc: str,
    **kwargs: Any,
) -> M:
    """Create a metric or return existing one to avoid duplicate registration.

    This function is thread-safe and handles the common pattern of checking
    if a metric is already registered before creating a new one.

    Args:
        metric_class: The prometheus_client metric class (Counter, Gauge, etc.)
        name: The metric name (must be unique)
        doc: The metric documentation string
        **kwargs: Additional arguments passed to the metric constructor
            (e.g., labelnames, buckets for Histogram)

    Returns:
        The metric instance (either newly created or existing)

    Example:
        >>> from prometheus_client import Counter
        >>> counter = safe_metric(Counter, 'my_counter_total', 'My counter')
        >>> counter.inc()
    """
    with _registry_lock:
        # Check if metric already exists in registry
        if name in REGISTRY._names_to_collectors:
            existing = REGISTRY._names_to_collectors[name]
            return existing

        # Create new metric
        metric = metric_class(name, doc, **kwargs)
        _registered_metrics[name] = metric_class.__name__
        return metric


def get_metric(name: str) -> Any | None:
    """Get an existing metric by name.

    Args:
        name: The metric name

    Returns:
        The metric instance or None if not found
    """
    with _registry_lock:
        return REGISTRY._names_to_collectors.get(name)


def is_metric_registered(name: str) -> bool:
    """Check if a metric is already registered.

    Args:
        name: The metric name

    Returns:
        True if the metric exists, False otherwise
    """
    with _registry_lock:
        return name in REGISTRY._names_to_collectors


def list_registered_metrics() -> dict[str, str]:
    """List all metrics registered through safe_metric.

    Returns:
        Dict mapping metric names to their types
    """
    with _registry_lock:
        return dict(_registered_metrics)


def get_all_prometheus_metrics() -> dict[str, Any]:
    """Get all metrics from the Prometheus registry.

    Returns:
        Dict mapping metric names to metric instances
    """
    with _registry_lock:
        return dict(REGISTRY._names_to_collectors)


# Convenience aliases for common metric types
def safe_counter(name: str, doc: str, **kwargs: Any) -> Counter:
    """Create a safe Counter metric."""
    return safe_metric(Counter, name, doc, **kwargs)


def safe_gauge(name: str, doc: str, **kwargs: Any) -> Gauge:
    """Create a safe Gauge metric."""
    return safe_metric(Gauge, name, doc, **kwargs)


def safe_histogram(name: str, doc: str, **kwargs: Any) -> Histogram:
    """Create a safe Histogram metric."""
    return safe_metric(Histogram, name, doc, **kwargs)


def safe_summary(name: str, doc: str, **kwargs: Any) -> Summary:
    """Create a safe Summary metric."""
    return safe_metric(Summary, name, doc, **kwargs)


# For backwards compatibility, also export as _safe_metric
_safe_metric = safe_metric


__all__ = [
    '_safe_metric',  # Backwards compatibility
    'get_all_prometheus_metrics',
    # Introspection
    'get_metric',
    'is_metric_registered',
    'list_registered_metrics',
    # Type-specific helpers
    'safe_counter',
    'safe_gauge',
    'safe_histogram',
    # Main function
    'safe_metric',
    'safe_summary',
]

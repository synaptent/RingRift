"""Unified Metrics Publisher - Abstraction for multi-backend metrics publishing (December 2025).

This module provides a unified interface for publishing metrics to multiple backends:
- Prometheus (counters, gauges, histograms)
- Event bus (for distributed event-driven metrics)
- Logging (structured log output)
- StatsD (optional)

Instead of directly using prometheus_client everywhere, use MetricsPublisher for:
- Consistent metric naming
- Multi-backend publishing
- Automatic labeling
- Event correlation

Usage:
    from app.metrics.unified_publisher import (
        MetricsPublisher,
        get_metrics_publisher,
        publish_counter,
        publish_gauge,
        publish_histogram,
    )

    # Get publisher
    publisher = get_metrics_publisher()

    # Publish metrics
    publisher.counter("training_steps", 1, labels={"config": "square8_2p"})
    publisher.gauge("model_elo", 1650, labels={"config": "square8_2p"})
    publisher.histogram("inference_latency", 0.05, labels={"model": "v42"})

    # Or use convenience functions
    publish_counter("games_completed", 1, config="square8_2p")
    publish_gauge("active_models", 5)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricConfig:
    """Configuration for the metrics publisher."""

    # Backend enables
    enable_prometheus: bool = True
    enable_event_bus: bool = True
    enable_logging: bool = False  # Disabled by default (verbose)
    enable_statsd: bool = False

    # StatsD config
    statsd_host: str = "localhost"
    statsd_port: int = 8125

    # Metric prefix
    prefix: str = "ringrift"

    # Default labels applied to all metrics
    default_labels: dict[str, str] = field(default_factory=dict)

    # Histogram buckets
    default_histogram_buckets: tuple = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def counter(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Increment a counter."""

    @abstractmethod
    def gauge(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Set a gauge value."""

    @abstractmethod
    def histogram(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Record a histogram observation."""


class PrometheusBackend(MetricsBackend):
    """Prometheus metrics backend."""

    def __init__(self, prefix: str, histogram_buckets: tuple):
        self.prefix = prefix
        self.histogram_buckets = histogram_buckets
        self._counters: dict[tuple[str, tuple[str, ...]], Any] = {}
        self._gauges: dict[tuple[str, tuple[str, ...]], Any] = {}
        self._histograms: dict[tuple[str, tuple[str, ...]], Any] = {}

    def _get_counter(self, name: str, labels: dict[str, str]) -> Any:
        """Get or create a counter."""
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        label_names = tuple(sorted(labels.keys()))

        key = (full_name, label_names)
        if key not in self._counters:
            try:
                from prometheus_client import Counter
                self._counters[key] = Counter(
                    full_name,
                    f"Counter for {name}",
                    labelnames=label_names,
                )
            except (ImportError, ValueError, TypeError) as e:
                logger.debug(f"Failed to create counter {name}: {e}")
                return None

        return self._counters.get(key)

    def _get_gauge(self, name: str, labels: dict[str, str]) -> Any:
        """Get or create a gauge."""
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        label_names = tuple(sorted(labels.keys()))

        key = (full_name, label_names)
        if key not in self._gauges:
            try:
                from prometheus_client import Gauge
                self._gauges[key] = Gauge(
                    full_name,
                    f"Gauge for {name}",
                    labelnames=label_names,
                )
            except (ImportError, ValueError, TypeError) as e:
                logger.debug(f"Failed to create gauge {name}: {e}")
                return None

        return self._gauges.get(key)

    def _get_histogram(self, name: str, labels: dict[str, str]) -> Any:
        """Get or create a histogram."""
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        label_names = tuple(sorted(labels.keys()))

        key = (full_name, label_names)
        if key not in self._histograms:
            try:
                from prometheus_client import Histogram
                self._histograms[key] = Histogram(
                    full_name,
                    f"Histogram for {name}",
                    labelnames=label_names,
                    buckets=self.histogram_buckets,
                )
            except (ImportError, ValueError, TypeError) as e:
                logger.debug(f"Failed to create histogram {name}: {e}")
                return None

        return self._histograms.get(key)

    def counter(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Increment a counter."""
        counter = self._get_counter(name, labels)
        if counter:
            try:
                counter.labels(**labels).inc(value)
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to increment counter {name}: {e}")

    def gauge(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Set a gauge value."""
        gauge = self._get_gauge(name, labels)
        if gauge:
            try:
                gauge.labels(**labels).set(value)
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to set gauge {name}: {e}")

    def histogram(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Record a histogram observation."""
        hist = self._get_histogram(name, labels)
        if hist:
            try:
                hist.labels(**labels).observe(value)
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to observe histogram {name}: {e}")


class EventBusBackend(MetricsBackend):
    """Event bus metrics backend - publishes metrics as events."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def _publish(self, metric_type: str, name: str, value: float, labels: dict[str, str]) -> None:
        """Publish metric as event."""
        try:
            from app.coordination.event_router import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.METRICS_UPDATED,
                payload={
                    "metric_type": metric_type,
                    "metric_name": f"{self.prefix}_{name}" if self.prefix else name,
                    "value": value,
                    "labels": labels,
                    "timestamp": time.time(),
                },
                source="metrics_publisher",
            )

            bus = get_event_bus()
            bus.publish_sync(event)

        except (ImportError, AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"Failed to publish metric event: {e}")

    def counter(self, name: str, value: float, labels: dict[str, str]) -> None:
        self._publish("counter", name, value, labels)

    def gauge(self, name: str, value: float, labels: dict[str, str]) -> None:
        self._publish("gauge", name, value, labels)

    def histogram(self, name: str, value: float, labels: dict[str, str]) -> None:
        self._publish("histogram", name, value, labels)


class LoggingBackend(MetricsBackend):
    """Logging backend - outputs metrics as structured logs."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self._logger = logging.getLogger("metrics")

    def _log(self, metric_type: str, name: str, value: float, labels: dict[str, str]) -> None:
        """Log metric."""
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        labels_str = ", ".join(f"{k}={v}" for k, v in labels.items())
        self._logger.info(f"[{metric_type}] {full_name}={value} ({labels_str})")

    def counter(self, name: str, value: float, labels: dict[str, str]) -> None:
        self._log("counter", name, value, labels)

    def gauge(self, name: str, value: float, labels: dict[str, str]) -> None:
        self._log("gauge", name, value, labels)

    def histogram(self, name: str, value: float, labels: dict[str, str]) -> None:
        self._log("histogram", name, value, labels)


class MetricsPublisher:
    """Unified metrics publisher with multi-backend support.

    Publishes metrics to all enabled backends:
    - Prometheus (default)
    - Event bus (for distributed systems)
    - Logging (for debugging)
    - StatsD (optional)
    """

    def __init__(self, config: MetricConfig | None = None):
        """Initialize metrics publisher.

        Args:
            config: Configuration (default: MetricConfig())
        """
        self.config = config or MetricConfig()
        self._backends: list[MetricsBackend] = []

        # Initialize backends
        if self.config.enable_prometheus:
            try:
                self._backends.append(PrometheusBackend(
                    prefix=self.config.prefix,
                    histogram_buckets=self.config.default_histogram_buckets,
                ))
            except (ImportError, ValueError, TypeError) as e:
                logger.debug(f"Prometheus backend unavailable: {e}")

        if self.config.enable_event_bus:
            self._backends.append(EventBusBackend(prefix=self.config.prefix))

        if self.config.enable_logging:
            self._backends.append(LoggingBackend(prefix=self.config.prefix))

        logger.debug(f"[MetricsPublisher] Initialized with {len(self._backends)} backends")

    def _merge_labels(self, labels: dict[str, str] | None = None) -> dict[str, str]:
        """Merge provided labels with default labels."""
        merged = dict(self.config.default_labels)
        if labels:
            merged.update(labels)
        return merged

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by (default: 1)
            labels: Metric labels
        """
        merged_labels = self._merge_labels(labels)
        for backend in self._backends:
            try:
                backend.counter(name, value, merged_labels)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Backend failed for counter {name}: {e}")

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        merged_labels = self._merge_labels(labels)
        for backend in self._backends:
            try:
                backend.gauge(name, value, merged_labels)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Backend failed for gauge {name}: {e}")

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        merged_labels = self._merge_labels(labels)
        for backend in self._backends:
            try:
                backend.histogram(name, value, merged_labels)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Backend failed for histogram {name}: {e}")

    def timer(self, name: str, labels: dict[str, str] | None = None) -> MetricTimer:
        """Create a context manager for timing operations.

        Args:
            name: Metric name for the histogram
            labels: Metric labels

        Returns:
            MetricTimer context manager
        """
        return MetricTimer(self, name, labels)


class MetricTimer:
    """Context manager for timing operations and recording to histogram."""

    def __init__(
        self,
        publisher: MetricsPublisher,
        name: str,
        labels: dict[str, str] | None = None,
    ):
        self.publisher = publisher
        self.name = name
        self.labels = labels
        self._start_time: float = 0.0

    def __enter__(self) -> MetricTimer:
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.time() - self._start_time
        self.publisher.histogram(self.name, elapsed, self.labels)


# Singleton instance
_publisher: MetricsPublisher | None = None


def get_metrics_publisher(config: MetricConfig | None = None) -> MetricsPublisher:
    """Get the global metrics publisher singleton.

    Args:
        config: Configuration (only used on first call)

    Returns:
        MetricsPublisher instance
    """
    global _publisher
    if _publisher is None:
        _publisher = MetricsPublisher(config)
    return _publisher


def reset_metrics_publisher() -> None:
    """Reset the publisher singleton (for testing)."""
    global _publisher
    _publisher = None


# =============================================================================
# Convenience functions for common metric operations
# =============================================================================

def publish_counter(name: str, value: float = 1.0, **labels) -> None:
    """Publish a counter metric.

    Args:
        name: Metric name
        value: Value to increment by
        **labels: Metric labels as keyword arguments
    """
    get_metrics_publisher().counter(name, value, labels if labels else None)


def publish_gauge(name: str, value: float, **labels) -> None:
    """Publish a gauge metric.

    Args:
        name: Metric name
        value: Gauge value
        **labels: Metric labels as keyword arguments
    """
    get_metrics_publisher().gauge(name, value, labels if labels else None)


def publish_histogram(name: str, value: float, **labels) -> None:
    """Publish a histogram observation.

    Args:
        name: Metric name
        value: Observed value
        **labels: Metric labels as keyword arguments
    """
    get_metrics_publisher().histogram(name, value, labels if labels else None)


def time_operation(name: str, **labels) -> MetricTimer:
    """Time an operation and record to histogram.

    Usage:
        with time_operation("inference_latency", model="v42"):
            result = model.infer(input)

    Args:
        name: Metric name
        **labels: Metric labels

    Returns:
        MetricTimer context manager
    """
    return get_metrics_publisher().timer(name, labels if labels else None)


__all__ = [
    "EventBusBackend",
    "LoggingBackend",
    "MetricConfig",
    "MetricTimer",
    "MetricType",
    "MetricsBackend",
    "MetricsPublisher",
    "PrometheusBackend",
    "get_metrics_publisher",
    # Convenience functions
    "publish_counter",
    "publish_gauge",
    "publish_histogram",
    "reset_metrics_publisher",
    "time_operation",
]

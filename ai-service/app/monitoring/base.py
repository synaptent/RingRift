"""Base classes for the unified monitoring framework.

This module provides abstract base classes that all monitors should inherit from,
ensuring a consistent interface across cluster, training, and data quality monitors.

Usage:
    from app.monitoring.base import HealthMonitor, HealthStatus, Alert

    class MyMonitor(HealthMonitor):
        def check_health(self) -> MonitoringResult:
            ...
        def should_alert(self) -> Optional[Alert]:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from app.config.thresholds import AlertLevel


class HealthStatus(str, Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    level: AlertLevel
    category: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    node: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "node": self.node,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Human-readable alert string."""
        parts = [f"[{self.level.value.upper()}]", self.message]
        if self.node:
            parts.insert(1, f"({self.node})")
        if self.metric_value is not None and self.threshold is not None:
            parts.append(f"[{self.metric_value:.1f}/{self.threshold:.1f}]")
        return " ".join(parts)


@dataclass
class MonitoringResult:
    """Result of a health check."""
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: dict[str, Any] = field(default_factory=dict)
    alerts: list[Alert] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def has_alerts(self) -> bool:
        """Check if there are any alerts."""
        return len(self.alerts) > 0

    @property
    def critical_alerts(self) -> list[Alert]:
        """Get only critical alerts."""
        return [a for a in self.alerts if a.level == AlertLevel.CRITICAL]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "alerts": [a.to_dict() for a in self.alerts],
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


class HealthMonitor(ABC):
    """Abstract base class for all health monitors.

    Concrete implementations should override:
    - check_health(): Perform the actual health check
    - get_name(): Return monitor name for logging

    Optional overrides:
    - should_alert(): Custom alert logic
    - format_report(): Custom report formatting
    """

    def __init__(self, name: str | None = None):
        """Initialize monitor.

        Args:
            name: Monitor name (default: class name)
        """
        self._name = name or self.__class__.__name__
        self._last_result: MonitoringResult | None = None
        self._last_check: datetime | None = None

    @property
    def name(self) -> str:
        """Get monitor name."""
        return self._name

    @property
    def last_result(self) -> MonitoringResult | None:
        """Get result of last health check."""
        return self._last_result

    @abstractmethod
    def check_health(self) -> MonitoringResult:
        """Perform health check and return result.

        Implementations should:
        1. Gather relevant metrics
        2. Compare against thresholds
        3. Generate alerts if needed
        4. Return MonitoringResult

        Returns:
            MonitoringResult with status, metrics, and any alerts
        """
        pass

    def should_alert(self) -> Alert | None:
        """Determine if an alert should be sent based on last check.

        Default implementation returns highest-severity alert from last check.
        Override for custom alert logic.

        Returns:
            Alert to send, or None if no alert needed
        """
        if not self._last_result or not self._last_result.alerts:
            return None

        # Return most severe alert
        severity_order = [AlertLevel.FATAL, AlertLevel.CRITICAL, AlertLevel.WARNING]
        for level in severity_order:
            for alert in self._last_result.alerts:
                if alert.level == level:
                    return alert
        return self._last_result.alerts[0] if self._last_result.alerts else None

    def format_report(self) -> str:
        """Format last result as human-readable report.

        Returns:
            Formatted report string
        """
        if not self._last_result:
            return f"{self.name}: No data"

        result = self._last_result
        lines = [
            f"=== {self.name} ===",
            f"Status: {result.status.value}",
            f"Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if result.metrics:
            lines.append("Metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")

        if result.alerts:
            lines.append(f"Alerts ({len(result.alerts)}):")
            for alert in result.alerts:
                lines.append(f"  {alert}")

        return "\n".join(lines)

    def run_check(self) -> MonitoringResult:
        """Run health check and update internal state.

        This is the main entry point for running checks.
        It handles timing, state updates, and error handling.

        Returns:
            MonitoringResult from the check
        """
        import time
        start = time.time()

        try:
            result = self.check_health()
        except Exception as e:
            result = MonitoringResult(
                status=HealthStatus.UNKNOWN,
                alerts=[Alert(
                    level=AlertLevel.CRITICAL,
                    category="monitor_error",
                    message=f"Health check failed: {e!s}",
                )],
            )

        result.duration_ms = (time.time() - start) * 1000
        self._last_result = result
        self._last_check = datetime.utcnow()

        return result


class CompositeMonitor(HealthMonitor):
    """Monitor that aggregates results from multiple sub-monitors.

    Useful for creating unified health endpoints that combine
    cluster, training, and data quality monitors.
    """

    def __init__(self, name: str = "CompositeMonitor"):
        super().__init__(name)
        self._monitors: list[HealthMonitor] = []

    def add_monitor(self, monitor: HealthMonitor) -> None:
        """Add a sub-monitor."""
        self._monitors.append(monitor)

    def remove_monitor(self, monitor: HealthMonitor) -> None:
        """Remove a sub-monitor."""
        self._monitors.remove(monitor)

    def check_health(self) -> MonitoringResult:
        """Run all sub-monitors and aggregate results."""
        all_metrics: dict[str, Any] = {}
        all_alerts: list[Alert] = []
        all_details: dict[str, Any] = {}
        worst_status = HealthStatus.HEALTHY

        for monitor in self._monitors:
            try:
                result = monitor.run_check()

                # Aggregate metrics with monitor name prefix
                for key, value in result.metrics.items():
                    all_metrics[f"{monitor.name}.{key}"] = value

                # Collect all alerts
                all_alerts.extend(result.alerts)

                # Store sub-monitor details
                all_details[monitor.name] = result.to_dict()

                # Update worst status
                if result.status == HealthStatus.UNHEALTHY:
                    worst_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.DEGRADED
                elif result.status == HealthStatus.UNKNOWN and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.UNKNOWN

            except Exception as e:
                all_alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="sub_monitor_error",
                    message=f"Sub-monitor {monitor.name} failed: {e!s}",
                ))

        return MonitoringResult(
            status=worst_status,
            metrics=all_metrics,
            alerts=all_alerts,
            details=all_details,
        )


# =============================================================================
# Monitor Registry (December 2025)
# =============================================================================

import logging
import threading

logger = logging.getLogger(__name__)


class MonitorRegistry:
    """Singleton registry for centralized monitor management.

    Provides a single point of access for all monitors in the system,
    enabling unified health checks and alert aggregation.

    Usage:
        from app.monitoring.base import MonitorRegistry, get_monitor_registry

        # Get registry singleton
        registry = get_monitor_registry()

        # Register a monitor
        registry.register(my_disk_monitor)
        registry.register(my_gpu_monitor, category="resources")

        # Run all health checks
        result = registry.check_all()

        # Get monitors by category
        resource_monitors = registry.get_monitors(category="resources")

        # Get all alerts
        alerts = registry.get_all_alerts()

    Benefits:
        - Centralized monitor discovery
        - Unified health endpoint
        - Category-based filtering
        - Event emission on health changes
    """

    _instance: Optional["MonitorRegistry"] = None
    _lock = threading.RLock()

    def __new__(cls) -> "MonitorRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._monitors: dict[str, HealthMonitor] = {}
        self._categories: dict[str, list[str]] = {}  # category -> monitor_names
        self._composite: CompositeMonitor = CompositeMonitor("GlobalHealth")
        self._last_overall_status: HealthStatus | None = None

    def register(
        self,
        monitor: HealthMonitor,
        category: str = "default",
    ) -> None:
        """Register a monitor.

        Args:
            monitor: HealthMonitor instance to register
            category: Category for grouping (e.g., "resources", "training", "cluster")
        """
        name = monitor.name
        with self._lock:
            self._monitors[name] = monitor
            self._composite.add_monitor(monitor)

            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)

            logger.info(f"Registered monitor: {name} (category: {category})")

    def unregister(self, monitor_name: str) -> HealthMonitor | None:
        """Unregister a monitor by name.

        Args:
            monitor_name: Name of monitor to remove

        Returns:
            The removed monitor or None if not found
        """
        with self._lock:
            monitor = self._monitors.pop(monitor_name, None)
            if monitor:
                self._composite.remove_monitor(monitor)
                for cat_monitors in self._categories.values():
                    if monitor_name in cat_monitors:
                        cat_monitors.remove(monitor_name)
            return monitor

    def get_monitor(self, name: str) -> HealthMonitor | None:
        """Get a monitor by name."""
        return self._monitors.get(name)

    def get_monitors(
        self,
        category: str | None = None,
    ) -> list[HealthMonitor]:
        """Get monitors, optionally filtered by category.

        Args:
            category: If provided, only return monitors in this category

        Returns:
            List of HealthMonitor instances
        """
        if category:
            names = self._categories.get(category, [])
            return [self._monitors[n] for n in names if n in self._monitors]
        return list(self._monitors.values())

    def get_categories(self) -> list[str]:
        """Get all registered categories."""
        return list(self._categories.keys())

    def check_all(self) -> MonitoringResult:
        """Run health checks on all registered monitors.

        Returns:
            Aggregated MonitoringResult from all monitors
        """
        result = self._composite.run_check()

        # Emit event if status changed
        if self._last_overall_status != result.status:
            self._emit_status_change(self._last_overall_status, result.status)
            self._last_overall_status = result.status

        return result

    def check_category(self, category: str) -> MonitoringResult:
        """Run health checks on monitors in a specific category.

        Args:
            category: Category to check

        Returns:
            Aggregated MonitoringResult for the category
        """
        monitors = self.get_monitors(category)
        temp_composite = CompositeMonitor(f"Category_{category}")
        for monitor in monitors:
            temp_composite.add_monitor(monitor)
        return temp_composite.run_check()

    def get_all_alerts(self) -> list[Alert]:
        """Get all current alerts from all monitors."""
        alerts = []
        for monitor in self._monitors.values():
            if monitor._last_result and monitor._last_result.alerts:
                alerts.extend(monitor._last_result.alerts)
        return alerts

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all monitors and their status.

        Returns:
            Dict with monitor statuses, counts, and categories
        """
        statuses = {}
        for name, monitor in self._monitors.items():
            if monitor._last_result:
                statuses[name] = {
                    "status": monitor._last_result.status.value,
                    "last_check": monitor._last_check.isoformat() if monitor._last_check else None,
                    "alert_count": len(monitor._last_result.alerts),
                }
            else:
                statuses[name] = {"status": "not_checked", "last_check": None}

        return {
            "total_monitors": len(self._monitors),
            "categories": {cat: len(names) for cat, names in self._categories.items()},
            "monitors": statuses,
        }

    def _emit_status_change(
        self,
        old_status: HealthStatus | None,
        new_status: HealthStatus,
    ) -> None:
        """Emit event on overall status change."""
        try:
            from app.distributed.data_events import DataEvent, DataEventType, get_event_bus

            bus = get_event_bus()
            event = DataEvent(
                event_type=DataEventType.HEALTH_ALERT if new_status != HealthStatus.HEALTHY else DataEventType.CLUSTER_STATUS_CHANGED,
                payload={
                    "old_status": old_status.value if old_status else None,
                    "new_status": new_status.value,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                source="monitor_registry",
            )
            # Fire and forget
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(bus.publish(event))
            except RuntimeError:
                pass

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to emit status change event: {e}")


# Module-level singleton access
_registry: MonitorRegistry | None = None


def get_monitor_registry() -> MonitorRegistry:
    """Get the singleton MonitorRegistry instance."""
    global _registry
    if _registry is None:
        _registry = MonitorRegistry()
    return _registry


def register_monitor(
    monitor: HealthMonitor,
    category: str = "default",
) -> None:
    """Convenience function to register a monitor.

    Args:
        monitor: HealthMonitor to register
        category: Category for grouping
    """
    get_monitor_registry().register(monitor, category)

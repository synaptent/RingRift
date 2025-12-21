"""MetricsAnalysisOrchestrator - Unified metrics trend detection (December 2025).

This module provides centralized metrics analysis and trend detection for
the self-improvement loop. It tracks key metrics over time and detects
significant patterns like plateaus, regressions, and improvements.

Event Integration:
- Subscribes to METRICS_UPDATED: Track metric changes
- Subscribes to ELO_UPDATED: Track Elo changes
- Subscribes to TRAINING_PROGRESS: Track training metrics
- Emits analysis events when trends are detected

Key Responsibilities:
1. Track metrics over time with sliding windows
2. Detect plateaus (no improvement)
3. Detect regressions (significant degradation)
4. Detect improvements (significant gains)
5. Provide trend analysis for decision making

Usage:
    from app.coordination.metrics_analysis_orchestrator import (
        MetricsAnalysisOrchestrator,
        wire_metrics_events,
        get_metrics_orchestrator,
    )

    # Wire metrics events
    orchestrator = wire_metrics_events()

    # Get trend for a metric
    trend = orchestrator.get_trend("elo")
    print(f"Elo trend: {trend.direction}, change: {trend.change_rate:.2f}/epoch")

    # Check for plateau
    if orchestrator.is_plateau("val_loss"):
        print("Validation loss has plateaued")
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of a metric trend."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    PLATEAU = "plateau"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics and their optimization direction."""

    MINIMIZE = "minimize"  # Lower is better (loss)
    MAXIMIZE = "maximize"  # Higher is better (elo, win_rate)


@dataclass
class MetricPoint:
    """A single metric data point."""

    value: float
    timestamp: float = field(default_factory=time.time)
    epoch: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analysis of a metric's trend."""

    metric_name: str
    direction: TrendDirection
    current_value: float
    best_value: float
    worst_value: float
    change_rate: float  # Per epoch or per hour
    std_dev: float
    samples: int
    is_plateau: bool
    is_regression: bool
    plateau_epochs: int = 0
    regression_severity: float = 0.0
    window_start: float = 0.0
    window_end: float = 0.0


AnalysisResult = TrendAnalysis


@dataclass
class AnomalyDetection:
    """Detection of a metric anomaly."""

    metric_name: str
    anomaly_type: str  # "spike", "drop", "outlier"
    value: float
    expected_range: tuple
    severity: float  # Standard deviations from mean
    detected_at: float = field(default_factory=time.time)


class MetricTracker:
    """Tracks a single metric over time."""

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        window_size: int = 100,
        plateau_threshold: float = 0.001,
        plateau_window: int = 10,
        regression_threshold: float = 0.05,
        anomaly_threshold: float = 3.0,  # Standard deviations
    ):
        self.name = name
        self.metric_type = metric_type
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.regression_threshold = regression_threshold
        self.anomaly_threshold = anomaly_threshold

        self._history: deque[MetricPoint] = deque(maxlen=window_size)
        self._best_value: float | None = None
        self._worst_value: float | None = None
        self._epochs_since_improvement = 0
        self._last_improvement_value: float | None = None

    def add_point(self, value: float, epoch: int = 0, **metadata) -> None:
        """Add a data point."""
        point = MetricPoint(value=value, epoch=epoch, metadata=metadata)
        self._history.append(point)

        # Track best/worst
        if self._best_value is None:
            self._best_value = value
            self._worst_value = value
            self._last_improvement_value = value
        else:
            # Check for improvement
            is_improvement = False
            if self.metric_type == MetricType.MINIMIZE:
                if value < self._best_value - self.plateau_threshold:
                    self._best_value = value
                    is_improvement = True
                self._worst_value = max(self._worst_value, value)
            else:
                if value > self._best_value + self.plateau_threshold:
                    self._best_value = value
                    is_improvement = True
                self._worst_value = min(self._worst_value, value)

            if is_improvement:
                self._epochs_since_improvement = 0
                self._last_improvement_value = value
            else:
                self._epochs_since_improvement += 1

    def get_values(self) -> list[float]:
        """Get all values in the window."""
        return [p.value for p in self._history]

    def get_mean(self) -> float:
        """Get mean value."""
        values = self.get_values()
        return statistics.mean(values) if values else 0.0

    def get_std_dev(self) -> float:
        """Get standard deviation."""
        values = self.get_values()
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)

    def get_trend_direction(self) -> TrendDirection:
        """Determine trend direction."""
        if len(self._history) < self.plateau_window:
            return TrendDirection.UNKNOWN

        # Check for plateau
        if self._epochs_since_improvement >= self.plateau_window:
            return TrendDirection.PLATEAU

        # Calculate recent trend
        recent = list(self._history)[-self.plateau_window:]
        if len(recent) < 2:
            return TrendDirection.UNKNOWN

        # Linear regression slope
        x = list(range(len(recent)))
        y = [p.value for p in recent]
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] ** 2 for i in range(n))

        denominator = n * sum_xx - sum_x ** 2
        if denominator == 0:
            return TrendDirection.STABLE

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Determine direction based on metric type
        if abs(slope) < self.plateau_threshold:
            return TrendDirection.STABLE

        if self.metric_type == MetricType.MINIMIZE:
            return TrendDirection.IMPROVING if slope < 0 else TrendDirection.DEGRADING
        else:
            return TrendDirection.IMPROVING if slope > 0 else TrendDirection.DEGRADING

    def get_change_rate(self) -> float:
        """Get rate of change per data point."""
        if len(self._history) < 2:
            return 0.0

        recent = list(self._history)[-self.plateau_window:]
        if len(recent) < 2:
            return 0.0

        first_value = recent[0].value
        last_value = recent[-1].value

        return (last_value - first_value) / len(recent)

    def is_plateau(self) -> bool:
        """Check if metric is in plateau."""
        return self._epochs_since_improvement >= self.plateau_window

    def is_regression(self) -> bool:
        """Check if metric has regressed significantly."""
        if self._best_value is None or len(self._history) == 0:
            return False

        current = self._history[-1].value

        regression = self._compute_regression_severity(current)
        return regression > self.regression_threshold

    def _compute_regression_severity(self, current: float) -> float:
        """Compute regression severity as a fraction of best value."""
        if self._best_value is None or self._best_value <= 0:
            return 0.0

        if self.metric_type == MetricType.MINIMIZE:
            return max(0.0, (current - self._best_value) / self._best_value)
        return max(0.0, (self._best_value - current) / self._best_value)

    def check_anomaly(self) -> AnomalyDetection | None:
        """Check if latest value is an anomaly."""
        if len(self._history) < 10:
            return None

        values = self.get_values()
        current = values[-1]
        mean = statistics.mean(values[:-1])
        std = statistics.stdev(values[:-1]) if len(values) > 2 else 1.0

        if std == 0:
            return None

        z_score = abs(current - mean) / std

        if z_score > self.anomaly_threshold:
            anomaly_type = "spike" if current > mean else "drop"
            return AnomalyDetection(
                metric_name=self.name,
                anomaly_type=anomaly_type,
                value=current,
                expected_range=(mean - 2 * std, mean + 2 * std),
                severity=z_score,
            )

        return None

    def analyze(self) -> TrendAnalysis:
        """Get full trend analysis."""
        values = self.get_values()
        current_value = values[-1] if values else 0.0
        regression_severity = self._compute_regression_severity(current_value)

        return TrendAnalysis(
            metric_name=self.name,
            direction=self.get_trend_direction(),
            current_value=current_value,
            best_value=self._best_value or 0.0,
            worst_value=self._worst_value or 0.0,
            change_rate=self.get_change_rate(),
            std_dev=self.get_std_dev(),
            samples=len(values),
            is_plateau=self.is_plateau(),
            is_regression=self.is_regression(),
            plateau_epochs=self._epochs_since_improvement,
            regression_severity=regression_severity,
            window_start=self._history[0].timestamp if self._history else 0.0,
            window_end=self._history[-1].timestamp if self._history else 0.0,
        )


class MetricsAnalysisOrchestrator:
    """Orchestrates metrics analysis and trend detection."""

    def __init__(
        self,
        window_size: int = 100,
        plateau_threshold: float = 0.001,
        plateau_window: int = 10,
    ):
        """Initialize MetricsAnalysisOrchestrator.

        Args:
            window_size: Size of sliding window for each metric
            plateau_threshold: Minimum change to not be plateau
            plateau_window: Epochs to detect plateau
        """
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window

        # Metric trackers
        self._trackers: dict[str, MetricTracker] = {}

        # Default metric types
        self._metric_types: dict[str, MetricType] = {
            "train_loss": MetricType.MINIMIZE,
            "val_loss": MetricType.MINIMIZE,
            "loss": MetricType.MINIMIZE,
            "elo": MetricType.MAXIMIZE,
            "win_rate": MetricType.MAXIMIZE,
            "accuracy": MetricType.MAXIMIZE,
            "policy_accuracy": MetricType.MAXIMIZE,
            "value_mse": MetricType.MINIMIZE,
        }

        # Anomaly history
        self._anomalies: list[AnomalyDetection] = []

        # Callbacks
        self._plateau_callbacks: list[Callable[[str], None]] = []
        self._regression_callbacks: list[Callable[[str, float], None]] = []
        self._anomaly_callbacks: list[Callable[[AnomalyDetection], None]] = []

        # Subscription state
        self._subscribed = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to metrics-related events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()

            router.subscribe(DataEventType.METRICS_UPDATED.value, self._on_metrics_updated)
            router.subscribe(DataEventType.ELO_UPDATED.value, self._on_elo_updated)
            router.subscribe(DataEventType.TRAINING_PROGRESS.value, self._on_training_progress)

            # Subscribe to cache events for window reset (December 2025)
            router.subscribe(DataEventType.CACHE_INVALIDATED.value, self._on_cache_invalidated)

            self._subscribed = True
            logger.info("[MetricsAnalysisOrchestrator] Subscribed to metrics events")
            return True

        except ImportError:
            logger.warning("[MetricsAnalysisOrchestrator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[MetricsAnalysisOrchestrator] Failed to subscribe: {e}")
            return False

    def _emit_plateau_detected(self, metric_name: str, tracker: MetricTracker) -> None:
        """Emit PLATEAU_DETECTED event (December 2025).

        Uses centralized event_emitters for consistent event emission.
        """
        try:
            import asyncio

            from app.coordination.event_emitters import emit_plateau_detected

            current_value = tracker.get_values()[-1] if tracker.get_values() else 0.0
            best_value = tracker._best_value or 0.0
            epochs_since = tracker._epochs_since_improvement
            plateau_type = "loss" if "loss" in metric_name else "elo" if "elo" in metric_name else "metric"

            try:
                asyncio.get_running_loop()
                asyncio.create_task(emit_plateau_detected(
                    metric_name=metric_name,
                    current_value=current_value,
                    best_value=best_value,
                    epochs_since_improvement=epochs_since,
                    plateau_type=plateau_type,
                ))
            except RuntimeError:
                asyncio.run(emit_plateau_detected(
                    metric_name=metric_name,
                    current_value=current_value,
                    best_value=best_value,
                    epochs_since_improvement=epochs_since,
                    plateau_type=plateau_type,
                ))

            logger.debug(f"[MetricsAnalysisOrchestrator] Emitted PLATEAU_DETECTED for {metric_name}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[MetricsAnalysisOrchestrator] Failed to emit plateau event: {e}")

    def _emit_regression_detected(self, metric_name: str, severity: float) -> None:
        """Emit REGRESSION_DETECTED event (December 2025).

        Uses centralized event_emitters for consistent event emission.
        """
        try:
            import asyncio

            from app.coordination.event_emitters import emit_regression_detected

            # Determine severity level
            if severity > 0.20:
                severity_level = "critical"
            elif severity > 0.10:
                severity_level = "severe"
            elif severity > 0.05:
                severity_level = "moderate"
            else:
                severity_level = "minor"

            # Get tracker for current/previous values
            tracker = self._trackers.get(metric_name)
            current_value = tracker.get_values()[-1] if tracker and tracker.get_values() else 0.0
            previous_value = tracker._best_value if tracker else 0.0

            try:
                asyncio.get_running_loop()
                asyncio.create_task(emit_regression_detected(
                    metric_name=metric_name,
                    current_value=current_value,
                    previous_value=previous_value,
                    severity=severity_level,
                ))
            except RuntimeError:
                asyncio.run(emit_regression_detected(
                    metric_name=metric_name,
                    current_value=current_value,
                    previous_value=previous_value,
                    severity=severity_level,
                ))

            logger.debug(f"[MetricsAnalysisOrchestrator] Emitted REGRESSION_DETECTED for {metric_name}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[MetricsAnalysisOrchestrator] Failed to emit regression event: {e}")

    def _get_or_create_tracker(self, name: str) -> MetricTracker:
        """Get or create a tracker for a metric."""
        if name not in self._trackers:
            metric_type = self._metric_types.get(name, MetricType.MINIMIZE)
            self._trackers[name] = MetricTracker(
                name=name,
                metric_type=metric_type,
                window_size=self.window_size,
                plateau_threshold=self.plateau_threshold,
                plateau_window=self.plateau_window,
            )
        return self._trackers[name]

    async def _on_metrics_updated(self, event) -> None:
        """Handle METRICS_UPDATED event."""
        payload = event.payload
        epoch = payload.get("epoch", 0)

        for key, value in payload.items():
            if key in ("epoch", "timestamp", "source"):
                continue
            if isinstance(value, (int, float)):
                self.record_metric(key, value, epoch=epoch)

    async def _on_elo_updated(self, event) -> None:
        """Handle ELO_UPDATED event."""
        payload = event.payload

        if "elo" in payload:
            self.record_metric("elo", payload["elo"])

        if "win_rate" in payload:
            self.record_metric("win_rate", payload["win_rate"])

    async def _on_training_progress(self, event) -> None:
        """Handle TRAINING_PROGRESS event."""
        payload = event.payload
        epoch = payload.get("epoch", 0)

        for key in ["train_loss", "val_loss", "loss", "accuracy"]:
            if key in payload:
                self.record_metric(key, payload[key], epoch=epoch)

    async def _on_cache_invalidated(self, event) -> None:
        """Handle CACHE_INVALIDATED - reset metric windows when caches are flushed.

        This prevents stale metrics from affecting trend detection after a cache
        invalidation, which may indicate a model change or data reset.
        """
        payload = event.payload

        invalidation_type = payload.get("invalidation_type", "")
        target_id = payload.get("target_id", "")
        count = payload.get("count", 0)

        # Determine which metrics to reset based on invalidation type
        metrics_to_reset = []

        if invalidation_type == "model":
            # Model invalidation: reset all training-related metrics
            metrics_to_reset = ["train_loss", "val_loss", "loss", "accuracy", "policy_accuracy", "value_mse"]
            logger.info(
                f"[MetricsAnalysisOrchestrator] Model cache invalidated ({target_id}): "
                "resetting training metrics"
            )
        elif invalidation_type == "evaluation":
            # Evaluation invalidation: reset ELO and win rate
            metrics_to_reset = ["elo", "win_rate"]
            logger.info(
                "[MetricsAnalysisOrchestrator] Evaluation cache invalidated: "
                "resetting evaluation metrics"
            )
        elif invalidation_type == "full" or count > 100:
            # Full flush or large invalidation: reset all metrics
            logger.warning(
                "[MetricsAnalysisOrchestrator] Full cache invalidation: "
                "resetting ALL metric windows"
            )
            self.reset_all_windows()
            return

        # Reset specific metrics
        for metric in metrics_to_reset:
            self.reset_window(metric)

    def reset_window(self, metric_name: str) -> bool:
        """Reset the sliding window for a specific metric.

        Args:
            metric_name: Name of the metric to reset

        Returns:
            True if metric was found and reset
        """
        if metric_name not in self._trackers:
            return False

        tracker = self._trackers[metric_name]

        # Clear history but preserve configuration
        tracker._history.clear()
        tracker._best_value = None
        tracker._worst_value = None
        tracker._epochs_since_improvement = 0
        tracker._last_improvement_value = None

        logger.debug(f"[MetricsAnalysisOrchestrator] Reset window for {metric_name}")
        return True

    def reset_all_windows(self) -> int:
        """Reset all metric windows.

        Returns:
            Number of metrics reset
        """
        count = 0
        for name in list(self._trackers.keys()):
            if self.reset_window(name):
                count += 1

        # Also clear anomaly history
        self._anomalies.clear()

        logger.info(f"[MetricsAnalysisOrchestrator] Reset {count} metric windows")
        return count

    def reset_windows_by_type(self, metric_type: MetricType) -> int:
        """Reset windows for metrics of a specific type.

        Args:
            metric_type: Type of metrics to reset (MINIMIZE or MAXIMIZE)

        Returns:
            Number of metrics reset
        """
        count = 0
        for name, tracker in self._trackers.items():
            if tracker.metric_type == metric_type and self.reset_window(name):
                count += 1

        logger.info(
            f"[MetricsAnalysisOrchestrator] Reset {count} {metric_type.value} metric windows"
        )
        return count

    def record_metric(
        self, name: str, value: float, epoch: int = 0, **metadata
    ) -> AnomalyDetection | None:
        """Record a metric value.

        Returns:
            AnomalyDetection if anomaly detected, None otherwise
        """
        tracker = self._get_or_create_tracker(name)
        was_plateau = tracker.is_plateau()
        was_regression = tracker.is_regression()

        tracker.add_point(value, epoch=epoch, **metadata)

        # Check for state changes
        is_plateau = tracker.is_plateau()
        is_regression = tracker.is_regression()

        if is_plateau and not was_plateau:
            for callback in self._plateau_callbacks:
                try:
                    callback(name)
                except Exception as e:
                    logger.error(f"[MetricsAnalysisOrchestrator] Plateau callback error: {e}")
            # Emit PLATEAU_DETECTED event (December 2025)
            self._emit_plateau_detected(name, tracker)
            logger.info(f"[MetricsAnalysisOrchestrator] Plateau detected for {name}")

        if is_regression and not was_regression:
            analysis = tracker.analyze()
            for callback in self._regression_callbacks:
                try:
                    callback(name, analysis.regression_severity)
                except Exception as e:
                    logger.error(f"[MetricsAnalysisOrchestrator] Regression callback error: {e}")
            # Emit REGRESSION_DETECTED event (December 2025)
            self._emit_regression_detected(name, analysis.regression_severity)
            logger.warning(f"[MetricsAnalysisOrchestrator] Regression detected for {name}")

        # Check for anomaly
        anomaly = tracker.check_anomaly()
        if anomaly:
            self._anomalies.append(anomaly)
            if len(self._anomalies) > 100:
                self._anomalies = self._anomalies[-100:]

            for callback in self._anomaly_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"[MetricsAnalysisOrchestrator] Anomaly callback error: {e}")

        return anomaly

    def set_metric_type(self, name: str, metric_type: MetricType) -> None:
        """Set the optimization direction for a metric."""
        self._metric_types[name] = metric_type
        if name in self._trackers:
            self._trackers[name].metric_type = metric_type

    def get_trend(self, name: str) -> TrendAnalysis | None:
        """Get trend analysis for a metric."""
        if name not in self._trackers:
            return None
        return self._trackers[name].analyze()

    def get_all_trends(self) -> dict[str, TrendAnalysis]:
        """Get trend analysis for all tracked metrics."""
        return {name: tracker.analyze() for name, tracker in self._trackers.items()}

    def is_plateau(self, name: str) -> bool:
        """Check if a metric is in plateau."""
        if name not in self._trackers:
            return False
        return self._trackers[name].is_plateau()

    def is_regression(self, name: str) -> bool:
        """Check if a metric has regressed."""
        if name not in self._trackers:
            return False
        return self._trackers[name].is_regression()

    def get_current_value(self, name: str) -> float | None:
        """Get current value of a metric."""
        if name not in self._trackers:
            return None
        values = self._trackers[name].get_values()
        return values[-1] if values else None

    def get_best_value(self, name: str) -> float | None:
        """Get best value of a metric."""
        if name not in self._trackers:
            return None
        return self._trackers[name]._best_value

    def on_plateau(self, callback: Callable[[str], None]) -> None:
        """Register callback for plateau detection."""
        self._plateau_callbacks.append(callback)

    def on_regression(self, callback: Callable[[str, float], None]) -> None:
        """Register callback for regression detection."""
        self._regression_callbacks.append(callback)

    def on_anomaly(self, callback: Callable[[AnomalyDetection], None]) -> None:
        """Register callback for anomaly detection."""
        self._anomaly_callbacks.append(callback)

    def get_anomalies(self, limit: int = 50) -> list[AnomalyDetection]:
        """Get recent anomalies."""
        return self._anomalies[-limit:]

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status for monitoring."""
        trends = self.get_all_trends()

        plateaus = [name for name, t in trends.items() if t.is_plateau]
        regressions = [name for name, t in trends.items() if t.is_regression]

        return {
            "metrics_tracked": len(self._trackers),
            "metrics": list(self._trackers.keys()),
            "plateaus": plateaus,
            "regressions": regressions,
            "anomalies_detected": len(self._anomalies),
            "trends": {
                name: {
                    "direction": t.direction.value,
                    "current": round(t.current_value, 4),
                    "best": round(t.best_value, 4),
                    "change_rate": round(t.change_rate, 6),
                }
                for name, t in trends.items()
            },
            "subscribed": self._subscribed,
        }


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_metrics_orchestrator: MetricsAnalysisOrchestrator | None = None


def get_metrics_orchestrator() -> MetricsAnalysisOrchestrator:
    """Get the global MetricsAnalysisOrchestrator singleton."""
    global _metrics_orchestrator
    if _metrics_orchestrator is None:
        _metrics_orchestrator = MetricsAnalysisOrchestrator()
    return _metrics_orchestrator


def wire_metrics_events() -> MetricsAnalysisOrchestrator:
    """Wire metrics events to the orchestrator."""
    orchestrator = get_metrics_orchestrator()
    orchestrator.subscribe_to_events()
    return orchestrator


def is_metric_plateau(name: str) -> bool:
    """Convenience function to check if metric is plateau."""
    return get_metrics_orchestrator().is_plateau(name)


def get_metric_trend(name: str) -> TrendAnalysis | None:
    """Convenience function to get metric trend."""
    return get_metrics_orchestrator().get_trend(name)

def record_metric(name: str, value: float, epoch: int = 0, **metadata) -> AnomalyDetection | None:
    """Convenience function to record a metric value."""
    return get_metrics_orchestrator().record_metric(name, value, epoch=epoch, **metadata)

def analyze_metrics() -> dict[str, TrendAnalysis]:
    """Convenience function to analyze all tracked metrics."""
    return get_metrics_orchestrator().get_all_trends()


__all__ = [
    "AnalysisResult",
    "AnomalyDetection",
    "MetricPoint",
    "MetricTracker",
    "MetricType",
    "MetricsAnalysisOrchestrator",
    "TrendAnalysis",
    "TrendDirection",
    "analyze_metrics",
    "get_metric_trend",
    "get_metrics_orchestrator",
    "is_metric_plateau",
    "record_metric",
    "wire_metrics_events",
]

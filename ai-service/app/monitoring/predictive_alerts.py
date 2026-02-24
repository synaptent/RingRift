"""
Proactive monitoring with predictive alerts.

This module provides early warning alerts by predicting issues before they occur,
rather than alerting after problems have already happened.

Note (December 29, 2025): This module uses a local AlertSeverity enum for backwards
compatibility with existing code that expects str values ("info", "warning", "critical").
For new code, prefer using app.coordination.alert_types.AlertSeverity (IntEnum: 0-4).
"""
from __future__ import annotations


import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config.thresholds import DISK_CRITICAL_PERCENT

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels (str-based for backwards compatibility).

    Note: For new code, prefer app.coordination.alert_types.AlertSeverity (IntEnum).
    This str-based enum is maintained for backwards compatibility with code that
    expects lowercase string values.
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of predictive alerts."""
    DISK_FULL = "disk_full"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    ELO_DEGRADATION = "elo_degradation"
    NODE_FAILURE = "node_failure"
    QUEUE_BACKLOG = "queue_backlog"
    TRAINING_STALL = "training_stall"
    MODEL_REGRESSION = "model_regression"


@dataclass
class PredictiveAlertConfig:
    """Configuration for predictive alerting."""

    # Enabled flag
    enabled: bool = True

    # Disk prediction
    disk_prediction_hours: int = 4          # Alert N hours before full
    disk_critical_threshold: float = float(DISK_CRITICAL_PERCENT)

    # Memory prediction
    memory_prediction_hours: int = 2
    memory_critical_threshold: float = 95.0

    # Elo degradation
    elo_trend_window_hours: int = 6
    elo_degradation_threshold: float = -5.0  # Alert if losing > 5 Elo/hour

    # Queue backlog
    queue_backlog_threshold: int = 50       # Alert if > 50 pending
    queue_growth_rate_threshold: float = 10.0  # Alert if growing > 10/hour

    # Training stall
    training_stall_hours: int = 6           # Alert if no training in N hours

    # Throttling
    alert_throttle_minutes: int = 30        # Min time between same alert
    max_alerts_per_hour: int = 20           # Rate limit


@dataclass
class Alert:
    """A predictive alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    target_id: str              # Node ID, model ID, etc.
    message: str
    action: str                 # Recommended action
    created_at: float = field(default_factory=time.time)
    predicted_issue_time: float | None = None  # When issue is predicted to occur
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSample:
    """A single metric sample."""
    timestamp: float
    value: float


class PredictiveAlertManager:
    """
    Alert before problems occur, not after.

    Analyzes trends in metrics to predict when issues will occur
    and alerts with enough lead time for proactive action.
    """

    def __init__(self, config: PredictiveAlertConfig | None = None):
        self.config = config or PredictiveAlertConfig()

        # Metric history for trend analysis
        self._disk_history: dict[str, list[MetricSample]] = {}
        self._memory_history: dict[str, list[MetricSample]] = {}
        self._elo_history: dict[str, list[MetricSample]] = {}
        self._queue_history: list[MetricSample] = []

        # Alert tracking for throttling
        self._recent_alerts: dict[str, float] = {}  # key -> last alert time
        self._alerts_this_hour: int = 0
        self._hour_start: float = time.time()

        # Notification callback
        self._notify_callback: Callable | None = None

    def set_notify_callback(self, callback: Callable) -> None:
        """Set callback for sending notifications."""
        self._notify_callback = callback

    def record_disk_usage(self, node_id: str, usage_percent: float) -> None:
        """Record disk usage for a node."""
        if node_id not in self._disk_history:
            self._disk_history[node_id] = []

        self._disk_history[node_id].append(
            MetricSample(timestamp=time.time(), value=usage_percent)
        )

        # Keep last 24 hours
        self._cleanup_history(self._disk_history[node_id], max_age_hours=24)

    def record_memory_usage(self, node_id: str, usage_percent: float) -> None:
        """Record memory usage for a node."""
        if node_id not in self._memory_history:
            self._memory_history[node_id] = []

        self._memory_history[node_id].append(
            MetricSample(timestamp=time.time(), value=usage_percent)
        )

        self._cleanup_history(self._memory_history[node_id], max_age_hours=24)

    def record_elo(self, model_id: str, elo: float) -> None:
        """Record Elo rating for a model."""
        if model_id not in self._elo_history:
            self._elo_history[model_id] = []

        self._elo_history[model_id].append(
            MetricSample(timestamp=time.time(), value=elo)
        )

        self._cleanup_history(self._elo_history[model_id], max_age_hours=48)

    def record_queue_depth(self, pending_count: int) -> None:
        """Record work queue depth."""
        self._queue_history.append(
            MetricSample(timestamp=time.time(), value=float(pending_count))
        )

        self._cleanup_history(self._queue_history, max_age_hours=24)

    def _cleanup_history(
        self,
        history: list[MetricSample],
        max_age_hours: int,
    ) -> None:
        """Remove old samples from history."""
        cutoff = time.time() - (max_age_hours * 3600)
        while history and history[0].timestamp < cutoff:
            history.pop(0)

    def _calculate_trend(
        self,
        samples: list[MetricSample],
        window_hours: int,
    ) -> float | None:
        """
        Calculate linear trend (slope) over the window.

        Returns rate of change per hour, or None if insufficient data.
        """
        if len(samples) < 3:
            return None

        cutoff = time.time() - (window_hours * 3600)
        recent = [s for s in samples if s.timestamp >= cutoff]

        if len(recent) < 3:
            return None

        # Linear regression
        n = len(recent)
        sum_x = sum(s.timestamp for s in recent)
        sum_y = sum(s.value for s in recent)
        sum_xy = sum(s.timestamp * s.value for s in recent)
        sum_x2 = sum(s.timestamp ** 2 for s in recent)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # Convert from per-second to per-hour
        return slope * 3600

    def _should_alert(self, alert_key: str) -> bool:
        """Check if we should emit an alert (throttling)."""
        if not self.config.enabled:
            return False

        # Reset hourly counter if needed
        if time.time() - self._hour_start > 3600:
            self._alerts_this_hour = 0
            self._hour_start = time.time()

        # Check rate limit
        if self._alerts_this_hour >= self.config.max_alerts_per_hour:
            return False

        # Check throttle
        last_alert = self._recent_alerts.get(alert_key, 0)
        return not time.time() - last_alert < self.config.alert_throttle_minutes * 60

    def _record_alert(self, alert_key: str) -> None:
        """Record that an alert was sent."""
        self._recent_alerts[alert_key] = time.time()
        self._alerts_this_hour += 1

    def predict_disk_full(
        self,
        node_id: str,
        hours_ahead: int | None = None,
    ) -> Alert | None:
        """
        Predict if disk will fill based on current write rate.

        Args:
            node_id: Node to check
            hours_ahead: Override prediction window

        Returns:
            Alert if disk predicted to fill within window, None otherwise
        """
        hours_ahead = hours_ahead or self.config.disk_prediction_hours

        if node_id not in self._disk_history:
            return None

        samples = self._disk_history[node_id]
        if not samples:
            return None

        current_usage = samples[-1].value

        # Check if already critical
        if current_usage >= self.config.disk_critical_threshold:
            alert_key = f"disk_critical_{node_id}"
            if not self._should_alert(alert_key):
                return None

            self._record_alert(alert_key)
            return Alert(
                alert_id=f"disk_{node_id}_{int(time.time())}",
                alert_type=AlertType.DISK_FULL,
                severity=AlertSeverity.CRITICAL,
                target_id=node_id,
                message=f"Node {node_id} disk at {current_usage:.1f}% (critical)",
                action="immediate_cleanup_required",
                metadata={"current_usage": current_usage},
            )

        # Calculate trend
        growth_rate = self._calculate_trend(samples, window_hours=4)
        if growth_rate is None or growth_rate <= 0:
            return None  # Not growing

        # Predict time to full
        remaining_capacity = 100 - current_usage
        hours_until_full = remaining_capacity / growth_rate

        if hours_until_full <= hours_ahead:
            alert_key = f"disk_prediction_{node_id}"
            if not self._should_alert(alert_key):
                return None

            self._record_alert(alert_key)
            predicted_time = time.time() + (hours_until_full * 3600)

            return Alert(
                alert_id=f"disk_{node_id}_{int(time.time())}",
                alert_type=AlertType.DISK_FULL,
                severity=AlertSeverity.WARNING,
                target_id=node_id,
                message=f"Node {node_id} disk predicted full in {hours_until_full:.1f}h",
                action="cleanup_recommended",
                predicted_issue_time=predicted_time,
                metadata={
                    "current_usage": current_usage,
                    "growth_rate_per_hour": growth_rate,
                    "hours_until_full": hours_until_full,
                },
            )

        return None

    def predict_elo_degradation(
        self,
        model_id: str,
        window_hours: int | None = None,
    ) -> Alert | None:
        """
        Detect early signs of Elo regression.

        Args:
            model_id: Model to check
            window_hours: Override trend window

        Returns:
            Alert if Elo degrading significantly, None otherwise
        """
        window_hours = window_hours or self.config.elo_trend_window_hours

        if model_id not in self._elo_history:
            return None

        samples = self._elo_history[model_id]
        if len(samples) < 10:
            return None

        slope = self._calculate_trend(samples, window_hours)
        if slope is None:
            return None

        if slope < self.config.elo_degradation_threshold:
            alert_key = f"elo_degradation_{model_id}"
            if not self._should_alert(alert_key):
                return None

            self._record_alert(alert_key)
            current_elo = samples[-1].value

            return Alert(
                alert_id=f"elo_{model_id}_{int(time.time())}",
                alert_type=AlertType.ELO_DEGRADATION,
                severity=AlertSeverity.WARNING,
                target_id=model_id,
                message=f"Model {model_id} showing Elo degradation ({slope:.1f}/hr)",
                action="investigation_recommended",
                metadata={
                    "current_elo": current_elo,
                    "elo_change_per_hour": slope,
                    "window_hours": window_hours,
                },
            )

        return None

    def predict_queue_backlog(self) -> Alert | None:
        """
        Predict if work queue will become backlogged.

        Returns:
            Alert if queue growing too fast or already backlogged
        """
        if not self._queue_history:
            return None

        current_depth = int(self._queue_history[-1].value)

        # Check immediate backlog
        if current_depth > self.config.queue_backlog_threshold:
            alert_key = "queue_backlog"
            if not self._should_alert(alert_key):
                return None

            self._record_alert(alert_key)
            return Alert(
                alert_id=f"queue_backlog_{int(time.time())}",
                alert_type=AlertType.QUEUE_BACKLOG,
                severity=AlertSeverity.WARNING,
                target_id="work_queue",
                message=f"Work queue backlogged: {current_depth} pending items",
                action="scale_up_recommended",
                metadata={"pending_count": current_depth},
            )

        # Check growth rate
        growth_rate = self._calculate_trend(self._queue_history, window_hours=1)
        if growth_rate is not None and growth_rate > self.config.queue_growth_rate_threshold:
            alert_key = "queue_growth"
            if not self._should_alert(alert_key):
                return None

            self._record_alert(alert_key)
            return Alert(
                alert_id=f"queue_growth_{int(time.time())}",
                alert_type=AlertType.QUEUE_BACKLOG,
                severity=AlertSeverity.WARNING,
                target_id="work_queue",
                message=f"Work queue growing rapidly: +{growth_rate:.1f}/hour",
                action="scale_up_recommended",
                metadata={
                    "pending_count": current_depth,
                    "growth_rate_per_hour": growth_rate,
                },
            )

        return None

    def check_training_stall(self, last_training_time: float) -> Alert | None:
        """
        Check if training has stalled.

        Args:
            last_training_time: Timestamp of last training completion

        Returns:
            Alert if training has been stalled
        """
        hours_since_training = (time.time() - last_training_time) / 3600

        if hours_since_training > self.config.training_stall_hours:
            alert_key = "training_stall"
            if not self._should_alert(alert_key):
                return None

            self._record_alert(alert_key)
            return Alert(
                alert_id=f"training_stall_{int(time.time())}",
                alert_type=AlertType.TRAINING_STALL,
                severity=AlertSeverity.WARNING,
                target_id="training_pipeline",
                message=f"No training completed in {hours_since_training:.1f} hours",
                action="check_training_pipeline",
                metadata={"hours_since_training": hours_since_training},
            )

        return None

    async def run_all_checks(
        self,
        node_ids: list[str],
        model_ids: list[str],
        last_training_time: float,
    ) -> list[Alert]:
        """
        Run all predictive checks and return alerts.

        Args:
            node_ids: Active node IDs to check
            model_ids: Production model IDs to check
            last_training_time: Timestamp of last training

        Returns:
            List of alerts that should be sent
        """
        alerts = []

        # Disk predictions
        for node_id in node_ids:
            alert = self.predict_disk_full(node_id)
            if alert:
                alerts.append(alert)

        # Elo degradation
        for model_id in model_ids:
            alert = self.predict_elo_degradation(model_id)
            if alert:
                alerts.append(alert)

        # Queue backlog
        alert = self.predict_queue_backlog()
        if alert:
            alerts.append(alert)

        # Training stall
        alert = self.check_training_stall(last_training_time)
        if alert:
            alerts.append(alert)

        return alerts

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert via the notification callback.

        Returns True if sent successfully.
        """
        if self._notify_callback is None:
            logger.warning(f"No notification callback set, dropping alert: {alert.message}")
            return False

        try:
            await self._notify_callback(alert)
            logger.info(f"Sent alert: {alert.alert_type.value} - {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alerting statistics for monitoring."""
        return {
            "enabled": self.config.enabled,
            "alerts_this_hour": self._alerts_this_hour,
            "max_alerts_per_hour": self.config.max_alerts_per_hour,
            "nodes_tracked": len(self._disk_history),
            "models_tracked": len(self._elo_history),
            "queue_samples": len(self._queue_history),
            "throttled_alerts": len(self._recent_alerts),
        }


def load_alert_config_from_yaml(yaml_config: dict[str, Any]) -> PredictiveAlertConfig:
    """Load PredictiveAlertConfig from YAML configuration dict."""
    monitoring = yaml_config.get("proactive_monitoring", {})

    return PredictiveAlertConfig(
        enabled=monitoring.get("enabled", True),
        disk_prediction_hours=monitoring.get("disk_prediction_hours", 4),
        disk_critical_threshold=monitoring.get("disk_critical_threshold", float(DISK_CRITICAL_PERCENT)),
        memory_prediction_hours=monitoring.get("memory_prediction_hours", 2),
        memory_critical_threshold=monitoring.get("memory_critical_threshold", 95.0),
        elo_trend_window_hours=monitoring.get("elo_trend_window_hours", 6),
        elo_degradation_threshold=monitoring.get("elo_degradation_threshold", -5.0),
        queue_backlog_threshold=monitoring.get("queue_backlog_threshold", 50),
        queue_growth_rate_threshold=monitoring.get("queue_growth_rate_threshold", 10.0),
        training_stall_hours=monitoring.get("training_stall_hours", 6),
        alert_throttle_minutes=monitoring.get("alert_throttle_minutes", 30),
        max_alerts_per_hour=monitoring.get("max_alerts_per_hour", 20),
    )

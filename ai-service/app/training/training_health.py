"""Training Health Monitoring and Alerts.

Provides real-time monitoring of training pipeline health:
- Track training run status across configs
- Detect stalled or failed training
- Monitor data freshness and model staleness
- Expose Prometheus metrics for alerting
- Generate health reports

Usage:
    from app.training.training_health import TrainingHealthMonitor

    monitor = TrainingHealthMonitor()

    # Record events
    monitor.record_training_start("square8_2p")
    monitor.record_training_complete("square8_2p", success=True, metrics={...})

    # Check health
    health = monitor.get_health_status()
    print(f"Overall health: {health.status}")

    # Get alerts
    alerts = monitor.get_active_alerts()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
HEALTH_DB_PATH = AI_SERVICE_ROOT / "data" / "training" / "health_state.json"

# Thresholds
MAX_TRAINING_HOURS = 4  # Alert if training runs longer than this
STALE_MODEL_HOURS = 24  # Alert if model hasn't been updated in this time
STALE_DATA_HOURS = 12  # Alert if no new data in this time
MIN_WIN_RATE = 0.35  # Alert if win rate drops below this


class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TrainingRunStatus:
    """Status of a single training run."""
    config_key: str
    started_at: float
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ConfigHealth:
    """Health status for a single config."""
    config_key: str
    last_training_time: float = 0
    last_training_success: bool = True
    consecutive_failures: int = 0
    last_data_time: float = 0
    game_count: int = 0
    model_count: int = 0
    win_rate: float = 0.5
    is_training: bool = False
    training_start_time: Optional[float] = None


@dataclass
class Alert:
    """A health alert."""
    id: str
    severity: AlertSeverity
    config_key: Optional[str]
    message: str
    created_at: float
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "config_key": self.config_key,
            "message": self.message,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


@dataclass
class HealthReport:
    """Overall health report."""
    status: HealthStatus
    timestamp: float
    configs: Dict[str, ConfigHealth]
    active_alerts: List[Alert]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "configs": {k: vars(v) for k, v in self.configs.items()},
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "summary": self.summary,
        }


class TrainingHealthMonitor:
    """Monitors training pipeline health."""

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or HEALTH_DB_PATH
        self._configs: Dict[str, ConfigHealth] = {}
        self._active_runs: Dict[str, TrainingRunStatus] = {}
        self._alerts: Dict[str, Alert] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                    for key, config_data in data.get("configs", {}).items():
                        self._configs[key] = ConfigHealth(config_key=key, **config_data)
            except Exception as e:
                logger.warning(f"Could not load health state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "configs": {k: {
                    "last_training_time": v.last_training_time,
                    "last_training_success": v.last_training_success,
                    "consecutive_failures": v.consecutive_failures,
                    "last_data_time": v.last_data_time,
                    "game_count": v.game_count,
                    "model_count": v.model_count,
                    "win_rate": v.win_rate,
                } for k, v in self._configs.items()},
                "saved_at": time.time(),
            }
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save health state: {e}")

    def _get_or_create_config(self, config_key: str) -> ConfigHealth:
        """Get or create config health record."""
        if config_key not in self._configs:
            self._configs[config_key] = ConfigHealth(config_key=config_key)
        return self._configs[config_key]

    def record_training_start(self, config_key: str) -> None:
        """Record that training has started for a config."""
        now = time.time()
        config = self._get_or_create_config(config_key)
        config.is_training = True
        config.training_start_time = now

        self._active_runs[config_key] = TrainingRunStatus(
            config_key=config_key,
            started_at=now,
        )

        # Clear any stalled training alerts
        self._resolve_alert(f"stalled_training:{config_key}")
        logger.info(f"Training started for {config_key}")

    def record_training_complete(
        self,
        config_key: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record that training has completed."""
        now = time.time()
        config = self._get_or_create_config(config_key)
        config.is_training = False
        config.training_start_time = None
        config.last_training_time = now
        config.last_training_success = success

        if success:
            config.consecutive_failures = 0
            config.model_count += 1
        else:
            config.consecutive_failures += 1
            self._create_alert(
                f"training_failed:{config_key}",
                AlertSeverity.WARNING if config.consecutive_failures < 3 else AlertSeverity.CRITICAL,
                config_key,
                f"Training failed for {config_key} ({config.consecutive_failures} consecutive failures)",
            )

        # Update run status
        if config_key in self._active_runs:
            run = self._active_runs[config_key]
            run.completed_at = now
            run.success = success
            run.metrics = metrics or {}
            run.error_message = error_message
            del self._active_runs[config_key]

        self._save_state()
        logger.info(f"Training {'completed' if success else 'failed'} for {config_key}")

    def record_data_update(self, config_key: str, game_count: int) -> None:
        """Record new data available for a config."""
        config = self._get_or_create_config(config_key)
        config.last_data_time = time.time()
        config.game_count = game_count
        self._resolve_alert(f"stale_data:{config_key}")
        self._save_state()

    def record_win_rate(self, config_key: str, win_rate: float) -> None:
        """Record win rate for a config."""
        config = self._get_or_create_config(config_key)
        config.win_rate = win_rate

        if win_rate < MIN_WIN_RATE:
            self._create_alert(
                f"low_win_rate:{config_key}",
                AlertSeverity.WARNING,
                config_key,
                f"Win rate for {config_key} dropped to {win_rate:.1%}",
            )
        else:
            self._resolve_alert(f"low_win_rate:{config_key}")

        self._save_state()

    def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        config_key: Optional[str],
        message: str,
    ) -> None:
        """Create or update an alert."""
        if alert_id not in self._alerts or self._alerts[alert_id].resolved_at is not None:
            self._alerts[alert_id] = Alert(
                id=alert_id,
                severity=severity,
                config_key=config_key,
                message=message,
                created_at=time.time(),
            )
            logger.warning(f"Alert created: {message}")

    def _resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert."""
        if alert_id in self._alerts and self._alerts[alert_id].resolved_at is None:
            self._alerts[alert_id].resolved_at = time.time()
            logger.info(f"Alert resolved: {alert_id}")

    def check_health(self) -> None:
        """Run health checks and update alerts."""
        now = time.time()

        for config_key, config in self._configs.items():
            # Check for stalled training
            if config.is_training and config.training_start_time:
                hours_running = (now - config.training_start_time) / 3600
                if hours_running > MAX_TRAINING_HOURS:
                    self._create_alert(
                        f"stalled_training:{config_key}",
                        AlertSeverity.CRITICAL,
                        config_key,
                        f"Training for {config_key} has been running for {hours_running:.1f} hours",
                    )

            # Check for stale model
            if config.last_training_time > 0:
                hours_since_training = (now - config.last_training_time) / 3600
                if hours_since_training > STALE_MODEL_HOURS:
                    self._create_alert(
                        f"stale_model:{config_key}",
                        AlertSeverity.WARNING,
                        config_key,
                        f"No training for {config_key} in {hours_since_training:.1f} hours",
                    )
                else:
                    self._resolve_alert(f"stale_model:{config_key}")

            # Check for stale data
            if config.last_data_time > 0:
                hours_since_data = (now - config.last_data_time) / 3600
                if hours_since_data > STALE_DATA_HOURS:
                    self._create_alert(
                        f"stale_data:{config_key}",
                        AlertSeverity.WARNING,
                        config_key,
                        f"No new data for {config_key} in {hours_since_data:.1f} hours",
                    )

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self._alerts.values() if a.resolved_at is None]

    def get_health_status(self) -> HealthReport:
        """Get overall health status."""
        self.check_health()

        active_alerts = self.get_active_alerts()
        critical_count = sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL)
        warning_count = sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING)

        if critical_count > 0:
            status = HealthStatus.UNHEALTHY
        elif warning_count > 0:
            status = HealthStatus.DEGRADED
        elif len(self._configs) > 0:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.UNKNOWN

        summary_parts = []
        if critical_count > 0:
            summary_parts.append(f"{critical_count} critical alerts")
        if warning_count > 0:
            summary_parts.append(f"{warning_count} warnings")

        training_count = sum(1 for c in self._configs.values() if c.is_training)
        if training_count > 0:
            summary_parts.append(f"{training_count} training in progress")

        summary = ", ".join(summary_parts) if summary_parts else "All systems operational"

        return HealthReport(
            status=status,
            timestamp=time.time(),
            configs=self._configs.copy(),
            active_alerts=active_alerts,
            summary=summary,
        )

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Training status metrics
        for config_key, config in self._configs.items():
            safe_key = config_key.replace("-", "_")
            lines.append(f'ringrift_training_is_running{{config="{config_key}"}} {1 if config.is_training else 0}')
            lines.append(f'ringrift_training_consecutive_failures{{config="{config_key}"}} {config.consecutive_failures}')
            lines.append(f'ringrift_training_last_success_timestamp{{config="{config_key}"}} {config.last_training_time}')
            lines.append(f'ringrift_training_model_count{{config="{config_key}"}} {config.model_count}')
            lines.append(f'ringrift_training_win_rate{{config="{config_key}"}} {config.win_rate}')
            lines.append(f'ringrift_training_game_count{{config="{config_key}"}} {config.game_count}')

        # Alert counts
        active_alerts = self.get_active_alerts()
        critical = sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING)
        lines.append(f'ringrift_training_alerts_critical {critical}')
        lines.append(f'ringrift_training_alerts_warning {warning}')

        return "\n".join(lines)


# Singleton instance
_monitor_instance: Optional[TrainingHealthMonitor] = None


def get_training_health_monitor() -> TrainingHealthMonitor:
    """Get the global training health monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TrainingHealthMonitor()
    return _monitor_instance

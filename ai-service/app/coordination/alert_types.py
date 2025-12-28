#!/usr/bin/env python3
"""Centralized Alert Types for Coordination Infrastructure.

December 27, 2025: Consolidates 8+ duplicate Alert/Severity enums scattered across:
- app/config/thresholds.py (AlertLevel)
- app/coordination/unified_health_manager.py (ErrorSeverity)
- app/coordination/daemon_watchdog.py (WatchdogAlert)
- app/coordination/stall_detection.py (StallSeverity)
- app/coordination/unified_replication_daemon.py (ReplicationAlertLevel)
- app/training/training_health.py (AlertSeverity, AlertLevel, Alert)
- app/training/regression_detector.py (RegressionSeverity)
- app/monitoring/alert_router.py (AlertSeverity, Alert)
- app/monitoring/predictive_alerts.py (AlertSeverity, Alert)

Usage:
    from app.coordination.alert_types import (
        AlertSeverity,
        Alert,
        AlertCategory,
        create_alert,
    )

    # Create an alert
    alert = create_alert(
        title="High GPU utilization",
        message="GPU 0 at 95% utilization",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.RESOURCE,
        source="gpu_monitor",
    )

Migration:
    Existing modules should import from this module instead of defining their own enums.
    Backward-compatible aliases are provided for gradual migration.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class AlertSeverity(IntEnum):
    """Unified alert severity levels (ordered by severity).

    Uses IntEnum for easy comparison (e.g., severity >= AlertSeverity.WARNING).

    Values:
        DEBUG (0): Debug-level information, not user-facing
        INFO (1): Informational, no action required
        WARNING (2): Potential issue, may require attention
        ERROR (3): Error occurred, action recommended
        CRITICAL (4): Critical issue, immediate action required
    """
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertCategory(str, Enum):
    """Categories for alerts to enable filtering and routing.

    Values:
        TRAINING: Training-related alerts (loss anomaly, convergence issues)
        EVALUATION: Model evaluation alerts (gauntlet failures, regression)
        RESOURCE: Resource utilization alerts (GPU, memory, disk)
        CLUSTER: Cluster health alerts (node failures, connectivity)
        SYNC: Data synchronization alerts (replication, distribution)
        QUALITY: Data quality alerts (corruption, schema issues)
        SYSTEM: System-level alerts (daemon failures, crashes)
    """
    TRAINING = "training"
    EVALUATION = "evaluation"
    RESOURCE = "resource"
    CLUSTER = "cluster"
    SYNC = "sync"
    QUALITY = "quality"
    SYSTEM = "system"


class AlertState(str, Enum):
    """Lifecycle states for alerts.

    Values:
        ACTIVE: Alert is currently active
        ACKNOWLEDGED: User acknowledged the alert
        RESOLVED: Issue has been resolved
        SUPPRESSED: Alert is temporarily suppressed
    """
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Unified alert dataclass for all coordination/monitoring systems.

    Attributes:
        alert_id: Unique identifier for this alert
        title: Short, descriptive title
        message: Detailed alert message
        severity: Alert severity level
        category: Alert category for filtering
        state: Current alert lifecycle state
        source: Component that generated the alert
        timestamp: Unix timestamp when alert was created
        metadata: Additional context-specific data
        config_key: Optional board config (e.g., "hex8_2p")
        node_id: Optional node identifier
    """
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    category: AlertCategory = AlertCategory.SYSTEM
    state: AlertState = AlertState.ACTIVE
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    config_key: str = ""
    node_id: str = ""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def is_critical(self) -> bool:
        """Check if alert is critical severity."""
        return self.severity >= AlertSeverity.CRITICAL

    @property
    def is_error_or_above(self) -> bool:
        """Check if alert is error or critical."""
        return self.severity >= AlertSeverity.ERROR

    @property
    def age_seconds(self) -> float:
        """Seconds since alert was created."""
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.name,
            "severity_value": self.severity.value,
            "category": self.category.value,
            "state": self.state.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "config_key": self.config_key,
            "node_id": self.node_id,
        }


def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    category: AlertCategory = AlertCategory.SYSTEM,
    source: str = "",
    config_key: str = "",
    node_id: str = "",
    **metadata: Any,
) -> Alert:
    """Factory function to create an Alert with common defaults.

    Args:
        title: Short, descriptive title
        message: Detailed alert message
        severity: Alert severity level
        category: Alert category for filtering
        source: Component that generated the alert
        config_key: Optional board config
        node_id: Optional node identifier
        **metadata: Additional context-specific data

    Returns:
        Configured Alert instance
    """
    return Alert(
        title=title,
        message=message,
        severity=severity,
        category=category,
        source=source,
        config_key=config_key,
        node_id=node_id,
        metadata=dict(metadata),
    )


# ============================================================================
# Backward-Compatible Aliases
# ============================================================================
# These allow gradual migration from existing enums to the unified ones.

# From app/config/thresholds.py
AlertLevel = AlertSeverity  # AlertLevel.CRITICAL == AlertSeverity.CRITICAL


# From app/coordination/unified_health_manager.py
class ErrorSeverity(Enum):
    """Alias for backward compatibility with unified_health_manager.py."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def to_alert_severity(self) -> AlertSeverity:
        """Convert to unified AlertSeverity."""
        mapping = {
            ErrorSeverity.LOW: AlertSeverity.INFO,
            ErrorSeverity.MEDIUM: AlertSeverity.WARNING,
            ErrorSeverity.HIGH: AlertSeverity.ERROR,
            ErrorSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }
        return mapping.get(self, AlertSeverity.WARNING)


# From app/coordination/stall_detection.py
class StallSeverity(str, Enum):
    """Alias for backward compatibility with stall_detection.py."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def to_alert_severity(self) -> AlertSeverity:
        """Convert to unified AlertSeverity."""
        mapping = {
            StallSeverity.LOW: AlertSeverity.INFO,
            StallSeverity.MEDIUM: AlertSeverity.WARNING,
            StallSeverity.HIGH: AlertSeverity.ERROR,
            StallSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }
        return mapping.get(self, AlertSeverity.WARNING)


# From app/coordination/unified_replication_daemon.py
class ReplicationAlertLevel(str, Enum):
    """Alias for backward compatibility with unified_replication_daemon.py."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_alert_severity(self) -> AlertSeverity:
        """Convert to unified AlertSeverity."""
        mapping = {
            ReplicationAlertLevel.INFO: AlertSeverity.INFO,
            ReplicationAlertLevel.WARNING: AlertSeverity.WARNING,
            ReplicationAlertLevel.ERROR: AlertSeverity.ERROR,
            ReplicationAlertLevel.CRITICAL: AlertSeverity.CRITICAL,
        }
        return mapping.get(self, AlertSeverity.WARNING)


# From app/training/regression_detector.py
class RegressionSeverity(Enum):
    """Alias for backward compatibility with regression_detector.py."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def to_alert_severity(self) -> AlertSeverity:
        """Convert to unified AlertSeverity."""
        mapping = {
            RegressionSeverity.LOW: AlertSeverity.INFO,
            RegressionSeverity.MEDIUM: AlertSeverity.WARNING,
            RegressionSeverity.HIGH: AlertSeverity.ERROR,
            RegressionSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }
        return mapping.get(self, AlertSeverity.WARNING)


# From app/training/unified_data_validator.py
class ValidationSeverity(Enum):
    """Alias for backward compatibility with unified_data_validator.py."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_alert_severity(self) -> AlertSeverity:
        """Convert to unified AlertSeverity."""
        mapping = {
            ValidationSeverity.INFO: AlertSeverity.INFO,
            ValidationSeverity.WARNING: AlertSeverity.WARNING,
            ValidationSeverity.ERROR: AlertSeverity.ERROR,
            ValidationSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }
        return mapping.get(self, AlertSeverity.WARNING)


# ============================================================================
# Alert Routing Helpers
# ============================================================================

def severity_to_log_level(severity: AlertSeverity) -> str:
    """Convert AlertSeverity to Python logging level name.

    Args:
        severity: The alert severity

    Returns:
        Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    return severity.name


def severity_to_color(severity: AlertSeverity) -> str:
    """Get ANSI color code for severity.

    Args:
        severity: The alert severity

    Returns:
        ANSI color escape code
    """
    colors = {
        AlertSeverity.DEBUG: "\033[90m",     # Gray
        AlertSeverity.INFO: "\033[92m",      # Green
        AlertSeverity.WARNING: "\033[93m",   # Yellow
        AlertSeverity.ERROR: "\033[91m",     # Red
        AlertSeverity.CRITICAL: "\033[95m",  # Magenta
    }
    return colors.get(severity, "\033[0m")


def severity_to_emoji(severity: AlertSeverity) -> str:
    """Get emoji indicator for severity.

    Args:
        severity: The alert severity

    Returns:
        Emoji character
    """
    emojis = {
        AlertSeverity.DEBUG: "",
        AlertSeverity.INFO: "",
        AlertSeverity.WARNING: "",
        AlertSeverity.ERROR: "",
        AlertSeverity.CRITICAL: "",
    }
    return emojis.get(severity, "")


# ============================================================================
# Type Exports
# ============================================================================

__all__ = [
    # Core types
    "AlertSeverity",
    "AlertCategory",
    "AlertState",
    "Alert",
    # Factory
    "create_alert",
    # Backward-compatible aliases
    "AlertLevel",
    "ErrorSeverity",
    "StallSeverity",
    "ReplicationAlertLevel",
    "RegressionSeverity",
    "ValidationSeverity",
    # Helpers
    "severity_to_log_level",
    "severity_to_color",
    "severity_to_emoji",
]

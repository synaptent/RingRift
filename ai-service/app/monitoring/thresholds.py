"""Monitoring thresholds - Re-exports from canonical source.

DEPRECATED: Import directly from app.config.thresholds instead.

This module is maintained for backwards compatibility only.
All thresholds are now consolidated in app/config/thresholds.py.

Usage (preferred):
    from app.config.thresholds import THRESHOLDS, get_threshold, should_alert, AlertLevel

Usage (deprecated, still works):
    from app.monitoring.thresholds import THRESHOLDS, get_threshold, should_alert
"""

# Re-export everything from the canonical source
from app.config.thresholds import (
    # Alert levels
    AlertLevel,
    # Threshold dict
    THRESHOLDS,
    MONITORING_THRESHOLDS,
    # Helper functions
    get_threshold,
    should_alert,
    get_all_thresholds,
    update_threshold,
)

__all__ = [
    "AlertLevel",
    "THRESHOLDS",
    "MONITORING_THRESHOLDS",
    "get_threshold",
    "should_alert",
    "get_all_thresholds",
    "update_threshold",
]

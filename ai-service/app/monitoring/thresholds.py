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
    MONITORING_THRESHOLDS,
    # Threshold dict
    THRESHOLDS,
    # Alert levels
    AlertLevel,
    get_all_thresholds,
    # Helper functions
    get_threshold,
    should_alert,
    update_threshold,
)

__all__ = [
    "MONITORING_THRESHOLDS",
    "THRESHOLDS",
    "AlertLevel",
    "get_all_thresholds",
    "get_threshold",
    "should_alert",
    "update_threshold",
]

"""P2P Utility Classes and Functions.

This package contains utility classes and functions extracted from p2p_orchestrator.py.

January 2026: Initial extraction of WebhookNotifier (~175 LOC saved).
              Incorporated existing utils.py (systemd utilities, safe_json_response).
"""

# Re-export systemd utilities (original utils.py content)
from .systemd_utils import (
    HAS_SYSTEMD,
    SYSTEMD_NOTIFIER,
    safe_json_response,
    systemd_notify_ready,
    systemd_notify_watchdog,
)

# Re-export extracted classes
from .webhook_notifier import WebhookNotifier

__all__ = [
    # Systemd utilities
    "HAS_SYSTEMD",
    "SYSTEMD_NOTIFIER",
    "systemd_notify_watchdog",
    "systemd_notify_ready",
    "safe_json_response",
    # Extracted classes
    "WebhookNotifier",
]

"""P2P Orchestrator General Utilities.

This module contains general utility functions for the P2P orchestrator.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

# Systemd watchdog support for service health monitoring
# When running under systemd with WatchdogSec set, we need to periodically
# notify systemd that the service is healthy. If we miss the deadline,
# systemd will restart the service.
try:
    import sdnotify
    SYSTEMD_NOTIFIER = sdnotify.SystemdNotifier()
    HAS_SYSTEMD = True
except ImportError:
    SYSTEMD_NOTIFIER = None
    HAS_SYSTEMD = False


def systemd_notify_watchdog():
    """Send watchdog ping to systemd if available.

    Call this periodically (e.g., in your main loop) when running as a
    systemd service with WatchdogSec configured. If called too infrequently,
    systemd will assume the service is hung and restart it.
    """
    if HAS_SYSTEMD and SYSTEMD_NOTIFIER:
        try:
            SYSTEMD_NOTIFIER.notify("WATCHDOG=1")
        except Exception:
            pass  # Ignore errors - we may not be running under systemd


def systemd_notify_ready():
    """Notify systemd that the service is ready.

    Call this once after your service has completed initialization.
    This is required for services configured with Type=notify.
    """
    if HAS_SYSTEMD and SYSTEMD_NOTIFIER:
        try:
            SYSTEMD_NOTIFIER.notify("READY=1")
        except Exception:
            pass

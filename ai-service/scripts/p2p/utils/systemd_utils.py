"""P2P Orchestrator General Utilities.

This module contains general utility functions for the P2P orchestrator.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

logger = logging.getLogger(__name__)

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
        except (OSError, RuntimeError):
            pass  # Ignore errors - we may not be running under systemd


def systemd_notify_ready():
    """Notify systemd that the service is ready.

    Call this once after your service has completed initialization.
    This is required for services configured with Type=notify.
    """
    if HAS_SYSTEMD and SYSTEMD_NOTIFIER:
        with contextlib.suppress(Exception):
            SYSTEMD_NOTIFIER.notify("READY=1")


async def safe_json_response(
    resp: "aiohttp.ClientResponse",
    default: Any = None,
    log_errors: bool = True,
) -> Tuple[Any, Optional[str]]:
    """Safely parse JSON response with validation.

    Validates HTTP status, content headers, and body before parsing JSON.
    Prevents crashes from empty responses, malformed JSON, or HTTP errors.

    Args:
        resp: aiohttp ClientResponse object
        default: Default value to return on error
        log_errors: Whether to log validation errors

    Returns:
        (data, error) tuple where:
        - data: Parsed JSON or default value
        - error: Error message string or None if successful

    Example:
        async with session.get(url) as resp:
            data, error = await safe_json_response(resp)
            if error:
                logger.warning(f"Failed to get status: {error}")
                return None
            return data
    """
    # Check status code
    if resp.status != 200:
        error = f"http_{resp.status}"
        if log_errors:
            logger.debug(f"HTTP error: {resp.status}")
        return default, error

    # Check content length (if provided)
    content_length = resp.headers.get("Content-Length", "")
    if content_length == "0":
        if log_errors:
            logger.debug("Empty response (Content-Length: 0)")
        return default, "empty_response"

    # Read body
    try:
        body = await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        error = f"read_error: {type(e).__name__}"
        if log_errors:
            logger.debug(f"Failed to read response body: {e}")
        return default, error

    # Check for empty body
    if not body or body.strip() == "":
        if log_errors:
            logger.debug("Empty response body")
        return default, "empty_body"

    # Parse JSON
    try:
        data = json.loads(body)
        return data, None
    except json.JSONDecodeError as e:
        error = f"json_error: {e.msg}"
        if log_errors:
            logger.debug(f"JSON decode error: {e}")
        return default, error

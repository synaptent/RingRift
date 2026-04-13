"""Shared helpers for coordination runner modules."""

from __future__ import annotations

import asyncio
from typing import Any


async def wait_for_daemon(daemon: Any, check_interval: float = 10.0) -> None:
    """Wait for a daemon to complete or be stopped.

    Supports:
    - Daemons with ``is_running`` property (BaseDaemon pattern)
    - Daemons with ``is_running()`` method
    - Daemons with ``_running`` attribute (legacy pattern)
    """
    while True:
        if hasattr(daemon, "is_running"):
            attr = getattr(daemon, "is_running")
            running = attr() if callable(attr) else attr
        elif hasattr(daemon, "_running"):
            running = daemon._running
        else:
            running = False
        if not running:
            break
        await asyncio.sleep(check_interval)

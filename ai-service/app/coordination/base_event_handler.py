"""Base Event Handler - DEPRECATED re-export module.

.. deprecated:: December 2025
    This module is deprecated and only exists for backward compatibility.
    All functionality has been moved to ``app.coordination.handler_base``.

    Migration:

    .. code-block:: python

        # Old (deprecated)
        from app.coordination.base_event_handler import BaseEventHandler, EventHandlerConfig

        # New (canonical)
        from app.coordination.handler_base import HandlerBase, EventHandlerConfig
        # Or use the alias:
        from app.coordination.handler_base import BaseEventHandler, EventHandlerConfig

    This module will be archived in Q2 2026.

Original purpose: Provided standardized patterns for event subscription lifecycle,
error handling, health checks, and graceful shutdown. These patterns are now in
``handler_base.py`` with enhanced functionality.
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "app.coordination.base_event_handler is deprecated. "
    "Use app.coordination.handler_base instead. "
    "This module will be archived in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location for backward compatibility
from app.coordination.handler_base import (
    BaseEventHandler,
    EventHandlerConfig,
    HandlerBase,
    HandlerStats,
    HealthCheckResult,
    CoordinatorStatus,
    create_handler_stats,
    safe_subscribe,
)

__all__ = [
    # Primary re-exports (what this module originally provided)
    "BaseEventHandler",
    "EventHandlerConfig",
    # Additional re-exports for completeness
    "HandlerBase",
    "HandlerStats",
    "HealthCheckResult",
    "CoordinatorStatus",
    "create_handler_stats",
    "safe_subscribe",
]

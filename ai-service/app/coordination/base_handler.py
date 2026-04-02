"""Deprecated re-export for HandlerBase compatibility.

Import directly from ``app.coordination.handler_base`` instead.
This shim will be removed in Q2 2026.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.base_handler is deprecated. "
    "Import from app.coordination.handler_base instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*app\.coordination\.handler_base.*",
    )
    from app.coordination.handler_base import (
        BaseEventHandler,
        BaseSingletonHandler,
        CoordinatorStatus,
        EventHandlerConfig,
        HandlerBase,
        HandlerStats,
        HealthCheckResult,
        MultiEventHandler,
        SafeEventEmitterMixin,
        create_handler_stats,
        safe_subscribe,
    )

__all__ = [
    "BaseEventHandler",
    "BaseSingletonHandler",
    "CoordinatorStatus",
    "EventHandlerConfig",
    "HandlerBase",
    "HandlerStats",
    "HealthCheckResult",
    "MultiEventHandler",
    "SafeEventEmitterMixin",
    "create_handler_stats",
    "safe_subscribe",
]

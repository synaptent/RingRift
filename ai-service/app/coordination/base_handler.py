"""DEPRECATED: Base classes for event handlers in the coordination system.

.. deprecated:: December 2025
    This module is deprecated. Use ``app.coordination.handler_base`` instead:

    .. code-block:: python

        # Old (deprecated)
        from app.coordination.base_handler import BaseEventHandler, BaseSingletonHandler

        # New (canonical)
        from app.coordination.handler_base import HandlerBase

    This module was archived December 27, 2025. Only a backward-compatibility
    shim remains. For the original implementation, see:
    archive/deprecated_coordination/_deprecated_base_handler.py
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "app.coordination.base_handler is deprecated. "
    "Use app.coordination.handler_base.HandlerBase instead. "
    "This module was archived December 27, 2025.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location for backward compatibility
from app.coordination.handler_base import (
    HandlerBase,
    HandlerStats,
    EventHandlerConfig,
)

# Backward-compat aliases
BaseEventHandler = HandlerBase
BaseSingletonHandler = HandlerBase

__all__ = [
    "BaseEventHandler",
    "BaseSingletonHandler",
    "HandlerBase",
    "HandlerStats",
    "EventHandlerConfig",
]

"""DEPRECATED: Backward compatibility shim for base_handler.

This module was archived December 28, 2025. Use the canonical module:

    from app.coordination.handler_base import HandlerBase

For the original implementation, see:
archive/deprecated_coordination/_deprecated_base_handler.py
"""

import warnings

warnings.warn(
    "app.coordination.base_handler is deprecated. "
    "Use app.coordination.handler_base.HandlerBase instead. "
    "This module was archived December 28, 2025.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from archive for backward compatibility
from archive.deprecated_coordination._deprecated_base_handler import *
from archive.deprecated_coordination._deprecated_base_handler import __all__

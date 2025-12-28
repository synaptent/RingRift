"""Retry utilities - DEPRECATED.

This module is deprecated. Use app.utils.retry instead.

December 2025: Consolidated into app/utils/retry.py which provides:
- Sync and async retry decorators
- Common retry configurations (RETRY_QUICK, RETRY_STANDARD, etc.)
- RetryConfig dataclass with exponential backoff
- Jitter support for avoiding thundering herd

Migration:
    # Old (deprecated)
    from scripts.lib.retry import retry, RetryConfig

    # New (canonical)
    from app.utils.retry import retry, RetryConfig, RETRY_STANDARD

See app/utils/retry.py for full documentation.
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "scripts.lib.retry is deprecated. Use app.utils.retry instead. "
    "Removal: Q2 2026. See: app/utils/retry.py",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location for backward compatibility
from app.utils.retry import (
    RETRY_HTTP,
    RETRY_PATIENT,
    RETRY_QUICK,
    RETRY_SSH,
    RETRY_STANDARD,
    RetryAttempt,
    RetryConfig,
    retry,
    retry_async,
    retry_on_exception,
    retry_on_exception_async,
    with_timeout,
)

__all__ = [
    "RetryConfig",
    "RetryAttempt",
    "retry",
    "retry_async",
    "retry_on_exception",
    "retry_on_exception_async",
    "with_timeout",
    "RETRY_QUICK",
    "RETRY_STANDARD",
    "RETRY_PATIENT",
    "RETRY_SSH",
    "RETRY_HTTP",
]

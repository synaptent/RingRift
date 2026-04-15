"""Compatibility shim for deprecated EBMO AI.

Historically, EBMO_AI lived at `app.ai.ebmo_ai`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

This module provides a stable import surface for code that depends on EBMO.
Prefer using `app.ai.neural_net` for new code.
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "app.ai.ebmo_ai is deprecated and will be removed in Q2 2026. "
    "Use app.ai.neural_net for new code.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the deprecated implementation with validation
try:
    from app.ai._deprecated_ebmo_ai import (
        EBMO_AI,
        EBMOConfig,
    )
    __all__ = ["EBMO_AI", "EBMOConfig"]
except ImportError as e:
    raise ImportError(
        f"Failed to import deprecated EBMO implementation: {e}. "
        "The compatibility implementation now lives under app.ai._deprecated_ebmo_ai."
    ) from e

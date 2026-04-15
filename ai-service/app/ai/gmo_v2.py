"""Compatibility shim for deprecated GMO v2 AI.

Historically, GMOv2AI lived at `app.ai.gmo_v2`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

This module provides a stable import surface for code that depends on GMOv2.
Prefer using `app.ai.neural_net` for new code.
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "app.ai.gmo_v2 is deprecated and will be removed in Q2 2026. "
    "Use app.ai.neural_net for new code.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the deprecated implementation with validation
try:
    from app.ai._deprecated_gmo_v2 import (
        AttentionStateEncoder,
        GMOv2AI,
        GMOv2Config,
        GMOv2ValueNet,
        MoveEncoderV2,
        create_gmo_v2,
    )
    __all__ = [
        "AttentionStateEncoder",
        "GMOv2AI",
        "GMOv2Config",
        "GMOv2ValueNet",
        "MoveEncoderV2",
        "create_gmo_v2",
    ]
except ImportError as e:
    raise ImportError(
        f"Failed to import deprecated GMOv2 implementation: {e}. "
        "The compatibility implementation now lives under app.ai._deprecated_gmo_v2."
    ) from e

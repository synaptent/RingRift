"""Compatibility shim for deprecated GMO modules.

Historically, GMO lived at `app.ai.gmo_ai`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

Some tests and older tooling still import from the legacy location.
Keep this module as a thin re-export layer so imports remain stable.

Canonical engine/rules code does NOT depend on GMO.
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "app.ai.gmo_ai is deprecated and will be removed in Q2 2026. "
    "Use app.ai.neural_net for new code.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the deprecated implementation with validation
try:
    from app.ai._deprecated_gmo_ai import (
        GMOAI,
        GMOConfig,
        GMOValueNetWithUncertainty,
        MoveEncoder,
        NoveltyTracker,
        StateEncoder,
        estimate_uncertainty,
        nll_loss_with_uncertainty,
        optimize_move_with_entropy,
        project_to_legal_move,
    )
    __all__ = [
        "GMOAI",
        "GMOConfig",
        "GMOValueNetWithUncertainty",
        "MoveEncoder",
        "NoveltyTracker",
        "StateEncoder",
        "estimate_uncertainty",
        "nll_loss_with_uncertainty",
        "optimize_move_with_entropy",
        "project_to_legal_move",
    ]
except ImportError as e:
    raise ImportError(
        f"Failed to import deprecated GMO implementation: {e}. "
        "The compatibility implementation now lives under app.ai._deprecated_gmo_ai."
    ) from e

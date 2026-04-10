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
    from archive.deprecated_ai.gmo_ai import (
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
        f"Failed to import GMO from archive/deprecated_ai: {e}. "
        "The GMO implementation has been archived. If you need this module, "
        "ensure archive/deprecated_ai/gmo_ai.py exists."
    ) from e

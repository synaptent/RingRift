"""Legacy IG-GMO AI import surface.

Some callers (including the historical CLI and the unit tests) expect to be
able to construct the experimental IG-GMO engine via ``AIType.IG_GMO``.

The concrete implementation lives in the private compatibility module
``app.ai._deprecated_ig_gmo``. This shim preserves the newer import path
``app.ai.ig_gmo.IGGMO`` without keeping active imports pointed at ``archive/``.
"""

from __future__ import annotations

import warnings

from app.ai._deprecated_ig_gmo import (
    GNNStateEncoder as GNNStateEncoder,
    IGGMO as IGGMO,
    IGGMOConfig as IGGMOConfig,
    SoftLegalityPredictor as SoftLegalityPredictor,
)

warnings.warn(
    (
        "app.ai.ig_gmo is deprecated; use app.ai.neural_net "
        "(or newer policy nets) instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "GNNStateEncoder",
    "IGGMO",
    "IGGMOConfig",
    "SoftLegalityPredictor",
]

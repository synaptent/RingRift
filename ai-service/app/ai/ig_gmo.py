"""Legacy IG-GMO AI import surface.

Some callers (including the historical CLI and the unit tests) expect to be
able to construct the experimental IG-GMO engine via ``AIType.IG_GMO``.

The concrete implementation lives in the archived/deprecated module
[`archive.deprecated_ai.ig_gmo`](ai-service/archive/deprecated_ai/ig_gmo.py:1).
This shim preserves the newer import path ``app.ai.ig_gmo.IGGMO`` without
reintroducing legacy code into the main package.
"""

from __future__ import annotations

import warnings

from archive.deprecated_ai.ig_gmo import IGGMO as IGGMO  # re-export

warnings.warn(
    (
        "app.ai.ig_gmo is deprecated; use app.ai.neural_net "
        "(or newer policy nets) instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["IGGMO"]

from __future__ import annotations

import os

from .interfaces import RulesEngine
from .default_engine import DefaultRulesEngine

_ENGINE_SINGLETON: RulesEngine | None = None


def get_rules_engine(version: str | None = None) -> RulesEngine:
    """
    Return a RulesEngine instance.

    The optional `version` parameter selects among rule sets. For now only
    a single TS-parity implementation is available; the argument is kept
    for forward compatibility.
    """
    global _ENGINE_SINGLETON

    if _ENGINE_SINGLETON is not None:
        return _ENGINE_SINGLETON

    selected = version or os.getenv("RINGRIFT_RULES_VERSION", "v1")

    # In future we may dispatch on `selected`. For P0.5, all versions map
    # to the default TS-aligned engine.
    if selected == "v1":
        _ENGINE_SINGLETON = DefaultRulesEngine()
    else:
        _ENGINE_SINGLETON = DefaultRulesEngine()

    return _ENGINE_SINGLETON
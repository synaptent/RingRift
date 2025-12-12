from __future__ import annotations

import os

from .interfaces import RulesEngine
from .default_engine import DefaultRulesEngine

_ENGINE_SINGLETON: RulesEngine | None = None


def reset_rules_engine() -> None:
    """Reset the global RulesEngine singleton.

    This is primarily intended for tests and for training/evaluation harnesses
    that want to opt into different performance/debug settings (for example
    skipping mutator shadow contracts) without requiring a fresh process.
    """
    global _ENGINE_SINGLETON
    _ENGINE_SINGLETON = None


def get_rules_engine(
    version: str | None = None,
    *,
    force_new: bool = False,
    skip_shadow_contracts: bool | None = None,
    mutator_first: bool | None = None,
) -> RulesEngine:
    """
    Return a RulesEngine instance.

    The optional `version` parameter selects among rule sets. For now only
    a single TS-parity implementation is available; the argument is kept
    for forward compatibility.
    """
    global _ENGINE_SINGLETON

    if force_new:
        _ENGINE_SINGLETON = None

    if _ENGINE_SINGLETON is not None:
        return _ENGINE_SINGLETON

    selected = version or os.getenv("RINGRIFT_RULES_VERSION", "v1")

    # In future we may dispatch on `selected`. For P0.5, all versions map
    # to the default TS-aligned engine.
    if selected == "v1":
        _ENGINE_SINGLETON = DefaultRulesEngine(
            mutator_first=mutator_first,
            skip_shadow_contracts=skip_shadow_contracts,
        )
    else:
        _ENGINE_SINGLETON = DefaultRulesEngine(
            mutator_first=mutator_first,
            skip_shadow_contracts=skip_shadow_contracts,
        )

    return _ENGINE_SINGLETON

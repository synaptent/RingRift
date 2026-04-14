"""Rules engine abstraction layer for the AI service.

This package exposes a versioned RulesEngine interface that wraps the
concrete GameEngine implementation, allowing AI strategies to depend
on a stable rules API rather than on internal engine details.

Exports:
    RulesEngine: Abstract rules interface
    get_rules_engine: Factory function to get rules engine instance
    MutableGameState: Efficient mutable state for make/unmake move pattern
    create_game_state: Factory function to create GameState instances
"""

from __future__ import annotations

import importlib

__all__ = [
    "RulesEngine",
    "get_rules_engine",
    "MutableGameState",
    "create_game_state",
]


def __dir__() -> list[str]:
    """Expose the intended rules facade surface for discoverability."""

    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str):
    if name == "get_rules_engine":
        return importlib.import_module("app.rules.factory").get_rules_engine
    if name == "RulesEngine":
        return importlib.import_module("app.rules.interfaces").RulesEngine
    if name == "MutableGameState":
        return importlib.import_module("app.rules.mutable_state").MutableGameState
    if name == "create_game_state":
        return importlib.import_module("app.testing.fixtures").create_game_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

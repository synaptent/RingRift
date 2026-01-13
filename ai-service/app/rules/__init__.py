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

from .factory import get_rules_engine
from .interfaces import RulesEngine  # re-export for convenience
from .mutable_state import MutableGameState
from app.testing.fixtures import create_game_state

__all__ = [
    "RulesEngine",
    "get_rules_engine",
    "MutableGameState",
    "create_game_state",
]

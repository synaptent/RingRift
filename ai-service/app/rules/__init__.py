"""Rules engine abstraction layer for the AI service.

This package exposes a versioned RulesEngine interface that wraps the
concrete GameEngine implementation, allowing AI strategies to depend
on a stable rules API rather than on internal engine details.
"""

from .factory import get_rules_engine
from .interfaces import RulesEngine  # re-export for convenience

__all__ = ["RulesEngine", "get_rules_engine"]

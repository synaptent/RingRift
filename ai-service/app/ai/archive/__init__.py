"""Archived AI implementations.

This package contains experimental AI implementations that are no longer
actively maintained but preserved for reference and potential feature
harvesting.

WARNING: These modules should not be used in new code. They are kept only
for backward compatibility with existing configurations that reference them.

Archived on: 2025-12-21
Updated: 2025-12-21 - GMOMCTSHybrid unarchived to app/ai/gmo_mcts_hybrid.py

For details on each archived module and its valuable patterns, see README.md.
"""

# Lazy imports to avoid loading archived code unless explicitly requested
__all__ = [
    "GMOMCTSHybrid",  # Re-exported from main ai module (unarchived)
    "GMOMCTSConfig",  # Re-exported from main ai module (unarchived)
    "CAGE_AI",
    "CAGEConfig",
    "EBMOOnlineAI",
    "EBMOOnlineConfig",
    "EBMOOnlineLearner",
]


def __getattr__(name: str):
    """Lazy import archived modules only when accessed."""
    if name in ("GMOMCTSHybrid", "GMOMCTSConfig"):
        # Unarchived 2025-12-21 - redirect to main ai module
        from app.ai.gmo_mcts_hybrid import GMOMCTSConfig, GMOMCTSHybrid
        return GMOMCTSHybrid if name == "GMOMCTSHybrid" else GMOMCTSConfig
    elif name in ("CAGE_AI", "CAGEConfig"):
        from .cage_ai import CAGE_AI
        from .cage_network import CAGEConfig
        return CAGE_AI if name == "CAGE_AI" else CAGEConfig
    elif name in ("EBMOOnlineAI", "EBMOOnlineConfig", "EBMOOnlineLearner"):
        from .ebmo_online import EBMOOnlineAI, EBMOOnlineConfig, EBMOOnlineLearner
        if name == "EBMOOnlineAI":
            return EBMOOnlineAI
        elif name == "EBMOOnlineConfig":
            return EBMOOnlineConfig
        else:
            return EBMOOnlineLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

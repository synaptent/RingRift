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


def __dir__() -> list[str]:
    """Expose the intended archived AI surface for discoverability."""

    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str) -> type:
    """Lazy import archived modules only when accessed.

    Returns:
        The requested archived class type.

    Raises:
        AttributeError: If the attribute is not a known archived class.
    """
    if name in ("GMOMCTSHybrid", "GMOMCTSConfig"):
        # Unarchived 2025-12-21 - redirect to main ai module
        from app.ai.gmo_mcts_hybrid import GMOMCTSConfig, GMOMCTSHybrid
        return GMOMCTSHybrid if name == "GMOMCTSHybrid" else GMOMCTSConfig
    elif name in ("CAGE_AI", "CAGEConfig"):
        from .cage_ai import CAGE_AI
        from .cage_network import CAGEConfig
        return CAGE_AI if name == "CAGE_AI" else CAGEConfig
    elif name in ("EBMOOnlineAI", "EBMOOnlineConfig", "EBMOOnlineLearner"):
        # Archived facade retained, but the canonical implementation now lives
        # in app.ai.ebmo_online_learner.
        from app.ai.ebmo_online_learner import (
            EBMOOnlineAI,
            EBMOOnlineConfig,
            EBMOOnlineLearner,
        )
        if name == "EBMOOnlineAI":
            return EBMOOnlineAI
        elif name == "EBMOOnlineConfig":
            return EBMOOnlineConfig
        else:
            return EBMOOnlineLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

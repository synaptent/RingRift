"""AI implementations for RingRift.

This package intentionally avoids importing concrete AI classes at import
 time to prevent circular dependencies between the rules engine and AI
 modules. Callers should import specific implementations directly, e.g.:

    from app.ai.base import BaseAI
    from app.ai.heuristic_ai import HeuristicAI

rather than importing from the package root.
"""

__all__: list[str] = []

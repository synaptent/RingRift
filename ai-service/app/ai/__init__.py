"""AI implementations for RingRift.

This package provides AI engines for playing RingRift. The recommended
approach is to use the factory module for creating AI instances:

    from app.ai import create_ai_from_difficulty, AIType

    # Create from difficulty level (recommended for gameplay)
    ai = create_ai_from_difficulty(difficulty=5, player_number=1)

    # Create with explicit type
    ai = AIFactory.create(AIType.MCTS, player_number=1, config=config)

For direct access to AI classes (advanced usage):

    from app.ai import BaseAI, HeuristicAI, MCTSAI, DescentAI

Architecture (December 2025):
- base.py: BaseAI abstract base class
- factory.py: AIFactory for creating AI instances
- heuristic_ai.py: Fast rule-based evaluation
- mcts_ai.py: Monte Carlo Tree Search
- descent_ai.py: Neural network + MCTS hybrid
- gumbel_mcts_ai.py: Gumbel-Top-k MCTS (AlphaZero-style)
- gmo_ai.py: Gradient Move Optimization (experimental)
- ig_gmo.py: Information-Gain GMO (research)
- neural_net.py: Neural network models
- nnue.py: NNUE evaluation (incrementally updated)
"""

# Factory is safe to import (uses lazy loading internally)
# Import base class (no circular dependency issues)
from app.ai.base import BaseAI
from app.ai.factory import (
    CANONICAL_DIFFICULTY_PROFILES,
    DIFFICULTY_DESCRIPTIONS,
    AIFactory,
    AIType,
    DifficultyProfile,
    create_ai,
    create_ai_from_difficulty,
    create_tournament_ai,
    get_all_difficulties,
    get_difficulty_description,
    get_difficulty_profile,
    get_randomness_for_difficulty,
    get_think_time_for_difficulty,
    select_ai_type,
    uses_neural_net,
)

# Lazy-load AI implementations to avoid circular imports
_AI_CLASSES = {
    "HeuristicAI": "app.ai.heuristic_ai",
    "MCTSAI": "app.ai.mcts_ai",
    "DescentAI": "app.ai.descent_ai",
    "GumbelMCTSAI": "app.ai.gumbel_mcts_ai",
    "MaxNAI": "app.ai.maxn_ai",
    "MinimaxAI": "app.ai.minimax_ai",
    "RandomAI": "app.ai.random_ai",
    "PolicyOnlyAI": "app.ai.policy_only_ai",
    "GMOAI": "app.ai.gmo_ai",
    "EBMOAI": "app.ai.ebmo_ai",
    "IGGMO": "app.ai.ig_gmo",
}


def __getattr__(name: str):
    """Lazy loading for AI implementation classes."""
    if name in _AI_CLASSES:
        import importlib
        module = importlib.import_module(_AI_CLASSES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Profile data
    "CANONICAL_DIFFICULTY_PROFILES",
    "DIFFICULTY_DESCRIPTIONS",
    "EBMOAI",
    "GMOAI",
    "IGGMO",
    "MCTSAI",
    # Factory class
    "AIFactory",
    # AI Type enum
    "AIType",
    # Base class
    "BaseAI",
    "DescentAI",
    # Type definitions
    "DifficultyProfile",
    "GumbelMCTSAI",
    # AI implementation classes (lazy-loaded)
    "HeuristicAI",
    "MaxNAI",
    "MinimaxAI",
    "PolicyOnlyAI",
    "RandomAI",
    # Convenience aliases
    "create_ai",
    "create_ai_from_difficulty",
    "create_tournament_ai",
    "get_all_difficulties",
    "get_difficulty_description",
    # Helper functions
    "get_difficulty_profile",
    "get_randomness_for_difficulty",
    "get_think_time_for_difficulty",
    "select_ai_type",
    "uses_neural_net",
]

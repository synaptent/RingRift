"""
Feature evaluators for HeuristicAI decomposition.

This package contains specialized evaluator classes that implement focused
subsets of the evaluation logic previously contained in the monolithic
HeuristicAI class. Each evaluator focuses on a specific domain of game
state evaluation.

Available Evaluators:
- MaterialEvaluator: Ring counts, stack heights, material balance
- PositionalEvaluator: Territory control, center positions, connectivity
- TacticalEvaluator: Captures, threats, vulnerability
- MobilityEvaluator: Movement options, pseudo-mobility, stack mobility
- StrategicEvaluator: Victory proximity, forced elimination risk, LPS
- EndgameEvaluator: Terminal detection, game phase, recovery potential

Future Evaluators (planned):
- LineStructureEvaluator: Line potential, marker connectivity
"""

from .endgame_evaluator import (
    EndgameEvaluator,
    EndgameScore,
    EndgameWeights,
)
from .material_evaluator import (
    MaterialEvaluator,
    MaterialScore,
    MaterialWeights,
)
from .mobility_evaluator import (
    MobilityEvaluator,
    MobilityScore,
    MobilityWeights,
)
from .positional_evaluator import (
    PositionalEvaluator,
    PositionalScore,
    PositionalWeights,
)
from .strategic_evaluator import (
    StrategicEvaluator,
    StrategicScore,
    StrategicWeights,
)
from .tactical_evaluator import (
    TacticalEvaluator,
    TacticalScore,
    TacticalWeights,
)

__all__ = [
    "EndgameEvaluator",
    "EndgameScore",
    "EndgameWeights",
    "MaterialEvaluator",
    "MaterialScore",
    "MaterialWeights",
    "MobilityEvaluator",
    "MobilityScore",
    "MobilityWeights",
    "PositionalEvaluator",
    "PositionalScore",
    "PositionalWeights",
    "StrategicEvaluator",
    "StrategicScore",
    "StrategicWeights",
    "TacticalEvaluator",
    "TacticalScore",
    "TacticalWeights",
]


def __dir__() -> list[str]:
    """Expose the intended evaluator surface for discoverability."""

    return sorted(set(globals()) | set(__all__))

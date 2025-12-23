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

Future Evaluators (planned):
- StrategicEvaluator: Victory proximity, forced elimination risk
- MobilityEvaluator: Movement options
- LineStructureEvaluator: Line potential, marker connectivity
- RecoveryEvaluator: Recovery mechanics (RR-CANON-R110-R115)
"""

from .material_evaluator import (
    MaterialEvaluator,
    MaterialScore,
    MaterialWeights,
)
from .positional_evaluator import (
    PositionalEvaluator,
    PositionalScore,
    PositionalWeights,
)
from .tactical_evaluator import (
    TacticalEvaluator,
    TacticalScore,
    TacticalWeights,
)

__all__ = [
    "MaterialEvaluator",
    "MaterialScore",
    "MaterialWeights",
    "PositionalEvaluator",
    "PositionalScore",
    "PositionalWeights",
    "TacticalEvaluator",
    "TacticalScore",
    "TacticalWeights",
]

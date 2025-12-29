"""Environment mixins for RingRiftEnv decomposition.

This package contains mixins that extract logical groupings of functionality
from the monolithic RingRiftEnv class to improve testability and maintainability.

Mixins:
    RewardCalculatorMixin: Terminal and shaped reward calculation
    TerminationHandlerMixin: Game termination detection and info population
    BookkeepingMoveHandlerMixin: Auto-bookkeeping move generation (RR-CANON-R075/R076)
    MoveGenerationMixin: Legal move surface with phase requirement handling
"""

from app.training.env_mixins.bookkeeping_handler import BookkeepingMoveHandlerMixin
from app.training.env_mixins.move_generation import MoveGenerationMixin
from app.training.env_mixins.reward_calculator import RewardCalculatorMixin
from app.training.env_mixins.termination_handler import TerminationHandlerMixin

__all__ = [
    "BookkeepingMoveHandlerMixin",
    "MoveGenerationMixin",
    "RewardCalculatorMixin",
    "TerminationHandlerMixin",
]

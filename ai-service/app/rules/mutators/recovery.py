"""Recovery action mutator.

Applies recovery_slide moves per RR-CANON-R110–R115.

**SSoT Policy:** This mutator must mirror the canonical recovery rules
defined in RULES_CANONICAL_SPEC.md §5.4. If this code disagrees with
the canonical rules or the TS implementation, this code must be updated.
"""

from app.models import GameState, Move
from app.rules.interfaces import Mutator
from app.rules.recovery import apply_recovery_slide


class RecoveryMutator(Mutator):
    """Mutator for recovery_slide moves."""

    def apply(self, state: GameState, move: Move) -> None:
        """
        Apply a recovery slide move to the state in-place.

        Steps:
        1. Move the marker from from_pos to to_pos
        2. Extract buried rings (cost = 1 + overlength)
        3. Return extracted rings to player's hand

        The line collapse and territory processing are handled by
        the turn orchestrator after the move is applied.
        """
        outcome = apply_recovery_slide(state, move)
        if not outcome.success:
            raise ValueError(f"Recovery slide failed: {outcome.error}")

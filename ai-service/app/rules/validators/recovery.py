"""Recovery action validator.

Validates recovery_slide moves per RR-CANON-R110–R115.

**SSoT Policy:** This validator must mirror the canonical recovery rules
defined in RULES_CANONICAL_SPEC.md §5.4. If this code disagrees with
the canonical rules or the TS implementation, this code must be updated.
"""

from app.models import GameState, Move, GamePhase, MoveType
from app.rules.interfaces import Validator
from app.rules.recovery import validate_recovery_slide


class RecoveryValidator(Validator):
    """Validator for recovery_slide moves."""

    def validate(self, state: GameState, move: Move) -> bool:
        """
        Validate a recovery slide move.

        Checks:
        1. Move type is RECOVERY_SLIDE
        2. Current phase is MOVEMENT
        3. Player matches current player
        4. Delegated validation via recovery module
        """
        # 1. Type Check
        if move.type != MoveType.RECOVERY_SLIDE:
            return False

        # 2. Phase Check - recovery only valid in MOVEMENT phase
        if state.current_phase != GamePhase.MOVEMENT:
            return False

        # 3. Turn Check
        if move.player != state.current_player:
            return False

        # 4. Delegate to canonical recovery validation
        result = validate_recovery_slide(state, move)
        return result.valid

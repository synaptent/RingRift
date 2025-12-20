"""Phase requirement types for the RingRift game engine.

Per RR-CANON-R076, the core rules layer MUST NOT auto-generate moves
(including no-action moves). Instead, it surfaces "phase requirements"
that tell hosts what bookkeeping move they must emit. Hosts are responsible
for constructing and applying the actual Move objects.
"""

from dataclasses import dataclass
from enum import Enum

from app.models import Position


class PhaseRequirementType(Enum):
    """Types of phase requirements that hosts must handle."""

    NO_PLACEMENT_ACTION_REQUIRED = "no_placement_action_required"
    NO_MOVEMENT_ACTION_REQUIRED = "no_movement_action_required"
    NO_LINE_ACTION_REQUIRED = "no_line_action_required"
    NO_TERRITORY_ACTION_REQUIRED = "no_territory_action_required"
    FORCED_ELIMINATION_REQUIRED = "forced_elimination_required"


@dataclass
class PhaseRequirement:
    """Structural data telling hosts what bookkeeping move is required.

    When get_valid_moves returns an empty list but the game is still ACTIVE,
    hosts should call get_phase_requirement to determine what no-action or
    forced-elimination move must be emitted to satisfy canonical phase rules.

    Attributes:
        type: The type of phase requirement.
        player: The player who must emit the bookkeeping move.
        eligible_positions: For FORCED_ELIMINATION_REQUIRED, the stack
            positions the player can eliminate from. Empty for no-action types.
    """

    type: PhaseRequirementType
    player: int
    eligible_positions: list[Position]

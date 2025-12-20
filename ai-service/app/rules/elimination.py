"""
EliminationAggregate - Canonical elimination logic for RingRift.

This is the SINGLE SOURCE OF TRUTH for all elimination semantics in Python.
All other code should delegate to this module.

Mirrors: src/shared/engine/aggregates/EliminationAggregate.ts

Canonical Rules (from RULES_CANONICAL_SPEC.md):

| Context              | Cost                    | Eligible Stacks                              | Reference      |
|----------------------|-------------------------|----------------------------------------------|----------------|
| Line Processing      | 1 ring from top         | Any controlled stack (including height-1)    | RR-CANON-R122  |
| Territory Processing | Entire cap              | Any controlled stack (including height-1)    | RR-CANON-R145  |
| Forced Elimination   | Entire cap              | Any controlled stack (including height-1)    | RR-CANON-R100  |
| Recovery Action      | 1 buried ring extraction| Any stack with player's buried ring          | RR-CANON-R113  |

All controlled stacks (including height-1 standalone rings) are eligible for
line, territory, and forced elimination. Recovery uses buried ring extraction.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EliminationContext(Enum):
    """
    Context in which elimination occurs. Determines cost and eligibility.

    - LINE: Line processing reward (RR-CANON-R122) - 1 ring, any stack
    - TERRITORY: Territory self-elimination (RR-CANON-R145) - entire cap, eligible stacks only
    - FORCED: Forced elimination when no moves (RR-CANON-R100) - entire cap, any stack
    - RECOVERY: Recovery buried ring extraction (RR-CANON-R113) - 1 buried ring
    """

    LINE = "line"
    TERRITORY = "territory"
    FORCED = "forced"
    RECOVERY = "recovery"


class EliminationReason(Enum):
    """Reason for elimination - used for audit trail and debugging."""

    LINE_REWARD_OPTION1 = "line_reward_option1"  # Line collapse Option 1
    LINE_REWARD_EXACT = "line_reward_exact"  # Exact-length line
    TERRITORY_SELF_ELIMINATION = "territory_self_elimination"  # Territory processing
    FORCED_ELIMINATION_ANM = "forced_elimination_anm"  # Forced elimination due to ANM
    RECOVERY_BURIED_EXTRACTION = "recovery_buried_extraction"  # Recovery action
    CAPTURE_OVERTAKE = "capture_overtake"  # Capture that results in ring elimination


@dataclass
class StackEligibility:
    """Stack eligibility result with explanation."""

    eligible: bool
    reason: str


@dataclass
class EliminationAuditEvent:
    """Audit event for tracking elimination operations."""

    timestamp: datetime
    context: EliminationContext
    reason: EliminationReason | None
    player: int
    stack_position: tuple
    rings_eliminated: int
    stack_height_before: int
    stack_height_after: int
    cap_height_before: int
    controlling_player_before: int
    controlling_player_after: int | None


@dataclass
class EliminationResult:
    """Result of an elimination operation."""

    success: bool
    rings_eliminated: int
    updated_stack: list[int] | None  # None if stack removed
    error: str | None = None
    audit_event: EliminationAuditEvent | None = None


def calculate_cap_height(rings: list[int]) -> int:
    """
    Calculate cap height - number of consecutive rings from top belonging to controlling player.

    Note: Python uses reversed ring order (bottom-to-top) while TypeScript uses top-to-bottom.
    rings[-1] is the top ring in Python.

    Args:
        rings: Array of player numbers representing the stack (index 0 = bottom, -1 = top)

    Returns:
        Cap height (0 if stack is empty)
    """
    if not rings:
        return 0

    top_player = rings[-1]
    cap_height = 0
    for ring in reversed(rings):
        if ring == top_player:
            cap_height += 1
        else:
            break
    return cap_height


def is_stack_eligible_for_elimination(
    rings: list[int],
    controlling_player: int,
    context: EliminationContext,
    player: int,
) -> StackEligibility:
    """
    Check if a stack is eligible for elimination in a given context.

    Eligibility rules per canonical spec:
    - Line (RR-CANON-R122): Any controlled stack, including height-1
    - Territory (RR-CANON-R145): Any controlled stack, including height-1
    - Forced (RR-CANON-R100): Any controlled stack, including height-1
    - Recovery (RR-CANON-R113): Any stack containing player's buried ring

    Args:
        rings: Stack rings (bottom to top)
        controlling_player: Player who controls the stack
        context: Elimination context
        player: Player performing elimination

    Returns:
        StackEligibility with eligible flag and reason
    """
    # Recovery has special rules - checks for buried rings, not control
    if context == EliminationContext.RECOVERY:
        # Find any buried ring belonging to player (not including top)
        has_buried_ring = any(ring == player for ring in rings[:-1]) if len(rings) > 1 else False
        if not has_buried_ring:
            return StackEligibility(eligible=False, reason="No buried rings of player in stack")
        return StackEligibility(eligible=True, reason="Stack contains buried ring of player")

    # For line/territory/forced: must control the stack
    if controlling_player != player:
        return StackEligibility(eligible=False, reason="Player does not control stack")

    cap_height = calculate_cap_height(rings)
    if cap_height <= 0:
        return StackEligibility(eligible=False, reason="No cap to eliminate")

    # Line, Territory, and Forced: any controlled stack is eligible (including height-1)
    # Per RR-CANON-R022, RR-CANON-R122, RR-CANON-R145, RR-CANON-R100
    return StackEligibility(eligible=True, reason=f"{context.value} allows any controlled stack")


def get_rings_to_eliminate(rings: list[int], context: EliminationContext) -> int:
    """
    Calculate how many rings to eliminate based on context.

    Args:
        rings: Stack rings (bottom to top)
        context: Elimination context

    Returns:
        Number of rings to eliminate
    """
    cap_height = calculate_cap_height(rings)

    if context == EliminationContext.LINE:
        # Line processing: always 1 ring (RR-CANON-R122)
        return 1

    if context in (EliminationContext.TERRITORY, EliminationContext.FORCED):
        # Territory and Forced: entire cap (RR-CANON-R145, RR-CANON-R100)
        return cap_height

    if context == EliminationContext.RECOVERY:
        # Recovery: 1 buried ring extraction (RR-CANON-R113)
        return 1

    return cap_height


def eliminate_from_stack(
    rings: list[int],
    controlling_player: int,
    context: EliminationContext,
    player: int,
    reason: EliminationReason | None = None,
    buried_ring_index: int | None = None,
    stack_position: tuple | None = None,
) -> EliminationResult:
    """
    Perform elimination from a stack.

    This is the CANONICAL elimination function. All elimination operations
    should go through this function to ensure consistent semantics.

    Args:
        rings: Stack rings (bottom to top)
        controlling_player: Player who controls the stack
        context: Elimination context
        player: Player performing the elimination
        reason: Optional reason for audit trail
        buried_ring_index: For recovery: index of buried ring to extract
        stack_position: Position of the stack (for audit)

    Returns:
        EliminationResult with success status and updated stack
    """
    if not rings:
        return EliminationResult(
            success=False,
            rings_eliminated=0,
            updated_stack=None,
            error="No rings in stack",
        )

    # Check eligibility
    eligibility = is_stack_eligible_for_elimination(rings, controlling_player, context, player)
    if not eligibility.eligible:
        return EliminationResult(
            success=False,
            rings_eliminated=0,
            updated_stack=list(rings),
            error=eligibility.reason,
        )

    # Determine rings to eliminate
    if context == EliminationContext.RECOVERY:
        # Recovery: extract bottommost buried ring of player
        rings_to_eliminate = 1
        # Find the bottommost buried ring of player (not including top)
        extract_index = buried_ring_index
        if extract_index is None:
            for i in range(len(rings) - 1):  # Exclude top ring
                if rings[i] == player:
                    extract_index = i
                    break

        if extract_index is None or extract_index >= len(rings) - 1 or rings[extract_index] != player:
            return EliminationResult(
                success=False,
                rings_eliminated=0,
                updated_stack=list(rings),
                error="No valid buried ring to extract",
            )

        # Remove the buried ring, keeping everything else
        remaining_rings = list(rings[:extract_index]) + list(rings[extract_index + 1 :])
    else:
        # Line/Territory/Forced: remove from top
        rings_to_eliminate = get_rings_to_eliminate(rings, context)
        remaining_rings = list(rings[:-rings_to_eliminate]) if rings_to_eliminate < len(rings) else []

    # Create audit event
    cap_height_before = calculate_cap_height(rings)
    stack_height_before = len(rings)
    stack_height_after = len(remaining_rings)

    controlling_player_after = remaining_rings[-1] if remaining_rings else None

    audit_event = EliminationAuditEvent(
        timestamp=datetime.now(),
        context=context,
        reason=reason,
        player=player,
        stack_position=stack_position or (0, 0),
        rings_eliminated=rings_to_eliminate,
        stack_height_before=stack_height_before,
        stack_height_after=stack_height_after,
        cap_height_before=cap_height_before,
        controlling_player_before=controlling_player,
        controlling_player_after=controlling_player_after,
    )

    return EliminationResult(
        success=True,
        rings_eliminated=rings_to_eliminate,
        updated_stack=remaining_rings if remaining_rings else None,
        audit_event=audit_event,
    )


def enumerate_eligible_stacks(
    stacks: dict[tuple, dict[str, Any]],
    player: int,
    context: EliminationContext,
    exclude_positions: set | None = None,
) -> list[tuple]:
    """
    Enumerate all eligible stack positions for elimination in a given context.

    Args:
        stacks: Dictionary of position -> stack data (with 'rings' and 'controlling_player')
        player: Player performing elimination
        context: Elimination context
        exclude_positions: Positions to exclude (e.g., region being processed)

    Returns:
        List of eligible stack positions
    """
    eligible = []
    exclude = exclude_positions or set()

    for pos, stack_data in stacks.items():
        if pos in exclude:
            continue

        rings = stack_data.get("rings", [])
        controlling_player = stack_data.get("controlling_player", 0)

        eligibility = is_stack_eligible_for_elimination(rings, controlling_player, context, player)
        if eligibility.eligible:
            eligible.append(pos)

    return eligible


def has_eligible_elimination_target(
    stacks: dict[tuple, dict[str, Any]],
    player: int,
    context: EliminationContext,
    exclude_positions: set | None = None,
) -> bool:
    """
    Check if player has any eligible elimination targets for a given context.

    Args:
        stacks: Dictionary of position -> stack data
        player: Player to check
        context: Elimination context
        exclude_positions: Positions to exclude

    Returns:
        True if player has at least one eligible target
    """
    exclude = exclude_positions or set()

    for pos, stack_data in stacks.items():
        if pos in exclude:
            continue

        rings = stack_data.get("rings", [])
        controlling_player = stack_data.get("controlling_player", 0)

        eligibility = is_stack_eligible_for_elimination(rings, controlling_player, context, player)
        if eligibility.eligible:
            return True

    return False

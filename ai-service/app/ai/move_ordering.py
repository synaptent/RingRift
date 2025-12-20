"""
Move Ordering Heuristics for RingRift AI.

This module provides move ordering utilities for alpha-beta search algorithms.
Good move ordering improves pruning efficiency by examining likely-best moves first.

The module provides:
- `MoveTypePriority` - Standard priority values for move types
- `MovePriorityScorer` - Computes priority scores for moves
- `KillerMoveTable` - Tracks killer moves (refutation moves) at each depth
- `order_moves()` - Main function to order moves for search

Usage Example:
```python
from app.ai.move_ordering import MovePriorityScorer, KillerMoveTable, order_moves

scorer = MovePriorityScorer()
killer_table = KillerMoveTable()

# Order moves for search
ordered = order_moves(
    moves,
    depth=3,
    killer_table=killer_table,
    scorer=scorer,
)

# After finding a cutoff, store the killer move
killer_table.store(best_move, depth)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Move


class MoveTypePriority(IntEnum):
    """Priority values for different move types.

    Higher values indicate moves that should be searched first.
    These priorities are tuned for RingRift's game mechanics where:
    - Territory claims end the game immediately if threshold is met
    - Line formations can trigger elimination cascades
    - Captures are tactical forcing moves
    - Regular moves are positional
    """

    TERRITORY_CLAIM = 5
    LINE_FORMATION = 4
    RECOVERY_SLIDE = 3
    CHAIN_CAPTURE = 2
    OVERTAKING_CAPTURE = 1
    MOVE_STACK = 0
    PLACE_RING = 0
    SKIP_PLACEMENT = -1
    NO_ACTION = -2
    DEFAULT = 0


# Bonus multipliers for scored move ordering
PRIORITY_BONUS_MULTIPLIER = {
    "territory_claim": 10000.0,
    "line_formation": 5000.0,
    "chain_capture": 2000.0,
    "overtaking_capture": 1000.0,
    "recovery_slide": 500.0,
    "move_stack": 0.0,
    "place_ring": 0.0,
}


@dataclass
class MovePriorityScorer:
    """Computes priority scores for moves.

    The scorer supports multiple modes:
    - Simple priority: Returns integer priority based on move type
    - Bonus priority: Returns float bonus suitable for adding to evaluation scores
    - Custom priorities: Override default priorities for specific use cases

    Attributes
    ----------
    custom_priorities : Dict[str, int]
        Custom priority overrides by move type string.
    bonus_multipliers : Dict[str, float]
        Bonus multipliers for scored ordering.
    """

    custom_priorities: dict[str, int] = field(default_factory=dict)
    bonus_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(PRIORITY_BONUS_MULTIPLIER)
    )

    def get_priority(self, move: Move) -> int:
        """Get integer priority for a move.

        Parameters
        ----------
        move : Move
            The move to score.

        Returns
        -------
        int
            Priority value (higher = search first).
        """
        move_type = str(move.type.value if hasattr(move.type, "value") else move.type)

        # Check custom overrides first
        if move_type in self.custom_priorities:
            return self.custom_priorities[move_type]

        # Map to enum
        type_map = {
            "territory_claim": MoveTypePriority.TERRITORY_CLAIM,
            "line_formation": MoveTypePriority.LINE_FORMATION,
            "recovery_slide": MoveTypePriority.RECOVERY_SLIDE,
            "chain_capture": MoveTypePriority.CHAIN_CAPTURE,
            "overtaking_capture": MoveTypePriority.OVERTAKING_CAPTURE,
            "move_stack": MoveTypePriority.MOVE_STACK,
            "place_ring": MoveTypePriority.PLACE_RING,
            "skip_placement": MoveTypePriority.SKIP_PLACEMENT,
            "no_line_action": MoveTypePriority.NO_ACTION,
            "no_territory_action": MoveTypePriority.NO_ACTION,
        }

        return type_map.get(move_type, MoveTypePriority.DEFAULT)

    def get_bonus(self, move: Move) -> float:
        """Get float bonus for scored move ordering.

        This bonus is suitable for adding to evaluation scores to bias
        move ordering in favor of tactical moves.

        Parameters
        ----------
        move : Move
            The move to score.

        Returns
        -------
        float
            Bonus value to add to evaluation score.
        """
        move_type = str(move.type.value if hasattr(move.type, "value") else move.type)
        return self.bonus_multipliers.get(move_type, 0.0)

    def is_noisy_move(self, move: Move) -> bool:
        """Check if a move is "noisy" (tactical/forcing).

        Noisy moves are captures, line formations, and territory claims
        that are searched in quiescence search.

        Parameters
        ----------
        move : Move
            The move to check.

        Returns
        -------
        bool
            True if the move is noisy.
        """
        move_type = str(move.type.value if hasattr(move.type, "value") else move.type)
        noisy_types = {
            "territory_claim",
            "line_formation",
            "chain_capture",
            "overtaking_capture",
            "continue_capture_segment",
            "recovery_slide",  # RR-CANON-R110–R115: tactical marker recovery
        }
        return move_type in noisy_types


class KillerMoveTable:
    """Tracks killer moves at each search depth.

    Killer moves are moves that caused beta cutoffs at sibling nodes.
    They are tried early in move ordering because they often cause
    cutoffs at the current node too.

    The table maintains up to `max_killers` moves per depth level.

    Attributes
    ----------
    max_killers : int
        Maximum killer moves to store per depth.
    max_depth : int
        Maximum depth to track (to bound memory usage).
    """

    def __init__(self, max_killers: int = 2, max_depth: int = 100) -> None:
        self.max_killers = max_killers
        self.max_depth = max_depth
        self._table: dict[int, list[Move]] = {}

    def get(self, depth: int) -> list[Move]:
        """Get killer moves at a depth.

        Parameters
        ----------
        depth : int
            Search depth.

        Returns
        -------
        List[Move]
            List of killer moves (may be empty).
        """
        return self._table.get(depth, [])

    def store(self, move: Move, depth: int) -> None:
        """Store a killer move at a depth.

        The move is added to the front of the list. If the list
        exceeds max_killers, the oldest move is removed.

        Parameters
        ----------
        move : Move
            Move that caused a cutoff.
        depth : int
            Search depth where cutoff occurred.
        """
        if depth > self.max_depth:
            return

        killers = self._table.get(depth, [])

        # Don't add duplicates
        for k in killers:
            if moves_equal(move, k):
                return

        killers.insert(0, move)
        if len(killers) > self.max_killers:
            killers.pop()

        self._table[depth] = killers

    def clear(self) -> None:
        """Clear all killer moves."""
        self._table.clear()

    def is_killer(self, move: Move, depth: int) -> bool:
        """Check if a move is a killer at the given depth.

        Parameters
        ----------
        move : Move
            Move to check.
        depth : int
            Search depth.

        Returns
        -------
        bool
            True if move is a killer at this depth.
        """
        return any(moves_equal(move, k) for k in self.get(depth))


def moves_equal(move1: Move, move2: Move) -> bool:
    """Check if two moves are equal for killer move matching.

    This is a structural equality check, not identity.

    Parameters
    ----------
    move1, move2 : Move
        Moves to compare.

    Returns
    -------
    bool
        True if moves are equal.
    """
    if move1.type != move2.type:
        return False

    # Handle None positions
    if move1.to is None or move2.to is None:
        return move1.to is None and move2.to is None

    if move1.to.x != move2.to.x or move1.to.y != move2.to.y:
        return False

    # Check from positions
    from1 = getattr(move1, "from_pos", None)
    from2 = getattr(move2, "from_pos", None)

    if from1 is None and from2 is None:
        return True
    if from1 and from2:
        return from1.x == from2.x and from1.y == from2.y
    return False


def order_moves(
    moves: list[Move],
    depth: int = 0,
    killer_table: KillerMoveTable | None = None,
    scorer: MovePriorityScorer | None = None,
) -> list[Move]:
    """Order moves for alpha-beta search.

    Moves are ordered as:
    1. Killer moves at this depth (if killer_table provided)
    2. Non-killer moves sorted by priority (highest first)

    Parameters
    ----------
    moves : List[Move]
        Moves to order.
    depth : int
        Current search depth.
    killer_table : KillerMoveTable, optional
        Killer move table for killer heuristic.
    scorer : MovePriorityScorer, optional
        Scorer for priority values.

    Returns
    -------
    List[Move]
        Ordered moves.
    """
    if not moves:
        return moves

    if scorer is None:
        scorer = MovePriorityScorer()

    killers = []
    others = []

    for move in moves:
        is_killer = killer_table and killer_table.is_killer(move, depth)
        if is_killer:
            killers.append(move)
        else:
            others.append(move)

    # Sort others by priority (descending)
    others.sort(key=lambda m: scorer.get_priority(m), reverse=True)

    return killers + others


def order_moves_with_scores(
    moves: list[Move],
    scores: list[float],
    scorer: MovePriorityScorer | None = None,
) -> list[tuple[Move, float]]:
    """Order moves by evaluation score plus priority bonus.

    This is used for root-level move ordering where we have
    evaluation scores from the previous iteration.

    Parameters
    ----------
    moves : List[Move]
        Moves to order.
    scores : List[float]
        Corresponding evaluation scores.
    scorer : MovePriorityScorer, optional
        Scorer for bonus values.

    Returns
    -------
    List[Tuple[Move, float]]
        List of (move, adjusted_score) tuples sorted descending.
    """
    if not moves:
        return []

    if scorer is None:
        scorer = MovePriorityScorer()

    scored = []
    for move, score in zip(moves, scores, strict=False):
        bonus = scorer.get_bonus(move)
        scored.append((move, score + bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_noisy_moves(
    moves: list[Move],
    scorer: MovePriorityScorer | None = None,
) -> list[Move]:
    """Filter moves to only noisy (tactical) moves.

    Used by quiescence search to extend only on tactical moves.

    Parameters
    ----------
    moves : List[Move]
        All legal moves.
    scorer : MovePriorityScorer, optional
        Scorer to determine noisiness.

    Returns
    -------
    List[Move]
        Only the noisy moves.
    """
    if scorer is None:
        scorer = MovePriorityScorer()

    return [m for m in moves if scorer.is_noisy_move(m)]


def score_noisy_moves(
    noisy_moves: list[Move],
    scorer: MovePriorityScorer | None = None,
) -> list[tuple[int, Move]]:
    """Score noisy moves by priority.

    Parameters
    ----------
    noisy_moves : List[Move]
        Noisy moves to score.
    scorer : MovePriorityScorer, optional
        Scorer for priority values.

    Returns
    -------
    List[Tuple[int, Move]]
        List of (priority, move) tuples sorted descending.
    """
    if scorer is None:
        scorer = MovePriorityScorer()

    scored = [(scorer.get_priority(m), m) for m in noisy_moves]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

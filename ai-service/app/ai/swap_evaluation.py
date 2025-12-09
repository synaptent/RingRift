"""
Swap (Pie Rule) Evaluation Module for RingRift AI.

This module provides swap decision evaluation logic, extracted from HeuristicAI
for better separation of concerns and reusability across AI implementations.

The swap rule (pie rule) allows Player 2 to swap positions with Player 1 after
P1's opening move(s), inheriting P1's board position. This module evaluates
whether swapping is strategically advantageous based on opening position quality.

Evaluation approaches:
1. **Bonus-based evaluation** (``evaluate_swap_opening_bonus``):
   - Rewards center positions, adjacency to center, and stack height
   - Original approach from HeuristicAI v1.2

2. **Classifier-based evaluation** (``evaluate_swap_with_classifier``):
   - Uses normalized 0-1 opening strength scores
   - Position-type-specific adjustments (corner, edge, diagonal)
   - Enhanced approach from HeuristicAI v1.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, TYPE_CHECKING

from ..models import GameState, Position, RingStack

if TYPE_CHECKING:
    from .fast_geometry import FastGeometry


@dataclass
class SwapWeights:
    """Weight configuration for swap evaluation.

    These weights control how the swap evaluator scores P1's opening position.
    Zero weights produce zero swap bonus (random swap decisions), enabling
    full weight-space exploration during training.

    Attributes
    ----------
    opening_center : float
        Bonus per P1 stack in a center position.
    opening_adjacency : float
        Bonus for P1 stacks adjacent to center positions.
    opening_height : float
        Bonus per stack height on P1 stacks.
    corner_penalty : float
        Penalty for corner positions (weak openings).
    edge_bonus : float
        Bonus for edge positions (moderate openings).
    diagonal_bonus : float
        Bonus for strategic diagonal positions.
    opening_strength : float
        Multiplier for normalized opening strength (0-1 scale).
    exploration_temperature : float
        Temperature for stochastic exploration in swap decisions.
        0 = deterministic, >0 adds Gaussian noise during training.
    """

    opening_center: float = 15.0
    opening_adjacency: float = 3.0
    opening_height: float = 2.0
    corner_penalty: float = 8.0
    edge_bonus: float = 2.0
    diagonal_bonus: float = 6.0
    opening_strength: float = 20.0
    exploration_temperature: float = 0.0

    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "SwapWeights":
        """Extract swap weights from a HeuristicAI instance.

        Parameters
        ----------
        ai : HeuristicAI
            AI instance with weight attributes.

        Returns
        -------
        SwapWeights
            Weights extracted from the AI's current configuration.
        """
        return cls(
            opening_center=getattr(ai, "WEIGHT_SWAP_OPENING_CENTER", 15.0),
            opening_adjacency=getattr(ai, "WEIGHT_SWAP_OPENING_ADJACENCY", 3.0),
            opening_height=getattr(ai, "WEIGHT_SWAP_OPENING_HEIGHT", 2.0),
            corner_penalty=getattr(ai, "WEIGHT_SWAP_CORNER_PENALTY", 8.0),
            edge_bonus=getattr(ai, "WEIGHT_SWAP_EDGE_BONUS", 2.0),
            diagonal_bonus=getattr(ai, "WEIGHT_SWAP_DIAGONAL_BONUS", 6.0),
            opening_strength=getattr(ai, "WEIGHT_SWAP_OPENING_STRENGTH", 20.0),
            exploration_temperature=getattr(
                ai, "WEIGHT_SWAP_EXPLORATION_TEMPERATURE", 0.0
            ),
        )


class SwapEvaluator:
    """Evaluator for swap (pie rule) decisions.

    This class encapsulates all swap evaluation logic, providing both
    bonus-based and classifier-based evaluation approaches.

    Parameters
    ----------
    weights : SwapWeights
        Weight configuration for evaluation.
    fast_geo : FastGeometry, optional
        Pre-computed geometry tables for efficient position lookups.
        If not provided, will be lazily initialized on first use.
    """

    def __init__(
        self,
        weights: Optional[SwapWeights] = None,
        fast_geo: Optional["FastGeometry"] = None,
    ) -> None:
        self.weights = weights or SwapWeights()
        self._fast_geo = fast_geo
        self._center_cache: dict = {}

    @property
    def fast_geo(self) -> "FastGeometry":
        """Lazily initialize FastGeometry if not provided."""
        if self._fast_geo is None:
            from .fast_geometry import FastGeometry

            self._fast_geo = FastGeometry.get_instance()
        return self._fast_geo

    def get_center_positions(self, game_state: GameState) -> FrozenSet[str]:
        """Get center position keys for the board.

        Uses caching for repeated calls with the same board type.
        """
        board_type = game_state.board.type
        if board_type not in self._center_cache:
            self._center_cache[board_type] = self.fast_geo.get_center_positions(
                board_type
            )
        return self._center_cache[board_type]

    def get_adjacent_positions(
        self,
        position: Position,
        game_state: GameState,
    ) -> List[Position]:
        """Get adjacent positions around a position."""
        from ..rules.geometry import BoardGeometry

        return BoardGeometry.get_adjacent_positions(
            position, game_state.board.type, game_state.board.size
        )

    def is_corner_position(
        self,
        position: Position,
        game_state: GameState,
    ) -> bool:
        """Check if a position is at a board corner.

        Corner positions are generally weak for openings as they have
        limited mobility and expansion potential.

        Parameters
        ----------
        position : Position
            Position to check.
        game_state : GameState
            Current game state for board geometry.

        Returns
        -------
        bool
            True if position is at a corner.
        """
        size = game_state.board.size
        x, y = position.x, position.y
        corners = {(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)}
        return (x, y) in corners

    def is_edge_position(
        self,
        position: Position,
        game_state: GameState,
    ) -> bool:
        """Check if a position is on an edge (but not a corner).

        Edge positions have moderate strategic value - better than corners
        but not as flexible as center positions.

        Parameters
        ----------
        position : Position
            Position to check.
        game_state : GameState
            Current game state for board geometry.

        Returns
        -------
        bool
            True if position is on an edge (excluding corners).
        """
        size = game_state.board.size
        x, y = position.x, position.y
        on_edge = x == 0 or x == size - 1 or y == 0 or y == size - 1
        is_corner = self.is_corner_position(position, game_state)
        return on_edge and not is_corner

    def is_strategic_diagonal_position(
        self,
        position: Position,
        game_state: GameState,
    ) -> bool:
        """Check if a position is on a key diagonal (one step from center).

        Strategic diagonal positions offer good line-forming potential
        and board control. For an 8x8 board, these include positions like
        (2,2), (2,5), (5,2), (5,5) - one step diagonally from center.

        Parameters
        ----------
        position : Position
            Position to check.
        game_state : GameState
            Current game state for board geometry.

        Returns
        -------
        bool
            True if position is strategically diagonal to center.
        """
        size = game_state.board.size
        center_positions = self.get_center_positions(game_state)
        pos_key = position.to_key()

        # Already a center position - not a "diagonal" position
        if pos_key in center_positions:
            return False

        # Check if diagonally adjacent to any center position
        x, y = position.x, position.y
        diagonal_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in diagonal_offsets:
            adj_x, adj_y = x + dx, y + dy
            if 0 <= adj_x < size and 0 <= adj_y < size:
                adj_key = f"{adj_x},{adj_y}"
                if adj_key in center_positions:
                    return True
        return False

    def compute_opening_strength(
        self,
        position: Position,
        game_state: GameState,
    ) -> float:
        """Compute opening strength score for a position (0-1 scale).

        This classifier evaluates how strong an opening move is based on
        the position's strategic value:

        - Center positions: 0.9-1.0 (highest value)
        - Adjacent to center: 0.7-0.8
        - Strategic diagonals: 0.5-0.6
        - Edge positions: 0.3-0.4
        - Corner positions: 0.1-0.2 (lowest value)

        Parameters
        ----------
        position : Position
            The position to evaluate.
        game_state : GameState
            Current game state for board geometry.

        Returns
        -------
        float
            Opening strength score between 0.0 and 1.0.
        """
        pos_key = position.to_key()
        center_positions = self.get_center_positions(game_state)

        # Center: highest value
        if pos_key in center_positions:
            return 0.95

        # Adjacent to center: high value
        adjacent = self.get_adjacent_positions(position, game_state)
        for adj_pos in adjacent:
            if adj_pos.to_key() in center_positions:
                return 0.75

        # Strategic diagonal: medium-high value
        if self.is_strategic_diagonal_position(position, game_state):
            return 0.55

        # Edge (not corner): medium-low value
        if self.is_edge_position(position, game_state):
            return 0.35

        # Corner: lowest value
        if self.is_corner_position(position, game_state):
            return 0.15

        # Default: moderate value for other positions
        return 0.45

    def evaluate_swap_opening_bonus(
        self,
        game_state: GameState,
    ) -> float:
        """Evaluate the strategic value of P1's opening position.

        This method computes a bonus score that represents how valuable it
        would be for P2 to swap into P1's position. The bonus is based on:

        - How many P1 stacks occupy center positions (highest weight)
        - How many P1 stacks are adjacent to center positions
        - Total stack height of P1's stacks

        This is the original v1.2 bonus-based approach.

        Parameters
        ----------
        game_state : GameState
            Current game state where P2 is deciding whether to swap.

        Returns
        -------
        float
            Swap opening bonus score (0.0 if swap is not strategically
            valuable, positive otherwise based on P1's opening strength).
        """
        # Only applies to 2-player games
        if len(game_state.players) != 2:
            return 0.0

        # Find P1's stacks (the opponent's stacks at the time of swap)
        p1_number = 1
        p1_stacks = [
            s
            for s in game_state.board.stacks.values()
            if s.controlling_player == p1_number
        ]

        if not p1_stacks:
            return 0.0

        center_positions = self.get_center_positions(game_state)
        bonus = 0.0

        for stack in p1_stacks:
            pos_key = stack.position.to_key()

            # High bonus for center stacks
            if pos_key in center_positions:
                bonus += self.weights.opening_center

            # Adjacency bonus - check if near center
            adjacent = self.get_adjacent_positions(stack.position, game_state)
            for adj_pos in adjacent:
                if adj_pos.to_key() in center_positions:
                    bonus += self.weights.opening_adjacency
                    break  # Only count once per stack

            # Height bonus
            bonus += stack.stack_height * self.weights.opening_height

        return bonus

    def evaluate_swap_with_classifier(
        self,
        game_state: GameState,
    ) -> float:
        """Evaluate swap decision using the Opening Position Classifier.

        This enhanced swap evaluation uses the opening strength classifier
        to compute a normalized swap value. Unlike the simpler bonus-based
        approach in ``evaluate_swap_opening_bonus()``, this method:

        1. Computes a 0-1 opening strength for each P1 position
        2. Applies position-type-specific weight adjustments
        3. Returns a score that directly indicates swap desirability

        Parameters
        ----------
        game_state : GameState
            Current game state where P2 is deciding whether to swap.

        Returns
        -------
        float
            Swap value score. Higher scores indicate swapping is more
            advantageous. The score combines:
            - Normalized opening strength * opening_strength weight
            - Position-type bonuses/penalties (corner, edge, diagonal)
        """
        if len(game_state.players) != 2:
            return 0.0

        p1_number = 1
        p1_stacks = [
            s
            for s in game_state.board.stacks.values()
            if s.controlling_player == p1_number
        ]

        if not p1_stacks:
            return 0.0

        total_strength = 0.0
        total_bonus = 0.0

        for stack in p1_stacks:
            pos = stack.position

            # Compute normalized opening strength (0-1)
            strength = self.compute_opening_strength(pos, game_state)
            total_strength += strength

            # Apply position-type-specific adjustments
            if self.is_corner_position(pos, game_state):
                total_bonus -= self.weights.corner_penalty
            elif self.is_edge_position(pos, game_state):
                total_bonus += self.weights.edge_bonus
            elif self.is_strategic_diagonal_position(pos, game_state):
                total_bonus += self.weights.diagonal_bonus

            # Stack height still matters
            total_bonus += stack.stack_height * self.weights.opening_height

        # Combine strength-based and bonus-based evaluation
        # Average strength for multiple stacks (normalize by stack count)
        avg_strength = total_strength / len(p1_stacks) if p1_stacks else 0.0
        strength_score = avg_strength * self.weights.opening_strength

        return strength_score + total_bonus

    def evaluate_swap_light(
        self,
        stacks: dict,
        player_number: int,
    ) -> float:
        """Lightweight swap evaluation for fast paths.

        Simplified version that only counts opponent stacks without
        full position classification. Used in make/unmake evaluation paths.

        Parameters
        ----------
        stacks : dict
            Dictionary of stack positions to stack data.
        player_number : int
            The player considering the swap (usually P2).

        Returns
        -------
        float
            Simple swap bonus based on opponent stack count.
        """
        opp_player = 2 if player_number == 1 else 1

        opp_stacks = sum(
            1
            for s in stacks.values()
            if hasattr(s, "controlling_player") and s.controlling_player == opp_player
        )

        # Bonus for swapping into position with more stacks
        return opp_stacks * self.weights.opening_center * 0.5


# Convenience function for quick swap evaluation
def evaluate_swap_bonus(
    game_state: GameState,
    weights: Optional[SwapWeights] = None,
) -> float:
    """Convenience function for evaluating swap opening bonus.

    Parameters
    ----------
    game_state : GameState
        Current game state.
    weights : SwapWeights, optional
        Custom weights. Uses defaults if not provided.

    Returns
    -------
    float
        Swap opening bonus score.
    """
    evaluator = SwapEvaluator(weights=weights)
    return evaluator.evaluate_swap_opening_bonus(game_state)


def evaluate_swap_classifier(
    game_state: GameState,
    weights: Optional[SwapWeights] = None,
) -> float:
    """Convenience function for classifier-based swap evaluation.

    Parameters
    ----------
    game_state : GameState
        Current game state.
    weights : SwapWeights, optional
        Custom weights. Uses defaults if not provided.

    Returns
    -------
    float
        Swap classifier score.
    """
    evaluator = SwapEvaluator(weights=weights)
    return evaluator.evaluate_swap_with_classifier(game_state)

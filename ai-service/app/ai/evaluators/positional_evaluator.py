"""
PositionalEvaluator: Positional evaluation for HeuristicAI.

This module provides the PositionalEvaluator class which handles all
positional evaluation features:
- Territory control (current territory space advantage)
- Center control (stacks in center positions)
- Territory closure (marker clustering for enclosure)
- Territory safety (distance from opponent stacks)

Extracted from HeuristicAI as part of the decomposition plan
(docs/architecture/HEURISTIC_AI_DECOMPOSITION_PLAN.md).

Example usage:
    evaluator = PositionalEvaluator()
    score = evaluator.evaluate_positional(game_state, player_idx=1)
    
    # Get detailed breakdown
    breakdown = evaluator.get_breakdown(game_state, player_idx=1)
    print(breakdown)  # {'territory': 16.0, 'center_control': 8.0, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models import GameState
    from ..fast_geometry import FastGeometry


@dataclass
class PositionalWeights:
    """Weight configuration for positional evaluation.

    These weights control the relative importance of each positional feature
    in the overall evaluation. Default values match the HeuristicAI class
    constants for backward compatibility.

    Attributes:
        territory: Weight for territory space advantage.
        center_control: Weight for stacks in center positions.
        territory_closure: Weight for marker clustering (enclosure potential).
        territory_safety: Weight for distance from opponent stacks.
        stack_synergy: Weight for coordinated stacks that support each other.
        mutual_defense: Weight for stacks that protect each other.
        expansion_potential: Weight for territory growth opportunities.
        frontier_strength: Weight for strength of expansion frontier.
    """
    territory: float = 8.0
    center_control: float = 4.0
    territory_closure: float = 10.0
    territory_safety: float = 5.0
    stack_synergy: float = 4.0
    mutual_defense: float = 3.0
    expansion_potential: float = 5.0
    frontier_strength: float = 3.0

    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "PositionalWeights":
        """Create PositionalWeights from HeuristicAI instance weights.

        This factory method extracts the relevant WEIGHT_* attributes from
        a HeuristicAI instance to create a PositionalWeights configuration.

        Args:
            ai: HeuristicAI instance to extract weights from.

        Returns:
            PositionalWeights with values matching the AI's configuration.
        """
        return cls(
            territory=getattr(ai, "WEIGHT_TERRITORY", 8.0),
            center_control=getattr(ai, "WEIGHT_CENTER_CONTROL", 4.0),
            territory_closure=getattr(ai, "WEIGHT_TERRITORY_CLOSURE", 10.0),
            territory_safety=getattr(ai, "WEIGHT_TERRITORY_SAFETY", 5.0),
            stack_synergy=getattr(ai, "WEIGHT_STACK_SYNERGY", 4.0),
            mutual_defense=getattr(ai, "WEIGHT_MUTUAL_DEFENSE", 3.0),
            expansion_potential=getattr(ai, "WEIGHT_EXPANSION_POTENTIAL", 5.0),
            frontier_strength=getattr(ai, "WEIGHT_FRONTIER_STRENGTH", 3.0),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary format.

        Returns:
            Dictionary with weight names as keys (WEIGHT_* format).
        """
        return {
            "WEIGHT_TERRITORY": self.territory,
            "WEIGHT_CENTER_CONTROL": self.center_control,
            "WEIGHT_TERRITORY_CLOSURE": self.territory_closure,
            "WEIGHT_TERRITORY_SAFETY": self.territory_safety,
            "WEIGHT_STACK_SYNERGY": self.stack_synergy,
            "WEIGHT_MUTUAL_DEFENSE": self.mutual_defense,
            "WEIGHT_EXPANSION_POTENTIAL": self.expansion_potential,
            "WEIGHT_FRONTIER_STRENGTH": self.frontier_strength,
        }


@dataclass
class PositionalScore:
    """Result from positional evaluation with feature breakdown.

    Attributes:
        total: Sum of all positional feature scores.
        territory: Score from territory space advantage.
        center_control: Score from center position control.
        territory_closure: Score from marker clustering.
        territory_safety: Score from opponent proximity.
        stack_synergy: Score from coordinated stacks.
        expansion_potential: Score from territory growth opportunities.
    """
    total: float = 0.0
    territory: float = 0.0
    center_control: float = 0.0
    territory_closure: float = 0.0
    territory_safety: float = 0.0
    stack_synergy: float = 0.0
    expansion_potential: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for breakdown reporting."""
        return {
            "total": self.total,
            "territory": self.territory,
            "center_control": self.center_control,
            "territory_closure": self.territory_closure,
            "territory_safety": self.territory_safety,
            "stack_synergy": self.stack_synergy,
            "expansion_potential": self.expansion_potential,
        }


# Type alias for HeuristicAI to avoid circular import
if TYPE_CHECKING:
    from ..heuristic_ai import HeuristicAI


class PositionalEvaluator:
    """Evaluates positional control: territory, center, closure.
    
    This evaluator computes positional features for position evaluation:
    
    Features computed:
    - territory: Current territory space advantage vs best opponent
    - center_control: Stacks in center positions (control is valuable)
    - territory_closure: Marker clustering for enclosure potential
    - territory_safety: Penalty for opponent stacks near our markers
    
    All features use symmetric evaluation where appropriate
    (my_value - opponent_value).
    
    Example:
        evaluator = PositionalEvaluator()
        score = evaluator.evaluate_positional(game_state, player_idx=1)
        
        # With custom weights
        weights = PositionalWeights(territory=12.0, center_control=6.0)
        evaluator = PositionalEvaluator(weights=weights)
    """
    
    def __init__(
        self,
        weights: Optional[PositionalWeights] = None,
        fast_geo: Optional["FastGeometry"] = None,
    ) -> None:
        """Initialize PositionalEvaluator with optional weight overrides.
        
        Args:
            weights: Optional PositionalWeights configuration. If None, uses
                default weights matching HeuristicAI class constants.
            fast_geo: Optional FastGeometry instance for center position
                lookup. If None, lazily fetches singleton instance when needed.
        """
        self.weights = weights or PositionalWeights()
        self._fast_geo = fast_geo
    
    @property
    def fast_geo(self) -> "FastGeometry":
        """Lazily get FastGeometry singleton for board geometry operations."""
        if self._fast_geo is None:
            from ..fast_geometry import FastGeometry
            self._fast_geo = FastGeometry.get_instance()
        return self._fast_geo

    def set_geometry(self, fast_geo: "FastGeometry") -> None:
        """Set geometry provider for backward compatibility and tests."""
        self._fast_geo = fast_geo

    def evaluate_positional(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Compute total positional score for a player.
        
        This is the main entry point for positional evaluation. It computes
        all positional features and returns the weighted sum.
        
        Args:
            state: Current game state to evaluate.
            player_idx: Player number (1-indexed) to evaluate for.
            
        Returns:
            Total weighted positional score (positive = advantage for player).
        """
        result = self._compute_all_features(state, player_idx)
        return result.total
    
    def get_breakdown(
        self,
        state: "GameState",
        player_idx: int,
    ) -> dict[str, float]:
        """Get detailed breakdown of positional evaluation.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Dictionary with all feature scores and total.
        """
        result = self._compute_all_features(state, player_idx)
        return result.to_dict()
    
    def set_weights(self, weights: dict[str, float]) -> None:
        """Update weight values from a profile dictionary.

        This method allows dynamic weight updates from
        HEURISTIC_WEIGHT_PROFILES or other configuration sources.

        Args:
            weights: Dictionary with WEIGHT_* keys to update.
        """
        if "WEIGHT_TERRITORY" in weights:
            self.weights.territory = weights["WEIGHT_TERRITORY"]
        if "WEIGHT_CENTER_CONTROL" in weights:
            self.weights.center_control = weights["WEIGHT_CENTER_CONTROL"]
        if "WEIGHT_TERRITORY_CLOSURE" in weights:
            self.weights.territory_closure = weights["WEIGHT_TERRITORY_CLOSURE"]
        if "WEIGHT_TERRITORY_SAFETY" in weights:
            self.weights.territory_safety = weights["WEIGHT_TERRITORY_SAFETY"]
        if "WEIGHT_STACK_SYNERGY" in weights:
            self.weights.stack_synergy = weights["WEIGHT_STACK_SYNERGY"]
        if "WEIGHT_MUTUAL_DEFENSE" in weights:
            self.weights.mutual_defense = weights["WEIGHT_MUTUAL_DEFENSE"]
        if "WEIGHT_EXPANSION_POTENTIAL" in weights:
            self.weights.expansion_potential = weights["WEIGHT_EXPANSION_POTENTIAL"]
        if "WEIGHT_FRONTIER_STRENGTH" in weights:
            self.weights.frontier_strength = weights["WEIGHT_FRONTIER_STRENGTH"]
    
    def _compute_all_features(
        self,
        state: "GameState",
        player_idx: int,
    ) -> PositionalScore:
        """Compute all positional features and return detailed result.

        This is the internal workhorse that computes each feature
        independently. Made symmetric where appropriate.

        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).

        Returns:
            PositionalScore with all feature values and total.
        """
        result = PositionalScore()

        # Territory control (symmetric)
        result.territory = self._evaluate_territory(state, player_idx)

        # Center control
        result.center_control = self._evaluate_center_control(
            state, player_idx
        )

        # Territory closure (marker clustering)
        result.territory_closure = self._evaluate_territory_closure(
            state, player_idx
        )

        # Territory safety (opponent proximity)
        result.territory_safety = self._evaluate_territory_safety(
            state, player_idx
        )

        # Stack synergy (coordinated stacks)
        result.stack_synergy = self._evaluate_stack_synergy(
            state, player_idx
        )

        # Expansion potential (territory growth opportunities)
        result.expansion_potential = self._evaluate_expansion_potential(
            state, player_idx
        )

        # Compute total
        result.total = (
            result.territory +
            result.center_control +
            result.territory_closure +
            result.territory_safety +
            result.stack_synergy +
            result.expansion_potential
        )

        return result
    
    def _get_player(self, state: "GameState", player_idx: int):
        """Get player info by player number.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Player object or None if not found.
        """
        for p in state.players:
            if p.player_number == player_idx:
                return p
        return None
    
    def _evaluate_territory(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate territory control (symmetric).
        
        Computes the difference between our territory and the best opponent's
        territory, weighted by the territory weight.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Territory advantage score.
        """
        my_player = self._get_player(state, player_idx)
        if not my_player:
            return 0.0
        
        my_territory = my_player.territory_spaces
        
        # Compare with opponents (find max)
        opponent_territory = 0
        for player in state.players:
            if player.player_number != player_idx:
                opponent_territory = max(
                    opponent_territory,
                    player.territory_spaces
                )
        
        return (my_territory - opponent_territory) * self.weights.territory
    
    def _evaluate_center_control(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate control of center positions (symmetric).

        Center positions are strategically valuable. We gain points for
        controlling them and lose points when opponents control them.

        Made symmetric by computing (my_center_count - opponent_center_count)
        to ensure P1+P2 evaluations sum to 0.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Center control score (symmetric).
        """
        center_positions = self._get_center_positions(state)

        my_center_count = 0
        opp_center_count = 0

        for pos_key in center_positions:
            if pos_key in state.board.stacks:
                stack = state.board.stacks[pos_key]
                if stack.controlling_player == player_idx:
                    my_center_count += 1
                else:
                    opp_center_count += 1

        # Symmetric: advantage over opponents
        advantage = my_center_count - opp_center_count
        return advantage * self.weights.center_control
    
    def _get_center_positions(self, state: "GameState") -> frozenset:
        """Get center position keys for the board using FastGeometry cache.
        
        Args:
            state: Current game state.
            
        Returns:
            Frozenset of position keys for center positions.
        """
        return self.fast_geo.get_center_positions(state.board.type)
    
    def _evaluate_territory_closure(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate how close we are to enclosing a territory (symmetric).

        Uses a simplified metric based on marker clustering. Closer markers
        (lower average distance) indicate potential for territory enclosure.
        Also rewards having more markers as a prerequisite for territory.

        Made symmetric by computing (my_closure - max_opponent_closure) to
        ensure P1+P2 evaluations sum to 0.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Territory closure score (symmetric).
        """
        my_closure = self._compute_closure_score_for_player(state, player_idx)

        # Compute max opponent closure for symmetric evaluation
        opp_closures = [
            self._compute_closure_score_for_player(state, p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_closure = max(opp_closures) if opp_closures else 0.0

        # Symmetric: advantage over best opponent
        advantage = my_closure - max_opp_closure
        return advantage * self.weights.territory_closure

    def _compute_closure_score_for_player(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute raw territory closure score for a specific player.

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Raw closure score (before applying weight).
        """
        # Get markers for this player
        markers = [
            m for m in state.board.markers.values()
            if m.player == player_num
        ]

        if not markers:
            return 0.0

        # Calculate "clustering" - average distance between markers
        total_dist = 0.0
        count = 0

        # Sample a few pairs to estimate density if too many markers
        if len(markers) < 10:
            markers_to_check = markers
        else:
            markers_to_check = markers[:10]

        for i, m1 in enumerate(markers_to_check):
            for m2 in markers_to_check[i + 1:]:
                dist = (
                    abs(m1.position.x - m2.position.x) +
                    abs(m1.position.y - m2.position.y)
                )
                if m1.position.z is not None and m2.position.z is not None:
                    dist += abs(m1.position.z - m2.position.z)

                total_dist += dist
                count += 1

        if count == 0:
            return 0.0

        avg_dist = total_dist / count

        # Lower average distance is better (more clustered)
        clustering_score = 10.0 / max(1.0, avg_dist)

        # Also reward total number of markers as a prerequisite for territory
        marker_count_score = len(markers) * 0.5

        return clustering_score + marker_count_score
    
    def _evaluate_territory_safety(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate safety of potential territories (symmetric).

        Penalizes positions where opponent stacks are near our marker clusters,
        as they could disrupt our territory formation.

        Made symmetric by computing (my_safety - max_opponent_safety) to ensure
        P1+P2 evaluations sum to 0.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Territory safety score (symmetric).
        """
        my_safety = self._compute_safety_score_for_player(state, player_idx)

        # Compute max opponent safety for symmetric evaluation
        opp_safeties = [
            self._compute_safety_score_for_player(state, p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_safety = max(opp_safeties) if opp_safeties else 0.0

        # Symmetric: advantage over best opponent
        advantage = my_safety - max_opp_safety
        return advantage * self.weights.territory_safety

    def _compute_safety_score_for_player(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute raw territory safety score for a player.

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Raw safety score (0 = safe, negative = threatened).
        """
        board = state.board

        # Get player's markers and opponent stacks
        player_markers = [
            m for m in board.markers.values()
            if m.player == player_num
        ]
        opponent_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player != player_num
        ]

        if not player_markers or not opponent_stacks:
            return 0.0

        score = 0.0
        for marker in player_markers:
            min_dist = float('inf')
            for stack in opponent_stacks:
                # Manhattan distance approximation
                dist = (
                    abs(marker.position.x - stack.position.x) +
                    abs(marker.position.y - stack.position.y)
                )
                if (marker.position.z is not None and
                        stack.position.z is not None):
                    dist += abs(marker.position.z - stack.position.z)
                min_dist = min(min_dist, dist)

            # If opponent is very close (dist 1 or 2), penalty
            if min_dist <= 2:
                score -= (3.0 - min_dist)  # -2 for dist 1, -1 for dist 2

        return score

    def _evaluate_stack_synergy(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate stack synergy (coordination between stacks).

        Stack synergy measures how well our stacks work together:
        - Adjacent friendly stacks that protect each other (mutual defense)
        - Stacks positioned to support each other in captures
        - Coordinated stacks that control space together

        Made symmetric by computing (my_synergy - max_opponent_synergy).

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Stack synergy score (positive = better coordination).
        """
        my_synergy = self._compute_synergy_score_for_player(state, player_idx)

        # Compute max opponent synergy for symmetric evaluation
        opp_synergies = [
            self._compute_synergy_score_for_player(state, p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_synergy = max(opp_synergies) if opp_synergies else 0.0

        # Symmetric: advantage over best opponent
        return (my_synergy - max_opp_synergy) * self.weights.stack_synergy

    def _compute_synergy_score_for_player(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute raw stack synergy score for a player.

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Raw synergy score.
        """
        board = state.board
        board_type = board.type
        stacks = board.stacks

        player_stacks = [
            s for s in stacks.values()
            if s.controlling_player == player_num
        ]

        if len(player_stacks) < 2:
            return 0.0

        synergy_score = 0.0
        mutual_defense_score = 0.0

        for stack in player_stacks:
            pos_key = stack.position.to_key()
            adjacent_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)

            for adj_key in adjacent_keys:
                if adj_key in stacks:
                    adj_stack = stacks[adj_key]

                    # Check for friendly adjacent stacks (synergy)
                    if adj_stack.controlling_player == player_num:
                        # Adjacent friendly stacks provide mutual support
                        synergy_score += 0.5

                        # Mutual defense: stacks that can protect each other
                        # (similar cap heights provide better mutual defense)
                        height_diff = abs(stack.cap_height - adj_stack.cap_height)
                        if height_diff <= 1:
                            mutual_defense_score += (
                                self.weights.mutual_defense * 0.3
                            )

        # Avoid double-counting (each pair counted twice)
        synergy_score /= 2.0
        mutual_defense_score /= 2.0

        return synergy_score + mutual_defense_score

    def _evaluate_expansion_potential(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate expansion potential (territory growth opportunities).

        Measures the potential for territory expansion:
        - Empty spaces adjacent to our stacks (room to grow)
        - Frontier strength (stacks at the edge of our territory)
        - Growth trajectory (direction of expansion)

        Made symmetric by computing (my_potential - max_opponent_potential).

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Expansion potential score (positive = more growth opportunity).
        """
        my_expansion = self._compute_expansion_score_for_player(
            state, player_idx
        )

        # Compute max opponent expansion for symmetric evaluation
        opp_expansions = [
            self._compute_expansion_score_for_player(state, p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_expansion = max(opp_expansions) if opp_expansions else 0.0

        # Symmetric: advantage over best opponent
        return (my_expansion - max_opp_expansion) * self.weights.expansion_potential

    def _compute_expansion_score_for_player(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute raw expansion potential score for a player.

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Raw expansion score.
        """
        board = state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces

        player_stacks = [
            s for s in stacks.values()
            if s.controlling_player == player_num
        ]

        if not player_stacks:
            return 0.0

        expansion_score = 0.0
        frontier_score = 0.0

        # Count empty adjacent spaces (expansion opportunities)
        expansion_spaces = set()

        for stack in player_stacks:
            pos_key = stack.position.to_key()
            adjacent_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)

            is_frontier_stack = False
            for adj_key in adjacent_keys:
                # Check for empty spaces (expansion opportunities)
                if adj_key not in stacks and adj_key not in collapsed:
                    expansion_spaces.add(adj_key)
                    is_frontier_stack = True

            # Frontier stacks (stacks at the edge of our territory)
            if is_frontier_stack:
                # Frontier strength based on stack height
                frontier_score += (
                    stack.cap_height * self.weights.frontier_strength * 0.2
                )

        # Expansion potential from empty adjacent spaces
        expansion_score = len(expansion_spaces) * 0.3

        return expansion_score + frontier_score

    # === Compatibility methods for HeuristicAI delegation ===
    # These methods match the original HeuristicAI method signatures
    # to enable gradual migration.
    
    def evaluate_territory(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate territory feature only.

        Compatibility method matching HeuristicAI._evaluate_territory
        signature.

        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Territory score.
        """
        return self._evaluate_territory(state, player_idx)
    
    def evaluate_center_control(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate center control feature only.
        
        Compatibility method matching HeuristicAI._evaluate_center_control
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Center control score.
        """
        return self._evaluate_center_control(state, player_idx)
    
    def evaluate_territory_closure(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate territory closure feature only.
        
        Compatibility method matching HeuristicAI._evaluate_territory_closure
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Territory closure score.
        """
        return self._evaluate_territory_closure(state, player_idx)
    
    def evaluate_territory_safety(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate territory safety feature only.
        
        Compatibility method matching HeuristicAI._evaluate_territory_safety
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Territory safety score.
        """
        return self._evaluate_territory_safety(state, player_idx)
    
    def evaluate_connectivity(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate piece connectivity score.

        This is implemented as territory closure which measures marker
        clustering/connectivity as a proxy for enclosure potential.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Connectivity score (via territory_closure).
        """
        return self._evaluate_territory_closure(state, player_idx)

    def evaluate_stack_synergy(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate stack synergy feature only.

        Compatibility method for HeuristicAI delegation.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Stack synergy score (positive = better coordination).
        """
        return self._evaluate_stack_synergy(state, player_idx)

    def evaluate_expansion_potential(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate expansion potential feature only.

        Compatibility method for HeuristicAI delegation.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Expansion potential score (positive = more growth opportunity).
        """
        return self._evaluate_expansion_potential(state, player_idx)

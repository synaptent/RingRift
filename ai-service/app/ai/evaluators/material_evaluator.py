"""
MaterialEvaluator: Material balance evaluation for HeuristicAI.

This module provides the MaterialEvaluator class which handles all
material-related evaluation features:
- Stack control (count, height, cap height)
- Ring counting (rings in hand, eliminated rings)
- Marker count
- Stack diversification (penalties/bonuses for stack distribution)

Extracted from HeuristicAI as part of the decomposition plan
(docs/architecture/HEURISTIC_AI_DECOMPOSITION_PLAN.md).

Example usage:
    evaluator = MaterialEvaluator()
    score = evaluator.evaluate_material(game_state, player_idx=1)
    
    # Get detailed breakdown
    breakdown = evaluator.get_breakdown(game_state, player_idx=1)
    print(breakdown)  # {'stack_control': 15.0, 'rings_in_hand': 6.0, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models import GameState


@dataclass
class MaterialWeights:
    """Weight configuration for material evaluation.
    
    These weights control the relative importance of each material feature
    in the overall evaluation. Default values match the HeuristicAI class
    constants for backward compatibility.
    
    Attributes:
        stack_control: Weight for relative stack count advantage.
        stack_height: Weight for effective stack height (diminishing).
        cap_height: Weight for capture power advantage (cap heights).
        rings_in_hand: Weight for ring reserve advantage.
        eliminated_rings: Weight for elimination progress toward victory.
        marker_count: Weight for marker density on board.
        no_stacks_penalty: Penalty for having zero stacks.
        single_stack_penalty: Penalty for having only one stack (vulnerable).
        stack_diversity_bonus: Bonus per additional stack beyond 1.
    """
    stack_control: float = 10.0
    stack_height: float = 5.0
    cap_height: float = 6.0
    rings_in_hand: float = 3.0
    eliminated_rings: float = 12.0
    marker_count: float = 1.5
    no_stacks_penalty: float = 50.0
    single_stack_penalty: float = 10.0
    stack_diversity_bonus: float = 2.0
    
    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "MaterialWeights":
        """Create MaterialWeights from HeuristicAI instance weights.
        
        This factory method extracts the relevant WEIGHT_* attributes from
        a HeuristicAI instance to create a MaterialWeights configuration.
        
        Args:
            ai: HeuristicAI instance to extract weights from.
            
        Returns:
            MaterialWeights with values matching the AI's configuration.
        """
        return cls(
            stack_control=getattr(ai, "WEIGHT_STACK_CONTROL", 10.0),
            stack_height=getattr(ai, "WEIGHT_STACK_HEIGHT", 5.0),
            cap_height=getattr(ai, "WEIGHT_CAP_HEIGHT", 6.0),
            rings_in_hand=getattr(ai, "WEIGHT_RINGS_IN_HAND", 3.0),
            eliminated_rings=getattr(ai, "WEIGHT_ELIMINATED_RINGS", 12.0),
            marker_count=getattr(ai, "WEIGHT_MARKER_COUNT", 1.5),
            no_stacks_penalty=getattr(
                ai, "WEIGHT_NO_STACKS_PENALTY", 50.0
            ),
            single_stack_penalty=getattr(
                ai, "WEIGHT_SINGLE_STACK_PENALTY", 10.0
            ),
            stack_diversity_bonus=getattr(
                ai, "WEIGHT_STACK_DIVERSITY_BONUS", 2.0
            ),
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary format.
        
        Returns:
            Dictionary with weight names as keys (WEIGHT_* format).
        """
        return {
            "WEIGHT_STACK_CONTROL": self.stack_control,
            "WEIGHT_STACK_HEIGHT": self.stack_height,
            "WEIGHT_CAP_HEIGHT": self.cap_height,
            "WEIGHT_RINGS_IN_HAND": self.rings_in_hand,
            "WEIGHT_ELIMINATED_RINGS": self.eliminated_rings,
            "WEIGHT_MARKER_COUNT": self.marker_count,
            "WEIGHT_NO_STACKS_PENALTY": self.no_stacks_penalty,
            "WEIGHT_SINGLE_STACK_PENALTY": self.single_stack_penalty,
            "WEIGHT_STACK_DIVERSITY_BONUS": self.stack_diversity_bonus,
        }


@dataclass
class MaterialScore:
    """Result from material evaluation with feature breakdown.
    
    Attributes:
        total: Sum of all material feature scores.
        stack_control: Score from relative stack count.
        stack_height: Score from effective stack heights.
        cap_height: Score from capture power (cap heights).
        stack_diversity: Score from stack diversification (symmetric).
        rings_in_hand: Score from ring reserves.
        eliminated_rings: Score from elimination progress.
        marker_count: Score from marker density.
    """
    total: float = 0.0
    stack_control: float = 0.0
    stack_height: float = 0.0
    cap_height: float = 0.0
    stack_diversity: float = 0.0
    rings_in_hand: float = 0.0
    eliminated_rings: float = 0.0
    marker_count: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for breakdown reporting."""
        return {
            "total": self.total,
            "stack_control": self.stack_control,
            "stack_height": self.stack_height,
            "cap_height": self.cap_height,
            "stack_diversity": self.stack_diversity,
            "rings_in_hand": self.rings_in_hand,
            "eliminated_rings": self.eliminated_rings,
            "marker_count": self.marker_count,
        }


# Type alias for HeuristicAI to avoid circular import
if TYPE_CHECKING:
    from ..heuristic_ai import HeuristicAI


class MaterialEvaluator:
    """Evaluates material balance: stacks, rings, markers.
    
    This evaluator computes material-related features for position evaluation:
    
    Features computed:
    - stack_control: Relative stack count advantage (my_stacks - opp_stacks)
    - stack_height: Effective height advantage with diminishing returns > 5
    - cap_height: Capture power advantage (sum of cap heights per rules §10.1)
    - stack_diversity: Symmetric penalty/bonus for stack diversification
    - rings_in_hand: Ring reserve advantage over best opponent
    - eliminated_rings: Progress toward ring elimination victory
    - marker_count: Total markers on board for this player
    
    All features except eliminated_rings and marker_count are computed
    symmetrically (my_value - opponent_value) to ensure zero-sum across
    players.
    
    Example:
        evaluator = MaterialEvaluator()
        score = evaluator.evaluate_material(game_state, player_idx=1)
        
        # With custom weights
        weights = MaterialWeights(stack_control=15.0, rings_in_hand=5.0)
        evaluator = MaterialEvaluator(weights=weights)
    """
    
    def __init__(self, weights: Optional[MaterialWeights] = None) -> None:
        """Initialize MaterialEvaluator with optional weight overrides.
        
        Args:
            weights: Optional MaterialWeights configuration. If None, uses
                default weights matching HeuristicAI class constants.
        """
        self.weights = weights or MaterialWeights()
    
    def evaluate_material(self, state: "GameState", player_idx: int) -> float:
        """Compute total material score for a player.
        
        This is the main entry point for material evaluation. It computes
        all material features and returns the weighted sum.
        
        Args:
            state: Current game state to evaluate.
            player_idx: Player number (1-indexed) to evaluate for.
            
        Returns:
            Total weighted material score (positive = advantage for player).
        """
        result = self._compute_all_features(state, player_idx)
        return result.total
    
    def count_rings(self, state: "GameState", player_idx: int) -> int:
        """Count total rings for a player across all stacks.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Total ring count in all stacks controlled by this player.
        """
        total = 0
        for stack in state.board.stacks.values():
            if stack.controlling_player == player_idx:
                total += stack.stack_height
        return total
    
    def get_height_advantage(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Compute stack height advantage for a player.
        
        Uses diminishing returns for height > 5 to discourage mega-stacks.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Height advantage (my_height - max_opponent_height).
        """
        my_height = 0.0
        max_opp_height = 0.0
        
        for stack in state.board.stacks.values():
            effective = self._effective_height(stack.stack_height)
            if stack.controlling_player == player_idx:
                my_height += effective
            else:
                # Track per-opponent, take max
                max_opp_height = max(max_opp_height, effective)
        
        return my_height - max_opp_height
    
    def get_breakdown(
        self,
        state: "GameState",
        player_idx: int,
    ) -> dict[str, float]:
        """Get detailed breakdown of material evaluation.
        
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
        if "WEIGHT_STACK_CONTROL" in weights:
            self.weights.stack_control = weights["WEIGHT_STACK_CONTROL"]
        if "WEIGHT_STACK_HEIGHT" in weights:
            self.weights.stack_height = weights["WEIGHT_STACK_HEIGHT"]
        if "WEIGHT_CAP_HEIGHT" in weights:
            self.weights.cap_height = weights["WEIGHT_CAP_HEIGHT"]
        if "WEIGHT_RINGS_IN_HAND" in weights:
            self.weights.rings_in_hand = weights["WEIGHT_RINGS_IN_HAND"]
        if "WEIGHT_ELIMINATED_RINGS" in weights:
            self.weights.eliminated_rings = weights["WEIGHT_ELIMINATED_RINGS"]
        if "WEIGHT_MARKER_COUNT" in weights:
            self.weights.marker_count = weights["WEIGHT_MARKER_COUNT"]
        if "WEIGHT_NO_STACKS_PENALTY" in weights:
            val = weights["WEIGHT_NO_STACKS_PENALTY"]
            self.weights.no_stacks_penalty = val
        if "WEIGHT_SINGLE_STACK_PENALTY" in weights:
            val = weights["WEIGHT_SINGLE_STACK_PENALTY"]
            self.weights.single_stack_penalty = val
        if "WEIGHT_STACK_DIVERSITY_BONUS" in weights:
            val = weights["WEIGHT_STACK_DIVERSITY_BONUS"]
            self.weights.stack_diversity_bonus = val
    
    def _compute_all_features(
        self,
        state: "GameState",
        player_idx: int,
    ) -> MaterialScore:
        """Compute all material features and return detailed result.
        
        This is the internal workhorse that computes each feature
        independently. Made symmetric where appropriate.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            MaterialScore with all feature values and total.
        """
        result = MaterialScore()
        
        # Collect stack data
        my_stacks = 0
        opponent_stacks = 0
        my_height = 0.0
        opponent_height = 0.0
        my_cap_height = 0
        opponent_cap_height = 0
        
        for stack in state.board.stacks.values():
            if stack.controlling_player == player_idx:
                my_stacks += 1
                my_height += self._effective_height(stack.stack_height)
                my_cap_height += stack.cap_height
            else:
                opponent_stacks += 1
                opponent_height += self._effective_height(stack.stack_height)
                opponent_cap_height += stack.cap_height

        # Normalize opponent values by number of opponents (GPU-style averaging)
        # This ensures symmetric evaluation in multi-player games
        num_opponents = max(1, len(state.players) - 1)
        opponent_stacks_avg = opponent_stacks / num_opponents
        opponent_height_avg = opponent_height / num_opponents
        opponent_cap_height_avg = opponent_cap_height / num_opponents

        # Stack control (symmetric with per-opponent averaging)
        result.stack_control = (
            (my_stacks - opponent_stacks_avg) * self.weights.stack_control
        )

        # Stack height (symmetric with per-opponent averaging)
        result.stack_height = (
            (my_height - opponent_height_avg) * self.weights.stack_height
        )

        # Cap height (symmetric with per-opponent averaging)
        result.cap_height = (
            (my_cap_height - opponent_cap_height_avg) * self.weights.cap_height
        )

        # Stack diversity (symmetric with per-opponent averaging)
        my_diversity = self._diversity_score(my_stacks)
        opp_diversity = self._diversity_score(opponent_stacks_avg)
        result.stack_diversity = my_diversity - opp_diversity
        
        # Rings in hand (symmetric vs best opponent)
        my_player = self._get_player(state, player_idx)
        if my_player:
            my_rings = my_player.rings_in_hand
            max_opp_rings = 0
            for p in state.players:
                if p.player_number != player_idx:
                    max_opp_rings = max(max_opp_rings, p.rings_in_hand)
            result.rings_in_hand = (
                (my_rings - max_opp_rings) * self.weights.rings_in_hand
            )
            
            # Eliminated rings (not symmetric - absolute progress)
            result.eliminated_rings = (
                my_player.eliminated_rings * self.weights.eliminated_rings
            )
        
        # Marker count (not symmetric - absolute count)
        my_markers = sum(
            1 for m in state.board.markers.values()
            if m.player == player_idx
        )
        result.marker_count = my_markers * self.weights.marker_count
        
        # Compute total
        result.total = (
            result.stack_control +
            result.stack_height +
            result.cap_height +
            result.stack_diversity +
            result.rings_in_hand +
            result.eliminated_rings +
            result.marker_count
        )
        
        return result
    
    def _effective_height(self, height: int) -> float:
        """Compute effective height with diminishing returns for height > 5.
        
        This prevents the AI from overvaluing mega-stacks by applying
        diminishing returns after height 5.
        
        Args:
            height: Actual stack height.
            
        Returns:
            Effective height value for scoring.
        """
        return height if height <= 5 else 5 + (height - 5) * 0.1
    
    def _diversity_score(self, stack_count: int) -> float:
        """Compute diversification score for a stack count.
        
        Per canonical rules, having multiple stacks provides strategic
        flexibility while having 0 or 1 stacks is risky.
        
        Args:
            stack_count: Number of stacks controlled.
            
        Returns:
            Diversification score (negative for 0-1 stacks, positive for 2+).
        """
        if stack_count == 0:
            return -self.weights.no_stacks_penalty
        elif stack_count == 1:
            return -self.weights.single_stack_penalty
        else:
            return stack_count * self.weights.stack_diversity_bonus
    
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
    
    # === Compatibility methods for HeuristicAI delegation ===
    # These methods match the original HeuristicAI method signatures
    # to enable gradual migration.
    
    def evaluate_stack_control(
        self, 
        state: "GameState", 
        player_idx: int,
    ) -> float:
        """Evaluate stack control feature only.
        
        This method computes just the stack_control feature, matching
        the signature of HeuristicAI._evaluate_stack_control for 
        compatibility during migration.
        
        Note: This also includes stack_height, cap_height, and diversity
        since _evaluate_stack_control computed all of these together.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Stack control score including height and diversity.
        """
        result = self._compute_all_features(state, player_idx)
        # Original _evaluate_stack_control returned sum of:
        # stack_control + stack_height + cap_height + diversity
        return (
            result.stack_control + 
            result.stack_height + 
            result.cap_height + 
            result.stack_diversity
        )
    
    def evaluate_rings_in_hand(
        self, 
        state: "GameState", 
        player_idx: int,
    ) -> float:
        """Evaluate rings in hand feature only.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Rings in hand score (symmetric).
        """
        my_player = self._get_player(state, player_idx)
        if not my_player:
            return 0.0
        
        my_rings = my_player.rings_in_hand
        max_opp_rings = 0
        for p in state.players:
            if p.player_number != player_idx:
                max_opp_rings = max(max_opp_rings, p.rings_in_hand)
        
        return (my_rings - max_opp_rings) * self.weights.rings_in_hand
    
    def evaluate_eliminated_rings(
        self, 
        state: "GameState", 
        player_idx: int,
    ) -> float:
        """Evaluate eliminated rings feature only.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Eliminated rings score.
        """
        my_player = self._get_player(state, player_idx)
        if not my_player:
            return 0.0
        
        return my_player.eliminated_rings * self.weights.eliminated_rings
    
    def evaluate_marker_count(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate marker count (symmetric).

        Made symmetric by computing (my_markers - max_opponent_markers)
        to ensure P1+P2 evaluations sum to 0.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Marker count score (symmetric).
        """
        my_markers = sum(
            1 for m in state.board.markers.values()
            if m.player == player_idx
        )

        # Compute max opponent markers for symmetric evaluation
        opp_marker_counts = [
            sum(1 for m in state.board.markers.values()
                if m.player == p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_markers = max(opp_marker_counts) if opp_marker_counts else 0

        # Symmetric: advantage over best opponent
        advantage = my_markers - max_opp_markers
        return advantage * self.weights.marker_count

"""
TacticalEvaluator: Tactical evaluation for HeuristicAI.

This module provides the TacticalEvaluator class which handles all
tactical evaluation features:
- Opponent threats (adjacent enemy stacks with capture advantage)
- Vulnerability (line-of-sight capture risk)
- Overtake potential (line-of-sight capture opportunities)

Extracted from HeuristicAI as part of the decomposition plan
(docs/architecture/HEURISTIC_AI_DECOMPOSITION_PLAN.md).

Example usage:
    evaluator = TacticalEvaluator()
    score = evaluator.evaluate_tactical(game_state, player_idx=1)
    
    # Get detailed breakdown
    breakdown = evaluator.get_breakdown(game_state, player_idx=1)
    print(breakdown)  # {'opponent_threats': -4.0, 'vulnerability': -8.0, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models import GameState, Position, RingStack
    from ..fast_geometry import FastGeometry


@dataclass
class TacticalWeights:
    """Weight configuration for tactical evaluation.
    
    These weights control the relative importance of each tactical feature
    in the overall evaluation. Default values match the HeuristicAI class
    constants for backward compatibility.
    
    Attributes:
        opponent_threat: Weight for adjacent enemy stacks with
            capture advantage.
        vulnerability: Weight for line-of-sight capture risk.
        overtake_potential: Weight for line-of-sight capture
            opportunities.
    """
    opponent_threat: float = 6.0
    vulnerability: float = 8.0
    overtake_potential: float = 8.0
    
    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "TacticalWeights":
        """Create TacticalWeights from HeuristicAI instance weights.
        
        This factory method extracts the relevant WEIGHT_* attributes from
        a HeuristicAI instance to create a TacticalWeights configuration.
        
        Args:
            ai: HeuristicAI instance to extract weights from.
            
        Returns:
            TacticalWeights with values matching the AI's configuration.
        """
        return cls(
            opponent_threat=getattr(ai, "WEIGHT_OPPONENT_THREAT", 6.0),
            vulnerability=getattr(ai, "WEIGHT_VULNERABILITY", 8.0),
            overtake_potential=getattr(ai, "WEIGHT_OVERTAKE_POTENTIAL", 8.0),
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary format.
        
        Returns:
            Dictionary with weight names as keys (WEIGHT_* format).
        """
        return {
            "WEIGHT_OPPONENT_THREAT": self.opponent_threat,
            "WEIGHT_VULNERABILITY": self.vulnerability,
            "WEIGHT_OVERTAKE_POTENTIAL": self.overtake_potential,
        }


@dataclass
class TacticalScore:
    """Result from tactical evaluation with feature breakdown.
    
    Attributes:
        total: Sum of all tactical feature scores.
        opponent_threats: Score from adjacent enemy stacks.
        vulnerability: Score from line-of-sight capture risk.
        overtake_potential: Score from line-of-sight capture opportunities.
    """
    total: float = 0.0
    opponent_threats: float = 0.0
    vulnerability: float = 0.0
    overtake_potential: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for breakdown reporting."""
        return {
            "total": self.total,
            "opponent_threats": self.opponent_threats,
            "vulnerability": self.vulnerability,
            "overtake_potential": self.overtake_potential,
        }


# Type alias for HeuristicAI to avoid circular import
if TYPE_CHECKING:
    from ..heuristic_ai import HeuristicAI


class TacticalEvaluator:
    """Evaluates tactical situations: threats, captures, vulnerabilities.
    
    This evaluator computes tactical features for position evaluation:
    
    Features computed:
    - opponent_threats: Adjacent enemy stacks that threaten our stacks
    - vulnerability: Our stacks that can be captured via line-of-sight
    - overtake_potential: Enemy stacks we can capture via line-of-sight
    
    The evaluator maintains a visibility cache per evaluation call to avoid
    redundant line-of-sight calculations when evaluating vulnerability and
    overtake potential.
    
    Example:
        evaluator = TacticalEvaluator()
        score = evaluator.evaluate_tactical(game_state, player_idx=1)
        
        # With custom weights
        weights = TacticalWeights(vulnerability=10.0, overtake_potential=12.0)
        evaluator = TacticalEvaluator(weights=weights)
    """
    
    def __init__(
        self,
        weights: Optional[TacticalWeights] = None,
        fast_geo: Optional["FastGeometry"] = None,
    ) -> None:
        """Initialize TacticalEvaluator with optional weight overrides.
        
        Args:
            weights: Optional TacticalWeights configuration. If None, uses
                default weights matching HeuristicAI class constants.
            fast_geo: Optional FastGeometry instance for adjacency lookup.
                If None, lazily fetches singleton instance when needed.
        """
        self.weights = weights or TacticalWeights()
        self._fast_geo = fast_geo
        # Per-evaluation visibility cache
        # (cleared at start of each evaluate call)
        self._visible_stacks_cache: dict[str, list] = {}
    
    @property
    def fast_geo(self) -> "FastGeometry":
        """Lazily get FastGeometry singleton for board geometry operations."""
        if self._fast_geo is None:
            from ..fast_geometry import FastGeometry
            self._fast_geo = FastGeometry.get_instance()
        return self._fast_geo
    
    def evaluate_tactical(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Compute total tactical score for a player.
        
        This is the main entry point for tactical evaluation. It computes
        all tactical features and returns the weighted sum.
        
        Args:
            state: Current game state to evaluate.
            player_idx: Player number (1-indexed) to evaluate for.
            
        Returns:
            Total weighted tactical score (positive = advantage for player).
        """
        result = self._compute_all_features(state, player_idx)
        return result.total
    
    def get_breakdown(
        self,
        state: "GameState",
        player_idx: int,
    ) -> dict[str, float]:
        """Get detailed breakdown of tactical evaluation.
        
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
        if "WEIGHT_OPPONENT_THREAT" in weights:
            self.weights.opponent_threat = weights["WEIGHT_OPPONENT_THREAT"]
        if "WEIGHT_VULNERABILITY" in weights:
            self.weights.vulnerability = weights["WEIGHT_VULNERABILITY"]
        if "WEIGHT_OVERTAKE_POTENTIAL" in weights:
            val = weights["WEIGHT_OVERTAKE_POTENTIAL"]
            self.weights.overtake_potential = val
    
    def _compute_all_features(
        self,
        state: "GameState",
        player_idx: int,
    ) -> TacticalScore:
        """Compute all tactical features and return detailed result.
        
        This is the internal workhorse that computes each feature
        independently. Clears the visibility cache at the start.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            TacticalScore with all feature values and total.
        """
        # Clear visibility cache for this evaluation
        self._visible_stacks_cache.clear()
        
        result = TacticalScore()
        
        # Opponent threats (adjacent stacks)
        result.opponent_threats = self._evaluate_opponent_threats(
            state, player_idx
        )
        
        # Vulnerability (LoS capture risk)
        result.vulnerability = self._evaluate_vulnerability(state, player_idx)
        
        # Overtake potential (LoS capture opportunities)
        result.overtake_potential = self._evaluate_overtake_potential(
            state, player_idx
        )
        
        # Compute total
        result.total = (
            result.opponent_threats +
            result.vulnerability +
            result.overtake_potential
        )
        
        return result
    
    def _evaluate_opponent_threats(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate opponent threats (stacks near our stacks).
        
        Computes a penalty based on enemy stacks adjacent to our stacks
        that have higher cap_height (capture power per rules §10.1).
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Opponent threat score (typically negative = penalized).
        """
        score = 0.0
        board = state.board
        board_type = board.type
        stacks = board.stacks
        
        my_stacks = [
            s for s in stacks.values()
            if s.controlling_player == player_idx
        ]
        
        for my_stack in my_stacks:
            # Use fast key-based adjacency lookup
            pos_key = my_stack.position.to_key()
            adj_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)
            for adj_key in adj_keys:
                if adj_key in stacks:
                    adj_stack = stacks[adj_key]
                    if adj_stack.controlling_player != player_idx:
                        # Opponent stack adjacent to ours is a threat.
                        # Capture power is based on cap height per compact
                        # rules §10.1, so we compare using cap height here
                        # rather than total stack height.
                        threat_level = (
                            adj_stack.cap_height - my_stack.cap_height
                        )
                        score -= (
                            threat_level * self.weights.opponent_threat * 0.5
                        )
        
        return score
    
    def _evaluate_vulnerability(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate vulnerability of our stacks to overtaking captures.
        
        Considers relative cap heights of stacks in clear line of sight.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Vulnerability score (negative = more vulnerable).
        """
        score = 0.0
        my_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player_idx
        ]
        
        for stack in my_stacks:
            visible_stacks = self._get_visible_stacks(stack.position, state)
            
            for adj_stack in visible_stacks:
                # Capture power is based on cap height per compact rules §10.1
                if (adj_stack.controlling_player != player_idx
                        and adj_stack.cap_height > stack.cap_height):
                    diff = adj_stack.cap_height - stack.cap_height
                    score -= diff * 1.0
        
        return score * self.weights.vulnerability
    
    def _evaluate_overtake_potential(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate our ability to overtake opponent stacks.
        
        Considers relative cap heights of stacks in clear line of sight.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Overtake potential score (positive = more opportunities).
        """
        score = 0.0
        my_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player_idx
        ]
        
        for stack in my_stacks:
            visible_stacks = self._get_visible_stacks(stack.position, state)
            
            for adj_stack in visible_stacks:
                # Capture power is based on cap height per compact rules §10.1
                if (adj_stack.controlling_player != player_idx
                        and stack.cap_height > adj_stack.cap_height):
                    diff = stack.cap_height - adj_stack.cap_height
                    score += diff * 1.0
        
        return score * self.weights.overtake_potential
    
    def _get_visible_stacks(
        self,
        position: "Position",
        state: "GameState",
    ) -> list["RingStack"]:
        """Compute line-of-sight visible stacks from a position.
        
        Optimized version: Uses raw coordinates instead of creating Position
        objects in the inner loop. This provides ~3x speedup for this function.
        
        Results are cached per-evaluation to avoid redundant computations
        when the same position is queried multiple times (e.g., vulnerability
        and overtake potential both iterate over the same stacks).
        
        Args:
            position: Starting position for LoS check.
            state: Current game state.
            
        Returns:
            List of stacks visible in line of sight from position.
        """
        from ...models import BoardType
        
        # Check cache first
        cache_key = position.to_key()
        if cache_key in self._visible_stacks_cache:
            return self._visible_stacks_cache[cache_key]
        
        visible: list = []
        board = state.board
        board_type = board.type
        stacks = board.stacks
        
        # Get directions from FastGeometry (cached)
        directions = self.fast_geo.get_los_directions(board_type)
        
        # Pre-compute bounds limits for inline checking
        is_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
        if is_hex:
            # Canonical hex board uses cube radius = (size - 1).
            # BoardState.size is 13 for canonical radius-12 board.
            board_size = int(getattr(board, "size", 13) or 13)
            limit = board_size - 1
        elif board_type == BoardType.SQUARE8:
            limit = 8
        else:  # SQUARE19
            limit = 19
        
        curr_x = position.x
        curr_y = position.y
        # For hex, compute z from x,y (constraint: x + y + z = 0)
        if is_hex:
            curr_z = (
                position.z
                if position.z is not None
                else -position.x - position.y
            )
        else:
            curr_z = 0
        
        for dx, dy, dz in directions:
            x, y, z = curr_x, curr_y, curr_z
            
            while True:
                x += dx
                y += dy
                if is_hex:
                    z += dz
                    # Inline hex bounds check
                    if abs(x) > limit or abs(y) > limit or abs(z) > limit:
                        break
                    # Hex keys include cube z coordinate.
                    pos_key = f"{x},{y},{z}"
                else:
                    # Inline square bounds check
                    if x < 0 or x >= limit or y < 0 or y >= limit:
                        break
                    pos_key = f"{x},{y}"
                
                stack = stacks.get(pos_key)
                if stack is not None:
                    visible.append(stack)
                    break
        
        # Cache result for this evaluation
        self._visible_stacks_cache[cache_key] = visible
        
        return visible
    
    # === Compatibility methods for HeuristicAI delegation ===
    # These methods match the original HeuristicAI method signatures
    # to enable gradual migration.
    
    def evaluate_opponent_threats(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate opponent threats feature only.
        
        Compatibility method matching HeuristicAI._evaluate_opponent_threats
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Opponent threats score.
        """
        return self._evaluate_opponent_threats(state, player_idx)
    
    def evaluate_vulnerability(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate vulnerability feature only.
        
        Compatibility method matching HeuristicAI._evaluate_vulnerability
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Vulnerability score.
        """
        # Clear cache to ensure fresh computation
        self._visible_stacks_cache.clear()
        return self._evaluate_vulnerability(state, player_idx)
    
    def evaluate_overtake_potential(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate overtake potential feature only.
        
        Compatibility method matching HeuristicAI._evaluate_overtake_potential
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Overtake potential score.
        """
        # Clear cache to ensure fresh computation
        self._visible_stacks_cache.clear()
        return self._evaluate_overtake_potential(state, player_idx)
    
    def get_visible_stacks(
        self,
        position: "Position",
        state: "GameState",
    ) -> list["RingStack"]:
        """Get line-of-sight visible stacks from a position.
        
        Public accessor for the visibility calculation.
        
        Args:
            position: Starting position for LoS check.
            state: Current game state.
            
        Returns:
            List of stacks visible in line of sight from position.
        """
        return self._get_visible_stacks(position, state)

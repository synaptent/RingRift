"""
StrategicEvaluator: Strategic and endgame evaluation for HeuristicAI.

This module provides the StrategicEvaluator class which handles all
strategic and endgame-related evaluation features:
- Victory proximity (rings and territory-based)
- Opponent victory threat detection
- Forced elimination risk assessment
- Last-player-standing (LPS) advantage in multiplayer
- Multi-leader threat detection in multiplayer

Extracted from HeuristicAI as part of the decomposition plan
(docs/architecture/HEURISTIC_AI_DECOMPOSITION_PLAN.md).

Example usage:
    evaluator = StrategicEvaluator()
    score = evaluator.evaluate_strategic_all(game_state, player_idx=1)
    
    # Get detailed breakdown
    breakdown = evaluator.get_breakdown(game_state, player_idx=1)
    print(breakdown)  # {'victory_proximity': 12.0, 'fe_risk': -4.0, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models import GameState, Player
    from ..fast_geometry import FastGeometry


@dataclass
class StrategicWeights:
    """Weight configuration for strategic evaluation.
    
    These weights control the relative importance of each strategic feature
    in the overall evaluation. Default values match the HeuristicAI class
    constants for backward compatibility.
    
    Attributes:
        victory_proximity: Weight for victory proximity score.
        opponent_victory_threat: Weight for opponent win threat penalty.
        forced_elimination_risk: Weight for FE vulnerability penalty.
        lps_action_advantage: Weight for LPS dynamics in 3+ player games.
        multi_leader_threat: Weight for single-opponent-ahead penalty.
        victory_threshold_bonus: Bonus when at/near victory.
        rings_proximity_factor: Factor for rings-based proximity.
        territory_proximity_factor: Factor for territory-based proximity.
    """
    victory_proximity: float = 20.0
    opponent_victory_threat: float = 6.0
    forced_elimination_risk: float = 4.0
    lps_action_advantage: float = 2.0
    multi_leader_threat: float = 2.0
    victory_threshold_bonus: float = 1000.0
    rings_proximity_factor: float = 50.0
    territory_proximity_factor: float = 50.0
    
    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "StrategicWeights":
        """Create StrategicWeights from HeuristicAI instance weights.
        
        This factory method extracts the relevant WEIGHT_* attributes from
        a HeuristicAI instance to create a StrategicWeights configuration.
        
        Args:
            ai: HeuristicAI instance to extract weights from.
            
        Returns:
            StrategicWeights with values matching the AI's configuration.
        """
        return cls(
            victory_proximity=getattr(ai, "WEIGHT_VICTORY_PROXIMITY", 20.0),
            opponent_victory_threat=getattr(
                ai, "WEIGHT_OPPONENT_VICTORY_THREAT", 6.0
            ),
            forced_elimination_risk=getattr(
                ai, "WEIGHT_FORCED_ELIMINATION_RISK", 4.0
            ),
            lps_action_advantage=getattr(
                ai, "WEIGHT_LPS_ACTION_ADVANTAGE", 2.0
            ),
            multi_leader_threat=getattr(
                ai, "WEIGHT_MULTI_LEADER_THREAT", 2.0
            ),
            victory_threshold_bonus=getattr(
                ai, "WEIGHT_VICTORY_THRESHOLD_BONUS", 1000.0
            ),
            rings_proximity_factor=getattr(
                ai, "WEIGHT_RINGS_PROXIMITY_FACTOR", 50.0
            ),
            territory_proximity_factor=getattr(
                ai, "WEIGHT_TERRITORY_PROXIMITY_FACTOR", 50.0
            ),
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary format.
        
        Returns:
            Dictionary with weight names as keys (WEIGHT_* format).
        """
        return {
            "WEIGHT_VICTORY_PROXIMITY": self.victory_proximity,
            "WEIGHT_OPPONENT_VICTORY_THREAT": self.opponent_victory_threat,
            "WEIGHT_FORCED_ELIMINATION_RISK": self.forced_elimination_risk,
            "WEIGHT_LPS_ACTION_ADVANTAGE": self.lps_action_advantage,
            "WEIGHT_MULTI_LEADER_THREAT": self.multi_leader_threat,
            "WEIGHT_VICTORY_THRESHOLD_BONUS": self.victory_threshold_bonus,
            "WEIGHT_RINGS_PROXIMITY_FACTOR": self.rings_proximity_factor,
            "WEIGHT_TERRITORY_PROXIMITY_FACTOR": self.territory_proximity_factor,
        }


@dataclass
class StrategicScore:
    """Result from strategic evaluation with feature breakdown.
    
    Attributes:
        total: Sum of all strategic feature scores.
        victory_proximity: Score from victory proximity (symmetric).
        opponent_victory_threat: Penalty for opponent win threat.
        forced_elimination_risk: Penalty for FE vulnerability.
        lps_action_advantage: Score from LPS dynamics (3+ players).
        multi_leader_threat: Penalty for single opponent ahead (3+ players).
    """
    total: float = 0.0
    victory_proximity: float = 0.0
    opponent_victory_threat: float = 0.0
    forced_elimination_risk: float = 0.0
    lps_action_advantage: float = 0.0
    multi_leader_threat: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for breakdown reporting."""
        return {
            "total": self.total,
            "victory_proximity": self.victory_proximity,
            "opponent_victory_threat": self.opponent_victory_threat,
            "forced_elimination_risk": self.forced_elimination_risk,
            "lps_action_advantage": self.lps_action_advantage,
            "multi_leader_threat": self.multi_leader_threat,
        }


# Type alias for HeuristicAI to avoid circular import
if TYPE_CHECKING:
    from ..heuristic_ai import HeuristicAI


class StrategicEvaluator:
    """Evaluates strategic factors: victory proximity, endgame, multiplayer.
    
    This evaluator computes strategic-level features for position evaluation:
    
    Features computed:
    - victory_proximity: How close we are to winning vs opponents (symmetric).
    - opponent_victory_threat: Penalty when opponents are closer to victory.
    - forced_elimination_risk: Penalty for positions with FE vulnerability.
    - lps_action_advantage: In 3+ player games, reward for being active when
      opponents are inactive.
    - multi_leader_threat: In 3+ player games, penalty when single opponent
      is pulling ahead.
    
    All features are computed symmetrically (my_value - max_opponent_value)
    to ensure zero-sum across players where appropriate.
    
    Example:
        evaluator = StrategicEvaluator()
        score = evaluator.evaluate_strategic_all(game_state, player_idx=1)
        
        # With custom weights
        weights = StrategicWeights(victory_proximity=30.0)
        evaluator = StrategicEvaluator(weights=weights)
    """
    
    def __init__(
        self,
        weights: Optional[StrategicWeights] = None,
        fast_geo: Optional["FastGeometry"] = None,
    ) -> None:
        """Initialize StrategicEvaluator with optional weight overrides.
        
        Args:
            weights: Optional StrategicWeights configuration. If None, uses
                default weights matching HeuristicAI class constants.
            fast_geo: Optional FastGeometry instance for adjacency lookup.
                If None, lazily fetches singleton instance when needed.
        """
        self.weights = weights or StrategicWeights()
        self._fast_geo = fast_geo
    
    @property
    def fast_geo(self) -> "FastGeometry":
        """Lazily get FastGeometry singleton for board geometry operations."""
        if self._fast_geo is None:
            from ..fast_geometry import FastGeometry
            self._fast_geo = FastGeometry.get_instance()
        return self._fast_geo
    
    def evaluate_strategic_all(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Compute total strategic score for a player.
        
        This is the main entry point for strategic evaluation. It computes
        all strategic features and returns the weighted sum.
        
        Args:
            state: Current game state to evaluate.
            player_idx: Player number (1-indexed) to evaluate for.
            
        Returns:
            Total weighted strategic score (positive = advantage for player).
        """
        result = self._compute_all_features(state, player_idx)
        return result.total
    
    def get_breakdown(
        self,
        state: "GameState",
        player_idx: int,
    ) -> dict[str, float]:
        """Get detailed breakdown of strategic evaluation.
        
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
        if "WEIGHT_VICTORY_PROXIMITY" in weights:
            self.weights.victory_proximity = weights["WEIGHT_VICTORY_PROXIMITY"]
        if "WEIGHT_OPPONENT_VICTORY_THREAT" in weights:
            val = weights["WEIGHT_OPPONENT_VICTORY_THREAT"]
            self.weights.opponent_victory_threat = val
        if "WEIGHT_FORCED_ELIMINATION_RISK" in weights:
            val = weights["WEIGHT_FORCED_ELIMINATION_RISK"]
            self.weights.forced_elimination_risk = val
        if "WEIGHT_LPS_ACTION_ADVANTAGE" in weights:
            val = weights["WEIGHT_LPS_ACTION_ADVANTAGE"]
            self.weights.lps_action_advantage = val
        if "WEIGHT_MULTI_LEADER_THREAT" in weights:
            val = weights["WEIGHT_MULTI_LEADER_THREAT"]
            self.weights.multi_leader_threat = val
        if "WEIGHT_VICTORY_THRESHOLD_BONUS" in weights:
            val = weights["WEIGHT_VICTORY_THRESHOLD_BONUS"]
            self.weights.victory_threshold_bonus = val
        if "WEIGHT_RINGS_PROXIMITY_FACTOR" in weights:
            val = weights["WEIGHT_RINGS_PROXIMITY_FACTOR"]
            self.weights.rings_proximity_factor = val
        if "WEIGHT_TERRITORY_PROXIMITY_FACTOR" in weights:
            val = weights["WEIGHT_TERRITORY_PROXIMITY_FACTOR"]
            self.weights.territory_proximity_factor = val
    
    def _compute_all_features(
        self,
        state: "GameState",
        player_idx: int,
    ) -> StrategicScore:
        """Compute all strategic features and return detailed result.
        
        This is the internal workhorse that computes each feature
        independently. Made symmetric where appropriate.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            StrategicScore with all feature values and total.
        """
        result = StrategicScore()
        
        # Victory proximity (symmetric)
        result.victory_proximity = self._evaluate_victory_proximity(
            state, player_idx
        )
        
        # Opponent victory threat
        result.opponent_victory_threat = self._evaluate_opponent_victory_threat(
            state, player_idx
        )
        
        # Forced elimination risk
        result.forced_elimination_risk = self._evaluate_forced_elimination_risk(
            state, player_idx
        )
        
        # Multiplayer features (only for 3+ players)
        num_players = len(state.players)
        if num_players > 2:
            result.lps_action_advantage = self._evaluate_lps_action_advantage(
                state, player_idx
            )
            result.multi_leader_threat = self._evaluate_multi_leader_threat(
                state, player_idx
            )
        
        # Compute total
        result.total = (
            result.victory_proximity +
            result.opponent_victory_threat +
            result.forced_elimination_risk +
            result.lps_action_advantage +
            result.multi_leader_threat
        )
        
        return result
    
    def _victory_proximity_base_for_player(
        self,
        state: "GameState",
        player: "Player",
    ) -> float:
        """Compute base victory proximity score for a player.
        
        Uses configurable weight constants for full training exploration.
        
        This method evaluates how close a player is to any victory condition:
        - LPS (Last Player Standing) via consecutive exclusive rounds
        - Ring elimination threshold
        - Territory victory threshold
        
        Args:
            state: Current game state.
            player: Player object to evaluate for.
            
        Returns:
            Victory proximity score (higher = closer to winning).
        """
        # LPS proximity: treat a player nearing the required consecutive
        # exclusive rounds as an imminent victory threat. This must respect
        # per-game overrides (rulesOptions.lpsRoundsRequired).
        lps_player = getattr(state, "lps_consecutive_exclusive_player", None)
        lps_rounds = getattr(state, "lps_consecutive_exclusive_rounds", 0)
        if lps_player == getattr(player, "player_number", None) and isinstance(
            lps_rounds, int
        ):
            required_rounds = getattr(
                state,
                "lps_rounds_required",
                getattr(state, "lpsRoundsRequired", 3),
            )
            if not isinstance(required_rounds, int) or required_rounds <= 0:
                required_rounds = 3

            if lps_rounds >= required_rounds and required_rounds >= 1:
                return self.weights.victory_threshold_bonus
            if lps_rounds > 0:
                if required_rounds <= 1:
                    return self.weights.victory_threshold_bonus
                denom = float(required_rounds - 1)
                frac = min(1.0, max(0.0, float(lps_rounds) / denom))
                return self.weights.victory_threshold_bonus * (0.90 + 0.09 * frac)

        rings_needed = state.victory_threshold - player.eliminated_rings
        territory_needed = (
            state.territory_victory_threshold - player.territory_spaces
        )

        if rings_needed <= 0 or territory_needed <= 0:
            return self.weights.victory_threshold_bonus

        score = 0.0
        score += (
            (1.0 / max(1, rings_needed)) * self.weights.rings_proximity_factor
        )
        score += (
            (1.0 / max(1, territory_needed)) *
            self.weights.territory_proximity_factor
        )
        return score
    
    def _get_player(
        self,
        state: "GameState",
        player_idx: int,
    ) -> Optional["Player"]:
        """Get player object by player number."""
        for p in state.players:
            if p.player_number == player_idx:
                return p
        return None
    
    def _evaluate_victory_proximity(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate how close we are to winning (relative to opponents).
        
        Made symmetric: computes (my_proximity - max_opponent_proximity) so
        that the evaluation sums to approximately zero across all players.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Victory proximity score (symmetric: my_value - max_opponent_value).
        """
        my_player = self._get_player(state, player_idx)
        if not my_player:
            return 0.0

        my_proximity = self._victory_proximity_base_for_player(state, my_player)

        # Find max opponent proximity for symmetric evaluation
        max_opponent_proximity = 0.0
        for p in state.players:
            if p.player_number != player_idx:
                opp_proximity = self._victory_proximity_base_for_player(state, p)
                max_opponent_proximity = max(max_opponent_proximity, opp_proximity)

        # Symmetric: advantage over best opponent
        return (
            (my_proximity - max_opponent_proximity) *
            self.weights.victory_proximity
        )
    
    def _evaluate_opponent_victory_threat(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """
        Evaluate how much closer the leading opponent is to victory
        than we are.

        This mirrors the self victory proximity computation and
        compares our proximity score to the maximum proximity score
        among all opponents. A positive gap is treated as a threat
        and converted into a penalty.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Opponent victory threat penalty (negative when opponents are ahead).
        """
        my_player = self._get_player(state, player_idx)
        if not my_player:
            return 0.0

        self_prox = self._victory_proximity_base_for_player(state, my_player)

        max_opp_prox = 0.0
        for p in state.players:
            if p.player_number == player_idx:
                continue
            prox = self._victory_proximity_base_for_player(state, p)
            if prox > max_opp_prox:
                max_opp_prox = prox

        # Symmetric evaluation: positive when ahead, negative when behind
        # OLD (asymmetric): max(0.0, raw_gap) only penalized when behind
        # NEW (symmetric): raw difference gives bonus when ahead, penalty when behind
        raw_gap = max_opp_prox - self_prox

        # Return symmetric score: when ahead (raw_gap < 0) get bonus,
        # when behind (raw_gap > 0) get penalty
        return -raw_gap * self.weights.opponent_victory_threat
    
    def _approx_real_actions_for_player(
        self,
        state: "GameState",
        player_number: int,
    ) -> int:
        """
        Approximate the number of "real" actions (moves + placements) available
        to the given player.

        - Counts one move per stack that has at least one legal-looking move
          (empty neighbor or capturable enemy stack).
        - Adds one additional action if the player has rings in hand and there
          exists at least one empty, non-collapsed space where a ring could be
          placed.
          
        Args:
            state: Current game state.
            player_number: Player to count actions for.
            
        Returns:
            Approximate count of real actions available.
        """
        board = state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces
        approx_moves = 0

        for stack in stacks.values():
            if stack.controlling_player != player_number:
                continue

            # Use fast key-based adjacency lookup
            pos_key = stack.position.to_key()
            adjacent_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)
            stack_has_any_move = False
            for key in adjacent_keys:
                if key in collapsed:
                    continue

                if key in stacks:
                    target = stacks[key]
                    if (
                        target.controlling_player != player_number
                        and stack.cap_height >= target.cap_height
                    ):
                        stack_has_any_move = True
                        break
                else:
                    stack_has_any_move = True
                    break

            if stack_has_any_move:
                approx_moves += 1

        approx_placement = 0
        player = next(
            (
                p
                for p in state.players
                if p.player_number == player_number
            ),
            None,
        )
        if player and player.rings_in_hand > 0:
            # Check if there's at least one empty space
            has_empty = any(
                key not in board.stacks and key not in board.collapsed_spaces
                for key in self._iterate_board_keys(board)
            )
            if has_empty:
                approx_placement = 1

        return approx_moves + approx_placement
    
    def _iterate_board_keys(self, board) -> list[str]:
        """
        Iterate over all logical board coordinate keys for the given board.

        Uses pre-computed board keys from FastGeometry for O(1) lookup
        instead of regenerating Position objects on every call.
        """
        return self.fast_geo.get_all_board_keys(board.type)
    
    def _evaluate_forced_elimination_risk(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate forced elimination risk (symmetric).

        Penalise positions where we control many stacks but have very few
        real actions (moves or placements), indicating forced-elimination risk.

        Made symmetric by computing (my_risk - max_opponent_risk) to ensure
        P1+P2 evaluations sum to 0.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Forced elimination risk penalty (symmetric).
        """
        my_risk = self._compute_fe_risk_for_player(state, player_idx)

        # Compute max opponent risk for symmetric evaluation
        opp_risks = [
            self._compute_fe_risk_for_player(state, p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_risk = max(opp_risks) if opp_risks else 0.0

        # Symmetric: my risk minus opponent risk
        # If I'm at more risk than opponent, this is negative (bad for me)
        relative_risk = my_risk - max_opp_risk
        return -relative_risk * self.weights.forced_elimination_risk

    def _compute_fe_risk_for_player(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute raw forced elimination risk factor for a player.

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Risk factor (0 = no risk, higher = more risk).
        """
        board = state.board
        player_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player == player_num
        ]
        controlled_stacks = len(player_stacks)
        if controlled_stacks == 0:
            return 0.0

        approx_actions = self._approx_real_actions_for_player(
            state,
            player_num,
        )
        ratio = approx_actions / max(1, controlled_stacks)

        if ratio >= 2.0:
            return 0.0
        elif ratio >= 1.0:
            return 2.0 - ratio
        else:
            return 1.0 + (1.0 - ratio)
    
    def _evaluate_lps_action_advantage(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """
        Last-player-standing action advantage heuristic.

        In 3+ player games, reward being one of the few players with real
        actions left and penalise being the only player without actions.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            LPS action advantage score (positive when we have advantage).
        """
        players = state.players
        if len(players) <= 2:
            return 0.0

        actions = {
            p.player_number: self._approx_real_actions_for_player(
                state,
                p.player_number,
            )
            for p in players
        }
        self_actions = actions.get(player_idx, 0)
        self_has = self_actions > 0

        opp_with_action = sum(
            1
            for p in players
            if p.player_number != player_idx
            and actions.get(p.player_number, 0) > 0
        )

        if not self_has:
            advantage = -1.0
        else:
            total_opponents = len(players) - 1
            if total_opponents <= 0:
                advantage = 0.0
            else:
                inactive_fraction = (
                    total_opponents - opp_with_action
                ) / float(total_opponents)
                advantage = inactive_fraction

        return advantage * self.weights.lps_action_advantage
    
    def _evaluate_multi_leader_threat(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """
        Multi-player leader threat heuristic.

        In 3+ player games, penalise positions where a single opponent is
        much closer to victory than the other opponents.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Multi-leader threat penalty (negative when single opponent ahead).
        """
        players = state.players
        if len(players) <= 2:
            return 0.0

        prox_by_player = {
            p.player_number: self._victory_proximity_base_for_player(state, p)
            for p in players
        }

        opp_prox = [
            prox_by_player[p.player_number]
            for p in players
            if p.player_number != player_idx
        ]
        if len(opp_prox) < 2:
            return 0.0

        opp_prox_sorted = sorted(opp_prox, reverse=True)
        opp_top1 = opp_prox_sorted[0]
        opp_top2 = opp_prox_sorted[1]

        leader_gap = max(0.0, opp_top1 - opp_top2)

        return -leader_gap * self.weights.multi_leader_threat
    
    # === Compatibility methods for HeuristicAI delegation ===
    # These methods match the original HeuristicAI method signatures
    # to enable gradual migration.
    
    def evaluate_victory_proximity(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate victory proximity feature only.
        
        Compatibility method matching HeuristicAI._evaluate_victory_proximity
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Victory proximity score.
        """
        return self._evaluate_victory_proximity(state, player_idx)
    
    def evaluate_opponent_victory_threat(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate opponent victory threat feature only.
        
        Compatibility method matching HeuristicAI._evaluate_opponent_victory_threat
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Opponent victory threat score.
        """
        return self._evaluate_opponent_victory_threat(state, player_idx)
    
    def evaluate_forced_elimination_risk(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate forced elimination risk feature only.
        
        Compatibility method matching HeuristicAI._evaluate_forced_elimination_risk
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Forced elimination risk score.
        """
        return self._evaluate_forced_elimination_risk(state, player_idx)
    
    def evaluate_lps_action_advantage(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate LPS action advantage feature only.
        
        Compatibility method matching HeuristicAI._evaluate_lps_action_advantage
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            LPS action advantage score.
        """
        return self._evaluate_lps_action_advantage(state, player_idx)
    
    def evaluate_multi_leader_threat(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate multi-leader threat feature only.
        
        Compatibility method matching HeuristicAI._evaluate_multi_leader_threat
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Multi-leader threat score.
        """
        return self._evaluate_multi_leader_threat(state, player_idx)

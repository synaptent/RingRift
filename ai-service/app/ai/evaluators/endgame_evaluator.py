"""
EndgameEvaluator: Endgame and terminal position evaluation for HeuristicAI.

This module provides the EndgameEvaluator class which handles all
endgame-related evaluation features:
- Terminal position detection
- Game phase detection (early/mid/late/endgame)
- Recovery potential evaluation (RR-CANON-R110-R115)
- Forced outcome detection

Extracted from HeuristicAI as part of the decomposition plan
(docs/architecture/HEURISTIC_AI_DECOMPOSITION_PLAN.md).

Example usage:
    evaluator = EndgameEvaluator()
    
    # Check if position is terminal
    is_terminal = evaluator.is_terminal_position(game_state)
    
    # Detect game phase
    phase = evaluator.get_game_phase(game_state)  # early/mid/late/endgame
    
    # Evaluate recovery potential
    score = evaluator.evaluate_recovery_potential(game_state, player_idx=1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models import GameState, Player
    from ..fast_geometry import FastGeometry


@dataclass
class EndgameWeights:
    """Weight configuration for endgame evaluation.

    These weights control the relative importance of each endgame feature
    in the overall evaluation. Default values match the HeuristicAI class
    constants for backward compatibility.

    Attributes:
        recovery_potential: Weight for having recovery as strategic option.
        recovery_eligibility: Bonus/penalty for recovery eligibility status.
        buried_ring_value: Value of buried rings as recovery resource.
        recovery_threat: Threat penalty from opponent's recovery potential.
        terminal_win_score: Score for winning terminal positions.
        terminal_loss_score: Score for losing terminal positions.
        terminal_draw_score: Score for drawn terminal positions.
        endgame_aggression: Increased capture value in endgame.
        phase_transition_bonus: Bonus for favorable phase entry.
    """
    recovery_potential: float = 6.0
    recovery_eligibility: float = 8.0
    buried_ring_value: float = 3.0
    recovery_threat: float = 5.0
    terminal_win_score: float = 100000.0
    terminal_loss_score: float = -100000.0
    terminal_draw_score: float = 0.0
    endgame_aggression: float = 4.0
    phase_transition_bonus: float = 3.0

    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "EndgameWeights":
        """Create EndgameWeights from HeuristicAI instance weights.

        This factory method extracts the relevant WEIGHT_* attributes from
        a HeuristicAI instance to create an EndgameWeights configuration.

        Args:
            ai: HeuristicAI instance to extract weights from.

        Returns:
            EndgameWeights with values matching the AI's configuration.
        """
        return cls(
            recovery_potential=getattr(ai, "WEIGHT_RECOVERY_POTENTIAL", 6.0),
            recovery_eligibility=getattr(
                ai, "WEIGHT_RECOVERY_ELIGIBILITY", 8.0
            ),
            buried_ring_value=getattr(ai, "WEIGHT_BURIED_RING_VALUE", 3.0),
            recovery_threat=getattr(ai, "WEIGHT_RECOVERY_THREAT", 5.0),
            terminal_win_score=100000.0,
            terminal_loss_score=-100000.0,
            terminal_draw_score=0.0,
            endgame_aggression=getattr(ai, "WEIGHT_ENDGAME_AGGRESSION", 4.0),
            phase_transition_bonus=getattr(ai, "WEIGHT_PHASE_TRANSITION_BONUS", 3.0),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary format.

        Returns:
            Dictionary with weight names as keys (WEIGHT_* format).
        """
        return {
            "WEIGHT_RECOVERY_POTENTIAL": self.recovery_potential,
            "WEIGHT_RECOVERY_ELIGIBILITY": self.recovery_eligibility,
            "WEIGHT_BURIED_RING_VALUE": self.buried_ring_value,
            "WEIGHT_RECOVERY_THREAT": self.recovery_threat,
            "TERMINAL_WIN_SCORE": self.terminal_win_score,
            "TERMINAL_LOSS_SCORE": self.terminal_loss_score,
            "TERMINAL_DRAW_SCORE": self.terminal_draw_score,
            "WEIGHT_ENDGAME_AGGRESSION": self.endgame_aggression,
            "WEIGHT_PHASE_TRANSITION_BONUS": self.phase_transition_bonus,
        }


@dataclass
class EndgameScore:
    """Result from endgame evaluation with feature breakdown.

    Attributes:
        total: Sum of all endgame feature scores.
        recovery_potential: Score from recovery strategic value.
        phase_transition: Score from phase-aware modulation.
        is_terminal: Whether position is terminal.
        game_phase: Current game phase ('early', 'mid', 'late', 'endgame').
        forced_outcome: Detected forced outcome or None.
    """
    total: float = 0.0
    recovery_potential: float = 0.0
    phase_transition: float = 0.0
    is_terminal: bool = False
    game_phase: str = "early"
    forced_outcome: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for breakdown reporting."""
        return {
            "total": self.total,
            "recovery_potential": self.recovery_potential,
            "phase_transition": self.phase_transition,
            "is_terminal": self.is_terminal,
            "game_phase": self.game_phase,
            "forced_outcome": self.forced_outcome,
        }


# Type alias for HeuristicAI to avoid circular import
if TYPE_CHECKING:
    from ..heuristic_ai import HeuristicAI


class EndgameEvaluator:
    """Evaluates endgame-specific factors.
    
    This evaluator computes endgame-level features for position evaluation:
    
    Features computed:
    - is_terminal_position: Whether the game has ended.
    - get_game_phase: Current phase of the game (early/mid/late/endgame).
    - detect_forced_outcome: Detection of forced win/loss/draw.
    - recovery_potential: Strategic value of recovery mechanics.
    
    The endgame evaluator helps the AI understand when the game is
    transitioning to endgame phases where different strategies apply.
    
    Example:
        evaluator = EndgameEvaluator()
        is_end = evaluator.is_terminal_position(game_state)
        phase = evaluator.get_game_phase(game_state)
        score = evaluator.evaluate_recovery_potential(game_state, player_idx=1)
    """
    
    def __init__(
        self,
        weights: Optional[EndgameWeights] = None,
        fast_geo: Optional["FastGeometry"] = None,
    ) -> None:
        """Initialize EndgameEvaluator with optional weight overrides.
        
        Args:
            weights: Optional EndgameWeights configuration. If None, uses
                default weights matching HeuristicAI class constants.
            fast_geo: Optional FastGeometry instance for adjacency lookup.
                If None, lazily fetches singleton instance when needed.
        """
        self.weights = weights or EndgameWeights()
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
    
    def evaluate_endgame_all(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Compute total endgame score for a player.
        
        This is the main entry point for endgame evaluation. It computes
        all endgame features and returns the weighted sum.
        
        Args:
            state: Current game state to evaluate.
            player_idx: Player number (1-indexed) to evaluate for.
            
        Returns:
            Total weighted endgame score (positive = advantage for player).
        """
        result = self._compute_all_features(state, player_idx)
        return result.total

    def evaluate_endgame(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Backward-compatible alias for evaluate_endgame_all()."""
        return self.evaluate_endgame_all(state, player_idx)

    def get_breakdown(
        self,
        state: "GameState",
        player_idx: int,
    ) -> dict:
        """Get detailed breakdown of endgame evaluation.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Dictionary with all feature scores and metadata.
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
        if "WEIGHT_RECOVERY_POTENTIAL" in weights:
            self.weights.recovery_potential = weights["WEIGHT_RECOVERY_POTENTIAL"]
        if "WEIGHT_RECOVERY_ELIGIBILITY" in weights:
            self.weights.recovery_eligibility = weights["WEIGHT_RECOVERY_ELIGIBILITY"]
        if "WEIGHT_BURIED_RING_VALUE" in weights:
            self.weights.buried_ring_value = weights["WEIGHT_BURIED_RING_VALUE"]
        if "WEIGHT_RECOVERY_THREAT" in weights:
            self.weights.recovery_threat = weights["WEIGHT_RECOVERY_THREAT"]
        if "WEIGHT_ENDGAME_AGGRESSION" in weights:
            self.weights.endgame_aggression = weights["WEIGHT_ENDGAME_AGGRESSION"]
        if "WEIGHT_PHASE_TRANSITION_BONUS" in weights:
            self.weights.phase_transition_bonus = weights["WEIGHT_PHASE_TRANSITION_BONUS"]
    
    def _compute_all_features(
        self,
        state: "GameState",
        player_idx: int,
    ) -> EndgameScore:
        """Compute all endgame features and return detailed result.

        This is the internal workhorse that computes each feature
        independently.

        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).

        Returns:
            EndgameScore with all feature values and metadata.
        """
        result = EndgameScore()

        # Terminal detection
        result.is_terminal = self.is_terminal_position(state)

        # Game phase detection
        result.game_phase = self.get_game_phase(state)

        # Forced outcome detection
        result.forced_outcome = self.detect_forced_outcome(state, player_idx)

        # Recovery potential (only compute if not terminal)
        if not result.is_terminal:
            result.recovery_potential = self._evaluate_recovery_potential(
                state, player_idx
            )

            # Phase transition bonus (only compute if not terminal)
            result.phase_transition = self._evaluate_phase_transition(
                state, player_idx, result.game_phase
            )

        # Compute total
        result.total = result.recovery_potential + result.phase_transition

        return result
    
    def is_terminal_position(self, state: "GameState") -> bool:
        """Detect if position is terminal (game has ended).
        
        A position is terminal if the game status is 'completed'.
        This indicates that a victory condition has been met or
        a stalemate/draw has occurred.
        
        Args:
            state: Current game state.
            
        Returns:
            True if the game has ended, False otherwise.
        """
        return state.game_status == "completed"
    
    def evaluate_terminal_position(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate a terminal position for a player.
        
        Returns a large positive score for wins, large negative for losses,
        and zero for draws.
        
        Args:
            state: Current game state (must be terminal).
            player_idx: Player number to evaluate for.
            
        Returns:
            Terminal position score.
        """
        if not self.is_terminal_position(state):
            return 0.0
        
        if state.winner == player_idx:
            return self.weights.terminal_win_score
        elif state.winner is not None:
            return self.weights.terminal_loss_score
        else:
            return self.weights.terminal_draw_score
    
    def detect_forced_outcome(
        self,
        state: "GameState",
        player_idx: int,
    ) -> Optional[str]:
        """Detect forced win/loss/draw outcomes.
        
        This method attempts to detect if the current position has a
        forced outcome. Currently detects:
        - 'win': If player has already won
        - 'loss': If another player has won
        - 'draw': If game ended in stalemate
        - None: No forced outcome detected (game in progress)
        
        Note: This is a simple detection based on terminal status.
        More sophisticated forced-mate detection would require search.
        
        Args:
            state: Current game state.
            player_idx: Player number to evaluate for.
            
        Returns:
            'win', 'loss', 'draw', or None if no forced outcome.
        """
        if not self.is_terminal_position(state):
            # Could add detection of imminent forced outcomes here
            # For now, only detect terminal positions
            return None
        
        if state.winner == player_idx:
            return "win"
        elif state.winner is not None:
            return "loss"
        else:
            return "draw"
    
    def get_game_phase(self, state: "GameState") -> str:
        """Detect game phase: 'early', 'mid', 'late', 'endgame'.
        
        Game phase is determined by analyzing:
        - Total rings remaining in hands (early indicator)
        - Number of stacks on board (board development)
        - Eliminated rings (progress towards victory)
        - Territory control (late game indicator)
        
        Phase thresholds (as fraction of initial rings):
        - early: > 60% rings in hands
        - mid: 30-60% rings in hands
        - late: 10-30% rings in hands
        - endgame: < 10% rings in hands OR near victory threshold
        
        Args:
            state: Current game state.
            
        Returns:
            Game phase string: 'early', 'mid', 'late', or 'endgame'.
        """
        # Handle terminal positions
        if self.is_terminal_position(state):
            return "endgame"
        
        # Calculate total rings in hands and initial rings
        total_rings_in_hand = sum(p.rings_in_hand for p in state.players)
        
        # Estimate initial rings - this is board-dependent
        # Use a heuristic: sum of rings in hand + rings on board + eliminated
        total_eliminated = sum(p.eliminated_rings for p in state.players)
        rings_on_board = sum(
            stack.total_height
            for stack in state.board.stacks.values()
        )
        estimated_initial = (
            total_rings_in_hand + rings_on_board + total_eliminated
        )
        
        # Avoid division by zero
        if estimated_initial == 0:
            estimated_initial = 1
        
        rings_in_hand_ratio = total_rings_in_hand / estimated_initial
        
        # Check for near-victory condition (endgame)
        for player in state.players:
            rings_to_victory = (
                state.victory_threshold - player.eliminated_rings
            )
            territory_to_victory = (
                state.territory_victory_threshold - player.territory_spaces
            )
            
            # If any player is close to victory (within 20%), it's endgame
            if rings_to_victory <= state.victory_threshold * 0.2:
                return "endgame"
            if territory_to_victory <= state.territory_victory_threshold * 0.2:
                return "endgame"
        
        # Phase based on rings remaining
        if rings_in_hand_ratio > 0.6:
            return "early"
        elif rings_in_hand_ratio > 0.3:
            return "mid"
        elif rings_in_hand_ratio > 0.1:
            return "late"
        else:
            return "endgame"
    
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
    
    def _evaluate_recovery_potential(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate recovery potential (symmetric).

        Recovery (RR-CANON-R110–R115) allows temporarily eliminated players to
        slide markers to form lines, paying costs with buried ring extraction.
        This heuristic captures:

        1. Value of having recovery available as a strategic option
        2. Threat from opponents who have recovery potential
        3. Value of buried rings as recovery resources

        Made symmetric by computing (my_recovery_value - max_opponent_value)
        to ensure P1+P2 evaluations sum to 0.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Score representing recovery strategic value (symmetric).
        """
        my_recovery = self._compute_recovery_score_for_player(state, player_idx)

        # Compute max opponent recovery for symmetric evaluation
        opp_scores = [
            self._compute_recovery_score_for_player(state, p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        max_opp_recovery = max(opp_scores) if opp_scores else 0.0

        # Symmetric: advantage over best opponent
        return my_recovery - max_opp_recovery

    def _compute_recovery_score_for_player(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute raw recovery value for a player.

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Raw recovery score (before symmetry adjustment).
        """
        # Import here to avoid circular imports
        from ...rules.core import count_buried_rings, is_eligible_for_recovery
        from ...rules.recovery import has_any_recovery_move

        score = 0.0

        eligible = is_eligible_for_recovery(state, player_num)
        buried = count_buried_rings(state.board, player_num)

        if eligible:
            # Bonus for having recovery available
            score += self.weights.recovery_eligibility

            # Additional bonus if we actually have recovery moves
            if has_any_recovery_move(state, player_num):
                score += self.weights.recovery_potential
        else:
            # Small bonus for buried rings even if not currently eligible
            # (potential future recovery resource)
            score += buried * self.weights.buried_ring_value * 0.3

        return score

    def _evaluate_phase_transition(
        self,
        state: "GameState",
        player_idx: int,
        game_phase: str,
    ) -> float:
        """Evaluate phase transition bonus (symmetric).

        As the game transitions to endgame, aggressive play becomes more
        important. This feature:

        1. Increases value of captures in late/endgame phases
        2. Provides bonus for being ahead when entering endgame
        3. Penalizes falling behind in endgame

        Made symmetric by computing (my_advantage - max_opponent_advantage).

        Args:
            state: Current game state.
            player_idx: Player number.
            game_phase: Current game phase from get_game_phase().

        Returns:
            Phase transition score (positive = favorable position).
        """
        # Phase multiplier: higher in later phases
        phase_multipliers = {
            "early": 0.0,
            "mid": 0.25,
            "late": 0.75,
            "endgame": 1.0,
        }
        phase_mult = phase_multipliers.get(game_phase, 0.0)

        if phase_mult <= 0:
            return 0.0

        # Compute position advantage relative to opponents
        my_advantage = self._compute_endgame_position_score(state, player_idx)

        # Find max opponent advantage
        max_opp_advantage = 0.0
        for p in state.players:
            if p.player_number != player_idx:
                opp_advantage = self._compute_endgame_position_score(
                    state, p.player_number
                )
                max_opp_advantage = max(max_opp_advantage, opp_advantage)

        # Symmetric: our advantage minus best opponent
        relative_advantage = my_advantage - max_opp_advantage

        # Apply phase multiplier and weights
        aggression_bonus = (
            relative_advantage
            * phase_mult
            * self.weights.endgame_aggression
        )

        # Additional transition bonus for being ahead when entering late/endgame
        transition_bonus = 0.0
        if relative_advantage > 0 and game_phase in ("late", "endgame"):
            transition_bonus = (
                self.weights.phase_transition_bonus
                * min(relative_advantage / 10.0, 1.0)  # Cap at 1.0
            )

        return aggression_bonus + transition_bonus

    def _compute_endgame_position_score(
        self,
        state: "GameState",
        player_num: int,
    ) -> float:
        """Compute endgame position strength for a player.

        Factors considered:
        - Eliminated rings (progress towards victory)
        - Territory controlled
        - Material advantage (stacks, heights)
        - Capture potential

        Args:
            state: Current game state.
            player_num: Player number.

        Returns:
            Position strength score.
        """
        player = self._get_player(state, player_num)
        if not player:
            return 0.0

        score = 0.0

        # Progress towards ring victory
        eliminated = player.eliminated_rings
        rings_needed = state.victory_threshold - eliminated
        if rings_needed <= 0:
            score += 100.0  # Near victory
        else:
            score += eliminated * 5.0  # Value per eliminated ring

        # Territory progress
        territory = player.territory_spaces
        territory_needed = state.territory_victory_threshold - territory
        if territory_needed <= 0:
            score += 100.0  # Near victory
        else:
            score += territory * 2.0  # Value per territory

        # Material advantage (stack control and capture potential)
        board = state.board
        player_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player == player_num
        ]

        for stack in player_stacks:
            # Value of controlling a stack
            score += 1.0
            # Capture potential (higher cap = more capture power)
            score += stack.cap_height * 0.5

        return score

    # === Compatibility methods for HeuristicAI delegation ===
    # These methods match the original HeuristicAI method signatures
    # to enable gradual migration.
    
    def evaluate_recovery_potential(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate recovery potential feature only.

        Compatibility method matching HeuristicAI._evaluate_recovery_potential
        signature.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Recovery potential score.
        """
        return self._evaluate_recovery_potential(state, player_idx)

    def evaluate_phase_transition(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate phase transition bonus feature only.

        Compatibility method for HeuristicAI delegation.

        Args:
            state: Current game state.
            player_idx: Player number.

        Returns:
            Phase transition score (positive = favorable endgame position).
        """
        game_phase = self.get_game_phase(state)
        return self._evaluate_phase_transition(state, player_idx, game_phase)

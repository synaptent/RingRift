"""
Evaluation Provider Protocol for RingRift AI.

This module defines the evaluation interface that decouples position evaluation
from AI search algorithms. It enables composition over inheritance, allowing
MinimaxAI, MCTSAI, and other tree-search AIs to use evaluation without
inheriting from HeuristicAI.

The design follows the Strategy pattern:
- `EvaluationProvider` - Protocol defining the evaluation interface
- `HeuristicEvaluator` - Concrete implementation using hand-crafted heuristics
- Future: `NNUEEvaluator`, `NeuralEvaluator` for learned evaluation functions

Usage Example:
```python
# Instead of inheritance:
class MinimaxAI(HeuristicAI):  # Inherits 46+ unused methods
    def search(self, state):
        score = self.evaluate_position(state)  # From parent

# Use composition:
class MinimaxAI(BaseAI):
    def __init__(self, player_number, config):
        super().__init__(player_number, config)
        self.evaluator = HeuristicEvaluator(player_number, config)

    def search(self, state):
        score = self.evaluator.evaluate(state)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..models import AIConfig, GameState


@runtime_checkable
class EvaluationProvider(Protocol):
    """Protocol defining the evaluation interface for AI agents.

    This protocol allows any object implementing `evaluate()` to be used
    as an evaluator, enabling flexible composition of evaluation strategies
    with search algorithms.

    Attributes
    ----------
    player_number : int
        The player number this evaluator evaluates positions for.
    """

    player_number: int

    def evaluate(self, game_state: GameState) -> float:
        """Evaluate a position from this player's perspective.

        Parameters
        ----------
        game_state : GameState
            The position to evaluate.

        Returns
        -------
        float
            Evaluation score. Positive values favor this player,
            negative values favor opponents.
        """
        ...

    def get_breakdown(self, game_state: GameState) -> dict[str, float]:
        """Get detailed evaluation breakdown.

        Parameters
        ----------
        game_state : GameState
            The position to evaluate.

        Returns
        -------
        Dict[str, float]
            Mapping of feature names to their contribution scores.
            Should always include a "total" key.
        """
        ...


@dataclass
class EvaluatorConfig:
    """Configuration for evaluation providers.

    This configuration is extracted from AIConfig to provide a clean
    interface for evaluator initialization without coupling to the
    full AIConfig structure.

    Attributes
    ----------
    difficulty : int
        Difficulty level (1-10), affects which weight profile is used.
    eval_mode : str
        Evaluation mode: "full" for all features, "light" for fast subset.
    heuristic_profile_id : Optional[str]
        Explicit weight profile ID, or None to infer from difficulty.
    """

    difficulty: int = 5
    eval_mode: str = "full"
    heuristic_profile_id: str | None = None

    @classmethod
    def from_ai_config(cls, config: AIConfig) -> EvaluatorConfig:
        """Create evaluator config from AIConfig.

        Parameters
        ----------
        config : AIConfig
            Full AI configuration.

        Returns
        -------
        EvaluatorConfig
            Extracted evaluator-specific configuration.
        """
        mode = getattr(config, "heuristic_eval_mode", None)
        return cls(
            difficulty=config.difficulty,
            eval_mode="light" if mode == "light" else "full",
            heuristic_profile_id=getattr(config, "heuristic_profile_id", None),
        )


class HeuristicEvaluator:
    """Hand-crafted heuristic position evaluator.

    This class encapsulates the core evaluation logic from HeuristicAI,
    providing position evaluation without the move selection, caching,
    and other AI-agent-specific functionality.

    The evaluator computes a weighted combination of position features:
    - Tier 0 (core): Stack control, territory, rings in hand, center control
    - Tier 1 (local): Opponent threats, mobility, stack mobility
    - Tier 2 (structural): Line potential, vulnerability, overtake potential
      (only computed in "full" eval_mode)

    Parameters
    ----------
    player_number : int
        The player to evaluate positions for.
    config : EvaluatorConfig or AIConfig
        Configuration for the evaluator.

    Example
    -------
    ```python
    evaluator = HeuristicEvaluator(player_number=1, config=EvaluatorConfig())
    score = evaluator.evaluate(game_state)
    breakdown = evaluator.get_breakdown(game_state)
    ```
    """

    # Default weight constants (can be overridden by profiles)
    WEIGHT_STACK_CONTROL = 10.0
    WEIGHT_STACK_HEIGHT = 5.0
    WEIGHT_CAP_HEIGHT = 6.0
    WEIGHT_TERRITORY = 8.0
    WEIGHT_RINGS_IN_HAND = 3.0
    WEIGHT_CENTER_CONTROL = 4.0
    WEIGHT_OPPONENT_THREAT = 6.0
    WEIGHT_MOBILITY = 4.0
    WEIGHT_ELIMINATED_RINGS = 12.0
    WEIGHT_LINE_POTENTIAL = 7.0
    WEIGHT_VICTORY_PROXIMITY = 20.0
    WEIGHT_MARKER_COUNT = 1.5
    WEIGHT_VULNERABILITY = 8.0
    WEIGHT_OVERTAKE_POTENTIAL = 8.0
    WEIGHT_TERRITORY_CLOSURE = 10.0
    WEIGHT_LINE_CONNECTIVITY = 6.0
    WEIGHT_TERRITORY_SAFETY = 5.0
    WEIGHT_STACK_MOBILITY = 4.0
    WEIGHT_OPPONENT_VICTORY_THREAT = 6.0
    WEIGHT_FORCED_ELIMINATION_RISK = 4.0
    WEIGHT_LPS_ACTION_ADVANTAGE = 2.0
    WEIGHT_MULTI_LEADER_THREAT = 2.0

    # Penalty/bonus weights
    WEIGHT_NO_STACKS_PENALTY = 50.0
    WEIGHT_SINGLE_STACK_PENALTY = 10.0
    WEIGHT_STACK_DIVERSITY_BONUS = 2.0
    WEIGHT_SAFE_MOVE_BONUS = 1.0
    WEIGHT_NO_SAFE_MOVES_PENALTY = 2.0
    WEIGHT_VICTORY_THRESHOLD_BONUS = 1000.0
    WEIGHT_RINGS_PROXIMITY_FACTOR = 50.0
    WEIGHT_TERRITORY_PROXIMITY_FACTOR = 50.0
    WEIGHT_TWO_IN_ROW = 1.0
    WEIGHT_THREE_IN_ROW = 2.0
    WEIGHT_FOUR_IN_ROW = 5.0
    WEIGHT_CONNECTED_NEIGHBOR = 1.0
    WEIGHT_GAP_POTENTIAL = 0.5
    WEIGHT_BLOCKED_STACK_PENALTY = 5.0

    def __init__(
        self,
        player_number: int,
        config: EvaluatorConfig | AIConfig | None = None,
    ) -> None:
        self.player_number = player_number

        # Handle both config types
        if config is None:
            self._config = EvaluatorConfig()
        elif isinstance(config, EvaluatorConfig):
            self._config = config
        else:
            self._config = EvaluatorConfig.from_ai_config(config)

        self.eval_mode = self._config.eval_mode

        # Lazy-loaded dependencies
        self._fast_geo = None
        self._visible_stacks_cache: dict = {}

        # Apply weight profile
        self._apply_weight_profile()

    @property
    def fast_geo(self):
        """Lazily load FastGeometry for efficient board operations."""
        if self._fast_geo is None:
            from .fast_geometry import FastGeometry

            self._fast_geo = FastGeometry.get_instance()
        return self._fast_geo

    def _apply_weight_profile(self) -> None:
        """Apply weight profile based on configuration."""
        from .heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        profile_id = self._config.heuristic_profile_id
        if not profile_id and 1 <= self._config.difficulty <= 10:
            profile_id = f"v1-heuristic-{self._config.difficulty}"

        if not profile_id:
            return

        weights = HEURISTIC_WEIGHT_PROFILES.get(profile_id)
        if weights:
            for name, value in weights.items():
                setattr(self, name, value)

    def evaluate(self, game_state: GameState) -> float:
        """Evaluate position from this player's perspective.

        Parameters
        ----------
        game_state : GameState
            Position to evaluate.

        Returns
        -------
        float
            Evaluation score (positive = good for this player).
        """
        # Check for game over
        if game_state.game_status == "completed":
            if game_state.winner == self.player_number:
                return 100000.0
            elif game_state.winner is not None:
                return -100000.0
            return 0.0

        # Compute component scores
        components = self._compute_components(game_state)
        return sum(components.values())

    def get_breakdown(self, game_state: GameState) -> dict[str, float]:
        """Get detailed evaluation breakdown.

        Parameters
        ----------
        game_state : GameState
            Position to evaluate.

        Returns
        -------
        Dict[str, float]
            Feature scores with "total" key.
        """
        components = self._compute_components(game_state)
        total = sum(components.values())
        return {"total": total, **components}

    def _compute_components(self, game_state: GameState) -> dict[str, float]:
        """Compute per-feature component scores."""
        self._visible_stacks_cache = {}
        scores: dict[str, float] = {}

        # Tier 0 (core) - always computed
        scores["stack_control"] = self._evaluate_stack_control(game_state)
        scores["territory"] = self._evaluate_territory(game_state)
        scores["rings_in_hand"] = self._evaluate_rings_in_hand(game_state)
        scores["center_control"] = self._evaluate_center_control(game_state)

        # Tier 1 (local) - always computed
        scores["opponent_threats"] = self._evaluate_opponent_threats(game_state)
        scores["mobility"] = self._evaluate_mobility(game_state)
        scores["eliminated_rings"] = self._evaluate_eliminated_rings(game_state)
        scores["victory_proximity"] = self._evaluate_victory_proximity(game_state)
        scores["opponent_victory_threat"] = self._evaluate_opponent_victory_threat(
            game_state
        )
        scores["marker_count"] = self._evaluate_marker_count(game_state)
        scores["stack_mobility"] = self._evaluate_stack_mobility(game_state)
        scores["multi_leader_threat"] = self._evaluate_multi_leader_threat(game_state)

        # Tier 2 (structural) - gated by eval_mode
        if self.eval_mode == "full":
            scores["line_potential"] = self._evaluate_line_potential(game_state)
            scores["vulnerability"] = self._evaluate_vulnerability(game_state)
            scores["overtake_potential"] = self._evaluate_overtake_potential(game_state)
            scores["territory_closure"] = self._evaluate_territory_closure(game_state)
            scores["line_connectivity"] = self._evaluate_line_connectivity(game_state)
            scores["territory_safety"] = self._evaluate_territory_safety(game_state)
            scores["forced_elimination_risk"] = self._evaluate_forced_elimination_risk(
                game_state
            )
            scores["lps_action_advantage"] = self._evaluate_lps_action_advantage(
                game_state
            )
        else:
            # Set to 0 in light mode
            for key in [
                "line_potential",
                "vulnerability",
                "overtake_potential",
                "territory_closure",
                "line_connectivity",
                "territory_safety",
                "forced_elimination_risk",
                "lps_action_advantage",
            ]:
                scores[key] = 0.0

        return scores

    # =========================================================================
    # Helper methods (delegated from HeuristicAI)
    # =========================================================================

    def _get_player_info(self, game_state: GameState, player_number: int | None = None):
        """Get player info from game state."""
        target = player_number if player_number is not None else self.player_number
        for p in game_state.players:
            if p.player_number == target:
                return p
        return None

    def _get_adjacent_keys(self, pos_key: str, board_type) -> list:
        """Get adjacent position keys using FastGeometry."""
        return self.fast_geo.get_adjacent_keys(pos_key, board_type)

    def _get_center_positions(self, game_state: GameState):
        """Get center position keys for the board."""
        return self.fast_geo.get_center_positions(game_state.board.type)

    def _iterate_board_keys(self, board) -> list:
        """Get all board position keys."""
        return self.fast_geo.get_all_board_keys(board.type)

    def _victory_proximity_base(self, game_state: GameState, player) -> float:
        """Compute base victory proximity for a player.

        Territory victory uses the dual-condition rule (RR-CANON-R062-v2):
          1. Territory >= floor(totalSpaces / numPlayers) + 1
          2. Territory > sum of all opponents' territory
        """
        lps_player = getattr(game_state, "lps_consecutive_exclusive_player", None)
        lps_rounds = getattr(game_state, "lps_consecutive_exclusive_rounds", 0)
        if lps_player == getattr(player, "player_number", None) and isinstance(
            lps_rounds, int
        ):
            required_rounds = getattr(
                game_state,
                "lps_rounds_required",
                getattr(game_state, "lpsRoundsRequired", 3),
            )
            if not isinstance(required_rounds, int) or required_rounds <= 0:
                required_rounds = 3

            if lps_rounds >= required_rounds and required_rounds >= 1:
                return self.WEIGHT_VICTORY_THRESHOLD_BONUS
            if lps_rounds > 0:
                if required_rounds <= 1:
                    return self.WEIGHT_VICTORY_THRESHOLD_BONUS
                denom = float(required_rounds - 1)
                frac = min(1.0, max(0.0, float(lps_rounds) / denom))
                return self.WEIGHT_VICTORY_THRESHOLD_BONUS * (0.90 + 0.09 * frac)

        rings_needed = game_state.victory_threshold - player.eliminated_rings

        # Territory victory: dual-condition rule (RR-CANON-R062-v2)
        # Get minimum threshold (use new field, fall back to computation for old states)
        from app.rules.core import BOARD_CONFIGS

        board_config = BOARD_CONFIGS.get(game_state.board.type)
        total_spaces = board_config.total_spaces if board_config else 64
        num_players = len(game_state.players)
        territory_minimum = getattr(
            game_state,
            "territory_victory_minimum",
            (total_spaces // num_players) + 1,
        )

        # Calculate opponent territory sum
        player_number = getattr(player, "player_number", None)
        opponent_territory = sum(
            p.territory_spaces
            for p in game_state.players
            if getattr(p, "player_number", None) != player_number
        )

        # Territory needed is the worse of: (1) reaching minimum, (2) exceeding opponents
        territory_needed_for_minimum = territory_minimum - player.territory_spaces
        territory_needed_for_dominance = opponent_territory + 1 - player.territory_spaces
        territory_needed = max(territory_needed_for_minimum, territory_needed_for_dominance)

        if rings_needed <= 0 or territory_needed <= 0:
            return self.WEIGHT_VICTORY_THRESHOLD_BONUS

        score = 0.0
        score += (1.0 / max(1, rings_needed)) * self.WEIGHT_RINGS_PROXIMITY_FACTOR
        score += (
            1.0 / max(1, territory_needed)
        ) * self.WEIGHT_TERRITORY_PROXIMITY_FACTOR
        return score

    # =========================================================================
    # Evaluation features
    # =========================================================================

    def _evaluate_stack_control(self, game_state: GameState) -> float:
        """Evaluate stack control."""
        score = 0.0
        my_stacks = 0
        opponent_stacks = 0
        my_height = 0.0
        opponent_height = 0.0
        my_cap_height = 0
        opponent_cap_height = 0

        for stack in game_state.board.stacks.values():
            h = stack.stack_height
            effective_height = h if h <= 5 else 5 + (h - 5) * 0.1

            if stack.controlling_player == self.player_number:
                my_stacks += 1
                my_height += effective_height
                my_cap_height += stack.cap_height
            else:
                opponent_stacks += 1
                opponent_height += effective_height
                opponent_cap_height += stack.cap_height

        # Stack diversification
        if my_stacks == 0:
            score -= self.WEIGHT_NO_STACKS_PENALTY
        elif my_stacks == 1:
            score -= self.WEIGHT_SINGLE_STACK_PENALTY
        else:
            score += my_stacks * self.WEIGHT_STACK_DIVERSITY_BONUS

        score += (my_stacks - opponent_stacks) * self.WEIGHT_STACK_CONTROL
        score += (my_height - opponent_height) * self.WEIGHT_STACK_HEIGHT
        score += (my_cap_height - opponent_cap_height) * self.WEIGHT_CAP_HEIGHT

        return score

    def _evaluate_territory(self, game_state: GameState) -> float:
        """Evaluate territory control."""
        my_player = self._get_player_info(game_state)
        if not my_player:
            return 0.0

        my_territory = my_player.territory_spaces
        opponent_territory = max(
            (
                p.territory_spaces
                for p in game_state.players
                if p.player_number != self.player_number
            ),
            default=0,
        )

        return (my_territory - opponent_territory) * self.WEIGHT_TERRITORY

    def _evaluate_rings_in_hand(self, game_state: GameState) -> float:
        """Evaluate rings remaining in hand."""
        my_player = self._get_player_info(game_state)
        if not my_player:
            return 0.0
        return my_player.rings_in_hand * self.WEIGHT_RINGS_IN_HAND

    def _evaluate_center_control(self, game_state: GameState) -> float:
        """Evaluate control of center positions."""
        score = 0.0
        center_positions = self._get_center_positions(game_state)

        for pos_key in center_positions:
            if pos_key in game_state.board.stacks:
                stack = game_state.board.stacks[pos_key]
                if stack.controlling_player == self.player_number:
                    score += self.WEIGHT_CENTER_CONTROL
                else:
                    score -= self.WEIGHT_CENTER_CONTROL * 0.5

        return score

    def _evaluate_opponent_threats(self, game_state: GameState) -> float:
        """Evaluate opponent threats near our stacks."""
        score = 0.0
        board = game_state.board
        stacks = board.stacks

        for stack in stacks.values():
            if stack.controlling_player != self.player_number:
                continue

            pos_key = stack.position.to_key()
            for adj_key in self._get_adjacent_keys(pos_key, board.type):
                if adj_key in stacks:
                    adj_stack = stacks[adj_key]
                    if adj_stack.controlling_player != self.player_number:
                        threat_level = adj_stack.cap_height - stack.cap_height
                        score -= threat_level * self.WEIGHT_OPPONENT_THREAT * 0.5

        return score

    def _evaluate_mobility(self, game_state: GameState) -> float:
        """Evaluate pseudo-mobility."""
        board = game_state.board
        stacks = board.stacks
        collapsed = board.collapsed_spaces

        my_mobility = 0
        opp_mobility = 0

        for stack in stacks.values():
            is_mine = stack.controlling_player == self.player_number
            pos_key = stack.position.to_key()

            for adj_key in self._get_adjacent_keys(pos_key, board.type):
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    target_is_mine = target.controlling_player == self.player_number
                    if is_mine != target_is_mine and stack.cap_height >= target.cap_height:
                        if is_mine:
                            my_mobility += 1
                        else:
                            opp_mobility += 1
                else:
                    if is_mine:
                        my_mobility += 1
                    else:
                        opp_mobility += 1

        return (my_mobility - opp_mobility) * self.WEIGHT_MOBILITY

    def _evaluate_eliminated_rings(self, game_state: GameState) -> float:
        """Evaluate eliminated rings."""
        my_player = self._get_player_info(game_state)
        if not my_player:
            return 0.0
        return my_player.eliminated_rings * self.WEIGHT_ELIMINATED_RINGS

    def _evaluate_victory_proximity(self, game_state: GameState) -> float:
        """Evaluate proximity to victory."""
        my_player = self._get_player_info(game_state)
        if not my_player:
            return 0.0
        base = self._victory_proximity_base(game_state, my_player)
        return base * self.WEIGHT_VICTORY_PROXIMITY

    def _evaluate_opponent_victory_threat(self, game_state: GameState) -> float:
        """Evaluate opponent's victory proximity relative to ours."""
        my_player = self._get_player_info(game_state)
        if not my_player:
            return 0.0

        self_prox = self._victory_proximity_base(game_state, my_player)
        max_opp_prox = max(
            (
                self._victory_proximity_base(game_state, p)
                for p in game_state.players
                if p.player_number != self.player_number
            ),
            default=0.0,
        )

        relative_threat = max(0.0, max_opp_prox - self_prox)
        return -relative_threat * self.WEIGHT_OPPONENT_VICTORY_THREAT

    def _evaluate_marker_count(self, game_state: GameState) -> float:
        """Evaluate number of markers."""
        count = sum(
            1
            for m in game_state.board.markers.values()
            if m.player == self.player_number
        )
        return count * self.WEIGHT_MARKER_COUNT

    def _evaluate_stack_mobility(self, game_state: GameState) -> float:
        """Evaluate mobility of individual stacks."""
        score = 0.0
        board = game_state.board
        stacks = board.stacks
        collapsed = board.collapsed_spaces

        for stack in stacks.values():
            if stack.controlling_player != self.player_number:
                continue

            pos_key = stack.position.to_key()
            valid_moves = 0

            for adj_key in self._get_adjacent_keys(pos_key, board.type):
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    if target.controlling_player != self.player_number:
                        if stack.cap_height >= target.cap_height:
                            valid_moves += 1
                    continue
                valid_moves += 1

            score += valid_moves
            if valid_moves == 0:
                score -= self.WEIGHT_BLOCKED_STACK_PENALTY

        return score * self.WEIGHT_STACK_MOBILITY

    def _evaluate_multi_leader_threat(self, game_state: GameState) -> float:
        """Evaluate multi-player leader threat."""
        players = game_state.players
        if len(players) <= 2:
            return 0.0

        opp_prox = sorted(
            (
                self._victory_proximity_base(game_state, p)
                for p in players
                if p.player_number != self.player_number
            ),
            reverse=True,
        )

        if len(opp_prox) < 2:
            return 0.0

        leader_gap = max(0.0, opp_prox[0] - opp_prox[1])
        return -leader_gap * self.WEIGHT_MULTI_LEADER_THREAT

    # Tier 2 features (only in full mode)

    def _evaluate_line_potential(self, game_state: GameState) -> float:
        """Evaluate line-forming potential."""
        from ..models import BoardType

        score = 0.0
        board = game_state.board
        markers = board.markers
        num_directions = 8 if board.type != BoardType.HEXAGONAL else 6

        my_markers = [m for m in markers.values() if m.player == self.player_number]

        for marker in my_markers:
            start_key = marker.position.to_key()

            for dir_idx in range(num_directions):
                key2 = self.fast_geo.offset_key_fast(start_key, dir_idx, 1, board.type)
                if key2 is None:
                    continue

                if key2 in markers and markers[key2].player == self.player_number:
                    score += self.WEIGHT_TWO_IN_ROW

                    key3 = self.fast_geo.offset_key_fast(
                        start_key, dir_idx, 2, board.type
                    )
                    if key3 and key3 in markers and markers[key3].player == self.player_number:
                        score += self.WEIGHT_THREE_IN_ROW

                        key4 = self.fast_geo.offset_key_fast(
                            start_key, dir_idx, 3, board.type
                        )
                        if key4 and key4 in markers and markers[key4].player == self.player_number:
                            score += self.WEIGHT_FOUR_IN_ROW

        return score * self.WEIGHT_LINE_POTENTIAL

    def _evaluate_vulnerability(self, game_state: GameState) -> float:
        """Evaluate vulnerability to overtaking captures."""
        score = 0.0

        for stack in game_state.board.stacks.values():
            if stack.controlling_player != self.player_number:
                continue

            visible = self._get_visible_stacks(stack.position, game_state)
            for adj_stack in visible:
                if adj_stack.controlling_player != self.player_number:
                    if adj_stack.cap_height > stack.cap_height:
                        diff = adj_stack.cap_height - stack.cap_height
                        score -= diff * 1.0

        return score * self.WEIGHT_VULNERABILITY

    def _evaluate_overtake_potential(self, game_state: GameState) -> float:
        """Evaluate ability to overtake opponent stacks."""
        score = 0.0

        for stack in game_state.board.stacks.values():
            if stack.controlling_player != self.player_number:
                continue

            visible = self._get_visible_stacks(stack.position, game_state)
            for adj_stack in visible:
                if adj_stack.controlling_player != self.player_number:
                    if stack.cap_height > adj_stack.cap_height:
                        diff = stack.cap_height - adj_stack.cap_height
                        score += diff * 1.0

        return score * self.WEIGHT_OVERTAKE_POTENTIAL

    def _evaluate_territory_closure(self, game_state: GameState) -> float:
        """Evaluate proximity to enclosing territory."""
        my_markers = [
            m
            for m in game_state.board.markers.values()
            if m.player == self.player_number
        ]

        if not my_markers:
            return 0.0

        # Sample pairs to estimate clustering
        markers_to_check = my_markers[:10] if len(my_markers) > 10 else my_markers

        total_dist = 0.0
        count = 0

        for i, m1 in enumerate(markers_to_check):
            for m2 in markers_to_check[i + 1:]:
                dist = abs(m1.position.x - m2.position.x) + abs(
                    m1.position.y - m2.position.y
                )
                if m1.position.z is not None and m2.position.z is not None:
                    dist += abs(m1.position.z - m2.position.z)
                total_dist += dist
                count += 1

        if count == 0:
            return 0.0

        avg_dist = total_dist / count
        clustering_score = 10.0 / max(1.0, avg_dist)
        marker_count_score = len(my_markers) * 0.5

        return (clustering_score + marker_count_score) * self.WEIGHT_TERRITORY_CLOSURE

    def _evaluate_line_connectivity(self, game_state: GameState) -> float:
        """Evaluate marker connectivity."""
        from ..models import BoardType

        score = 0.0
        board = game_state.board
        markers = board.markers
        collapsed = board.collapsed_spaces
        stacks = board.stacks
        num_directions = 8 if board.type != BoardType.HEXAGONAL else 6

        my_markers = [m for m in markers.values() if m.player == self.player_number]

        for marker in my_markers:
            start_key = marker.position.to_key()

            for dir_idx in range(num_directions):
                key1 = self.fast_geo.offset_key_fast(start_key, dir_idx, 1, board.type)
                if key1 is None:
                    continue

                key2 = self.fast_geo.offset_key_fast(start_key, dir_idx, 2, board.type)

                has_m1 = key1 in markers and markers[key1].player == self.player_number
                has_m2 = (
                    key2 is not None
                    and key2 in markers
                    and markers[key2].player == self.player_number
                )

                if has_m1:
                    score += self.WEIGHT_CONNECTED_NEIGHBOR
                if has_m2 and not has_m1 and key1 not in collapsed and key1 not in stacks:
                    score += self.WEIGHT_GAP_POTENTIAL

        return score * self.WEIGHT_LINE_CONNECTIVITY

    def _evaluate_territory_safety(self, game_state: GameState) -> float:
        """Evaluate safety of potential territories."""
        score = 0.0
        board = game_state.board

        my_markers = [
            m for m in board.markers.values() if m.player == self.player_number
        ]
        opponent_stacks = [
            s
            for s in board.stacks.values()
            if s.controlling_player != self.player_number
        ]

        if not my_markers or not opponent_stacks:
            return 0.0

        for marker in my_markers:
            min_dist = float("inf")
            for stack in opponent_stacks:
                dist = abs(marker.position.x - stack.position.x) + abs(
                    marker.position.y - stack.position.y
                )
                if marker.position.z is not None and stack.position.z is not None:
                    dist += abs(marker.position.z - stack.position.z)
                min_dist = min(min_dist, dist)

            if min_dist <= 2:
                score -= 3.0 - min_dist

        return score * self.WEIGHT_TERRITORY_SAFETY

    def _evaluate_forced_elimination_risk(self, game_state: GameState) -> float:
        """Evaluate forced elimination risk."""
        board = game_state.board
        my_stacks = [
            s
            for s in board.stacks.values()
            if s.controlling_player == self.player_number
        ]
        controlled = len(my_stacks)

        if controlled == 0:
            return 0.0

        approx_actions = self._approx_real_actions(game_state, self.player_number)
        ratio = approx_actions / max(1, controlled)

        if ratio >= 2.0:
            risk_factor = 0.0
        elif ratio >= 1.0:
            risk_factor = 2.0 - ratio
        else:
            risk_factor = 1.0 + (1.0 - ratio)

        return -risk_factor * self.WEIGHT_FORCED_ELIMINATION_RISK

    def _evaluate_lps_action_advantage(self, game_state: GameState) -> float:
        """Evaluate last-player-standing action advantage."""
        players = game_state.players
        if len(players) <= 2:
            return 0.0

        actions = {
            p.player_number: self._approx_real_actions(game_state, p.player_number)
            for p in players
        }
        self_actions = actions.get(self.player_number, 0)
        self_has = self_actions > 0

        opp_with_action = sum(
            1
            for p in players
            if p.player_number != self.player_number
            and actions.get(p.player_number, 0) > 0
        )

        if not self_has:
            advantage = -1.0
        else:
            total_opponents = len(players) - 1
            if total_opponents <= 0:
                advantage = 0.0
            else:
                inactive_fraction = (total_opponents - opp_with_action) / float(
                    total_opponents
                )
                advantage = inactive_fraction

        return advantage * self.WEIGHT_LPS_ACTION_ADVANTAGE

    def _approx_real_actions(self, game_state: GameState, player_number: int) -> int:
        """Approximate available actions for a player."""
        board = game_state.board
        stacks = board.stacks
        collapsed = board.collapsed_spaces
        approx_moves = 0

        for stack in stacks.values():
            if stack.controlling_player != player_number:
                continue

            pos_key = stack.position.to_key()
            has_move = False

            for adj_key in self._get_adjacent_keys(pos_key, board.type):
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    if (
                        target.controlling_player != player_number
                        and stack.cap_height >= target.cap_height
                    ):
                        has_move = True
                        break
                else:
                    has_move = True
                    break

            if has_move:
                approx_moves += 1

        # Check for placement
        player = next(
            (p for p in game_state.players if p.player_number == player_number), None
        )
        if player and player.rings_in_hand > 0:
            has_empty = any(
                key not in stacks and key not in collapsed
                for key in self._iterate_board_keys(board)
            )
            if has_empty:
                approx_moves += 1

        return approx_moves

    def _get_visible_stacks(self, position, game_state: GameState) -> list:
        """Get line-of-sight visible stacks from a position."""
        from ..models import BoardType

        cache_key = position.to_key()
        if cache_key in self._visible_stacks_cache:
            return self._visible_stacks_cache[cache_key]

        visible = []
        board = game_state.board
        stacks = board.stacks
        directions = self.fast_geo.get_los_directions(board.type)

        is_hex = board.type == BoardType.HEXAGONAL
        if is_hex:
            board_size = int(getattr(board, "size", 13) or 13)
            limit = board_size - 1
        else:
            limit = int(getattr(board, "size", 8) or 8)

        curr_x, curr_y = position.x, position.y
        if is_hex:
            curr_z = position.z if position.z is not None else -position.x - position.y
        else:
            curr_z = 0

        for dx, dy, dz in directions:
            x, y, z = curr_x, curr_y, curr_z

            while True:
                x += dx
                y += dy
                if is_hex:
                    z += dz
                    if abs(x) > limit or abs(y) > limit or abs(z) > limit:
                        break
                    pos_key = f"{x},{y},{z}"
                else:
                    if x < 0 or x >= limit or y < 0 or y >= limit:
                        break
                    pos_key = f"{x},{y}"

                stack = stacks.get(pos_key)
                if stack is not None:
                    visible.append(stack)
                    break

        self._visible_stacks_cache[cache_key] = visible
        return visible


# Factory function for easy creation
def create_evaluator(
    player_number: int,
    config: AIConfig | None = None,
    eval_mode: str = "full",
) -> HeuristicEvaluator:
    """Create a HeuristicEvaluator with the given configuration.

    Parameters
    ----------
    player_number : int
        Player to evaluate for.
    config : AIConfig, optional
        AI configuration. If None, uses defaults.
    eval_mode : str
        Evaluation mode: "full" or "light".

    Returns
    -------
    HeuristicEvaluator
        Configured evaluator instance.
    """
    if config:
        evaluator_config = EvaluatorConfig.from_ai_config(config)
        evaluator_config.eval_mode = eval_mode
    else:
        evaluator_config = EvaluatorConfig(eval_mode=eval_mode)

    return HeuristicEvaluator(player_number, evaluator_config)

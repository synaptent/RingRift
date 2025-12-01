"""
Heuristic AI implementation for RingRift
Uses strategic heuristics to evaluate and select moves

When `config.use_incremental_search` is True (the default), this config option
is available for API consistency with other AIs (MinimaxAI, MCTSAI, DescentAI).
However, HeuristicAI is a single-depth evaluator, so the benefit of the
make/unmake pattern is limited compared to tree-search AIs. The option is
stored but not actively used to modify behavior.

When `config.training_move_sample_limit` is set to a positive integer,
HeuristicAI will randomly sample at most that many moves for evaluation
instead of evaluating all valid moves. This is intended for training
performance optimization on large boards where move counts can exceed
thousands. The sampling uses the AI's RNG for deterministic reproducibility.
"""

import random
from typing import Optional, List, Dict

from .base import BaseAI
from ..models import (
    GameState,
    Move,
    RingStack,
    Position,
    AIConfig,
    Player as PlayerState,
)
from ..rules.geometry import BoardGeometry
from .heuristic_weights import HEURISTIC_WEIGHT_PROFILES


def _victory_proximity_base_for_player(
    game_state: GameState,
    player: PlayerState,
) -> float:
    rings_needed = game_state.victory_threshold - player.eliminated_rings
    territory_needed = (
        game_state.territory_victory_threshold - player.territory_spaces
    )

    if rings_needed <= 0 or territory_needed <= 0:
        return 1000.0

    score = 0.0
    score += (1.0 / max(1, rings_needed)) * 50.0
    score += (1.0 / max(1, territory_needed)) * 50.0
    return score


class HeuristicAI(BaseAI):
    """AI that uses heuristics to select strategic moves.
    
    HeuristicAI performs single-depth evaluation of all valid moves and
    selects the best one based on a weighted combination of heuristic
    features. Unlike tree-search AIs (Minimax, MCTS, Descent), it does
    not benefit significantly from the make/unmake pattern since it
    only evaluates one move ahead.
    
    The `use_incremental_search` config option is available for API
    consistency but has limited impact on HeuristicAI performance.
    """
    
    # Evaluation weights for different factors
    WEIGHT_STACK_CONTROL = 10.0
    WEIGHT_STACK_HEIGHT = 5.0
    WEIGHT_TERRITORY = 8.0
    WEIGHT_RINGS_IN_HAND = 3.0
    WEIGHT_CENTER_CONTROL = 4.0
    WEIGHT_ADJACENCY = 2.0
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

    def __init__(self, player_number: int, config: AIConfig):
        """
        Initialise HeuristicAI with an optional heuristic weight profile.

        When ``config.heuristic_profile_id`` is set (typically to the same
        value as the canonical difficulty ``profile_id`` from the ladder,
        e.g. ``"v1-heuristic-5"``), the corresponding entry in
        HEURISTIC_WEIGHT_PROFILES is used to override the class-level weight
        constants for this instance. If no profile is found, the built-in
        defaults defined above are used unchanged to preserve current
        behaviour.
        
        The ``use_incremental_search`` config option is read for API
        consistency with tree-search AIs but has minimal impact on
        HeuristicAI since it only performs single-depth evaluation.

        The optional ``heuristic_eval_mode`` config field controls whether
        this instance runs the full structural heuristic suite (``"full"``)
        or a lighter subset (``"light"``, which skips Tier-2 structural/
        global features entirely). Any value other than the literal string
        ``"light"`` (including ``None``) is normalised to ``"full"`` to
        preserve existing behaviour for callers that do not opt in
        explicitly.
        """
        super().__init__(player_number, config)
        
        # Read use_incremental_search for API consistency with other AIs.
        # Limited benefit for single-depth evaluation but maintains
        # consistent configuration interface across all AI implementations.
        self.use_incremental_search: bool = getattr(
            config, 'use_incremental_search', True
        )

        # Normalise heuristic evaluation mode. Only the literal string
        # "light" opts into the lightweight evaluator; everything else
        # (None, "full", unknown) is treated as "full" for backward
        # compatibility.
        mode = getattr(config, "heuristic_eval_mode", None)
        self.eval_mode: str = "light" if mode == "light" else "full"
        
        self._apply_weight_profile()

    def _apply_weight_profile(self) -> None:
        """Override evaluation weights for this instance from a profile.

        The profile id is taken from ``config.heuristic_profile_id`` when
        provided; otherwise we infer a ladder-aligned id of the form
        ``v1-heuristic-<difficulty>``. The concrete weight vectors live in
        :mod:`app.ai.heuristic_weights` and are looked up via the shared
        ``HEURISTIC_WEIGHT_PROFILES`` registry.

        This is deliberately lightweight: it simply sets attributes like
        ``WEIGHT_STACK_CONTROL`` on the instance, which shadow the class-level
        constants without changing them globally.
        """
        # Prefer an explicit profile id from AIConfig when provided.
        profile_id = getattr(self.config, "heuristic_profile_id", None)

        # If none is provided, attempt to infer from the canonical ladder
        # naming convention for heuristic difficulties (v1-heuristic-2/3/4/5).
        if not profile_id and 1 <= self.config.difficulty <= 10:
            inferred = f"v1-heuristic-{self.config.difficulty}"
            profile_id = inferred

        if not profile_id:
            return

        weights = HEURISTIC_WEIGHT_PROFILES.get(profile_id)
        if not weights:
            return

        for name, value in weights.items():
            setattr(self, name, value)
     
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using heuristic evaluation
        
        Args:
            game_state: Current game state
            
        Returns:
            Best heuristic move or None if no valid moves
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=500, max_ms=1500)
        
        # Get all valid moves using the canonical rules engine
        valid_moves = self.get_valid_moves(game_state)
        
        if not valid_moves:
            return None
        
        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
        else:
            # Apply training move sample limit if configured
            moves_to_evaluate = self._sample_moves_for_training(valid_moves)
            
            # Evaluate each move and pick the best one
            best_move = None
            best_score = float('-inf')
            
            for move in moves_to_evaluate:
                # Simulate the move to get the resulting state
                next_state = self.rules_engine.apply_move(game_state, move)
                score = self.evaluate_position(next_state)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            selected = (
                best_move
                if best_move
                else self.get_random_element(valid_moves)
            )
        
        self.move_count += 1
        return selected

    def _sample_moves_for_training(self, moves: List[Move]) -> List[Move]:
        """
        Sample moves for evaluation if training_move_sample_limit is set.
        
        This is a training/evaluation performance optimization that randomly
        samples a subset of moves when there are too many to evaluate
        efficiently. The sampling is deterministic when the AI's RNG seed
        is set, ensuring reproducibility.
        
        Args:
            moves: Full list of valid moves
            
        Returns:
            Either the original list (if no limit or under limit) or a
            random sample up to the configured limit.
        """
        limit = getattr(self.config, "training_move_sample_limit", None)
        
        # No sampling if limit is not configured or moves are under limit
        if limit is None or limit <= 0 or len(moves) <= limit:
            return moves
        
        # Use the AI's RNG for deterministic sampling when seeded
        rng_seed = getattr(self.config, "rng_seed", None)
        if rng_seed is not None:
            # Use a seeded local RNG for deterministic sampling
            local_rng = random.Random(rng_seed + self.move_count)
            return local_rng.sample(moves, limit)
        else:
            # Fallback to standard random sampling (non-deterministic)
            return random.sample(moves, limit)
    
    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the current position using heuristics
        
        Args:
            game_state: Current game state
            
        Returns:
            Evaluation score (positive = good for this AI)
        """
        # Check for game over first
        if game_state.game_status == "finished":
            if game_state.winner == self.player_number:
                return 100000.0
            elif game_state.winner is not None:
                return -100000.0
            else:
                return 0.0

        # Delegate to the shared component computation so that the scalar
        # evaluation and the per-feature breakdown remain strictly aligned.
        components = self._compute_component_scores(game_state)
        return sum(components.values())

    def _compute_component_scores(
        self,
        game_state: GameState,
    ) -> Dict[str, float]:
        """
        Compute per-feature component scores for the current position.

        This helper centralises mode-aware gating for Tier 0/1/2 features so
        that :meth:`evaluate_position` and :meth:`get_evaluation_breakdown`
        remain consistent. In ``"light"`` mode, Tier-2 structural features are
        not evaluated at all and are reported as ``0.0``.
        """
        scores: Dict[str, float] = {}

        # Tier 0 (core) features – always computed.
        scores["stack_control"] = self._evaluate_stack_control(game_state)
        scores["territory"] = self._evaluate_territory(game_state)
        scores["rings_in_hand"] = self._evaluate_rings_in_hand(game_state)
        scores["center_control"] = self._evaluate_center_control(game_state)

        # Tier 1 (local adjacency/mobility) features – always computed.
        scores["opponent_threats"] = self._evaluate_opponent_threats(
            game_state
        )
        scores["mobility"] = self._evaluate_mobility(game_state)

        # Remaining Tier 0 core features.
        scores["eliminated_rings"] = self._evaluate_eliminated_rings(
            game_state
        )

        # Tier 2 (structural/global) – gated by eval_mode.
        if self.eval_mode == "full":
            scores["line_potential"] = self._evaluate_line_potential(
                game_state
            )
        else:
            scores["line_potential"] = 0.0

        scores["victory_proximity"] = self._evaluate_victory_proximity(
            game_state
        )
        scores["opponent_victory_threat"] = (
            self._evaluate_opponent_victory_threat(game_state)
        )
        scores["marker_count"] = self._evaluate_marker_count(game_state)

        if self.eval_mode == "full":
            scores["vulnerability"] = self._evaluate_vulnerability(game_state)
            scores["overtake_potential"] = self._evaluate_overtake_potential(
                game_state
            )
            scores["territory_closure"] = self._evaluate_territory_closure(
                game_state
            )
            scores["line_connectivity"] = self._evaluate_line_connectivity(
                game_state
            )
            scores["territory_safety"] = self._evaluate_territory_safety(
                game_state
            )
        else:
            scores["vulnerability"] = 0.0
            scores["overtake_potential"] = 0.0
            scores["territory_closure"] = 0.0
            scores["line_connectivity"] = 0.0
            scores["territory_safety"] = 0.0

        # Tier 1 stack-level mobility – always computed.
        scores["stack_mobility"] = self._evaluate_stack_mobility(game_state)

        # High-signal structural features – gated by eval_mode.
        if self.eval_mode == "full":
            scores["forced_elimination_risk"] = (
                self._evaluate_forced_elimination_risk(game_state)
            )
            scores["lps_action_advantage"] = (
                self._evaluate_lps_action_advantage(game_state)
            )
        else:
            scores["forced_elimination_risk"] = 0.0
            scores["lps_action_advantage"] = 0.0

        # Multi-leader threat is considered core and always computed.
        scores["multi_leader_threat"] = self._evaluate_multi_leader_threat(
            game_state
        )

        return scores
     
    def get_evaluation_breakdown(
        self,
        game_state: GameState
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of position evaluation

        Args:
            game_state: Current game state

        Returns:
            Dictionary with evaluation components
        """
        components = self._compute_component_scores(game_state)
        total = sum(components.values())
        breakdown: Dict[str, float] = {"total": total}
        breakdown.update(components)
        return breakdown
    
    def _evaluate_stack_control(self, game_state: GameState) -> float:
        """Evaluate stack control"""
        score = 0.0
        my_stacks = 0
        opponent_stacks = 0
        my_height = 0
        opponent_height = 0
        
        for stack in game_state.board.stacks.values():
            if stack.controlling_player == self.player_number:
                my_stacks += 1
                # Diminishing returns for height > 5 to discourage mega-stacks
                h = stack.stack_height
                effective_height = h if h <= 5 else 5 + (h - 5) * 0.1
                my_height += effective_height
            else:
                opponent_stacks += 1
                h = stack.stack_height
                effective_height = h if h <= 5 else 5 + (h - 5) * 0.1
                opponent_height += effective_height
        
        # Reward having multiple stacks (risk diversification)
        if my_stacks == 0:
            score -= 50.0  # Huge penalty for no stacks
        elif my_stacks == 1:
            score -= 10.0  # Penalty for single stack (vulnerable)
        else:
            score += my_stacks * 2.0
            
        score += (my_stacks - opponent_stacks) * self.WEIGHT_STACK_CONTROL
        score += (my_height - opponent_height) * self.WEIGHT_STACK_HEIGHT
        
        return score
    
    def _evaluate_territory(self, game_state: GameState) -> float:
        """Evaluate territory control"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
        
        my_territory = my_player.territory_spaces
        
        # Compare with opponents
        opponent_territory = 0
        for player in game_state.players:
            if player.player_number != self.player_number:
                opponent_territory = max(
                    opponent_territory,
                    player.territory_spaces
                )

        return (my_territory - opponent_territory) * self.WEIGHT_TERRITORY
    
    def _evaluate_rings_in_hand(self, game_state: GameState) -> float:
        """Evaluate rings remaining in hand"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
        
        # Having rings in hand is good (more placement options)
        return my_player.rings_in_hand * self.WEIGHT_RINGS_IN_HAND
    
    def _evaluate_center_control(self, game_state: GameState) -> float:
        """Evaluate control of center positions"""
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
        """Evaluate opponent threats (stacks near our stacks)"""
        score = 0.0
        my_stacks = [s for s in game_state.board.stacks.values()
                     if s.controlling_player == self.player_number]
        
        for my_stack in my_stacks:
            adjacent = self._get_adjacent_positions(
                my_stack.position,
                game_state
            )
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                if adj_key in game_state.board.stacks:
                    adj_stack = game_state.board.stacks[adj_key]
                    if adj_stack.controlling_player != self.player_number:
                        # Opponent stack adjacent to ours is a threat.
                        # Capture power is based on cap height per compact
                        # rules §10.1, so we compare using cap height here
                        # rather than total stack height.
                        threat_level = (
                            adj_stack.cap_height - my_stack.cap_height
                        )
                        score -= (
                            threat_level * self.WEIGHT_OPPONENT_THREAT * 0.5
                        )

        return score
    
    def _evaluate_mobility(self, game_state: GameState) -> float:
        """Evaluate mobility (number of valid moves)"""
        # Optimization: Use pseudo-mobility instead of full move generation
        # Full move generation is too expensive for evaluation function.
 
        score = 0.0
 
        # My pseudo-mobility
        my_stacks = [
            s for s in game_state.board.stacks.values()
            if s.controlling_player == self.player_number
        ]
        my_mobility = 0
        for stack in my_stacks:
            # Count empty adjacent spaces + adjacent opponent stacks
            # we can capture
            adjacent = BoardGeometry.get_adjacent_positions(
                stack.position,
                game_state.board.type,
                game_state.board.size
            )
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                if adj_key in game_state.board.collapsed_spaces:
                    continue
                if adj_key in game_state.board.stacks:
                    target = game_state.board.stacks[adj_key]
                    if target.controlling_player != self.player_number:
                        # Capture power is based on cap height per compact
                        # rules §10.1, so we compare cap heights here.
                        if stack.cap_height >= target.cap_height:
                            my_mobility += 1
                else:
                    my_mobility += 1
 
        # Opponent pseudo-mobility
        opp_stacks = [
            s for s in game_state.board.stacks.values()
            if s.controlling_player != self.player_number
        ]
        opp_mobility = 0
        for stack in opp_stacks:
            adjacent = BoardGeometry.get_adjacent_positions(
                stack.position,
                game_state.board.type,
                game_state.board.size
            )
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                if adj_key in game_state.board.collapsed_spaces:
                    continue
                if adj_key in game_state.board.stacks:
                    target = game_state.board.stacks[adj_key]
                    if target.controlling_player == self.player_number:
                        # Capture power is based on cap height per compact
                        # rules §10.1, so we compare cap heights here.
                        if stack.cap_height >= target.cap_height:
                            opp_mobility += 1
                else:
                    opp_mobility += 1
        score = (my_mobility - opp_mobility) * self.WEIGHT_MOBILITY
        return score

    def _iterate_board_keys(self, board) -> List[str]:
        """
        Iterate over all logical board coordinate keys for the given board.

        This delegates to BoardManager._generate_all_positions_for_board to
        stay aligned with the shared rules/territory logic.
        """
        from ..board_manager import BoardManager

        keys: List[str] = []
        for pos in BoardManager._generate_all_positions_for_board(board):
            keys.append(pos.to_key())
        return keys

    def _approx_real_actions_for_player(
        self,
        game_state: GameState,
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
        """
        board = game_state.board
        approx_moves = 0

        for stack in board.stacks.values():
            if stack.controlling_player != player_number:
                continue

            adj_positions = self._get_adjacent_positions(
                stack.position,
                game_state,
            )
            stack_has_any_move = False
            for pos in adj_positions:
                key = pos.to_key()
                if key in board.collapsed_spaces:
                    continue

                if key in board.stacks:
                    target = board.stacks[key]
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
                for p in game_state.players
                if p.player_number == player_number
            ),
            None,
        )
        if player and player.rings_in_hand > 0:
            has_empty = any(
                key not in board.stacks and key not in board.collapsed_spaces
                for key in self._iterate_board_keys(board)
            )
            if has_empty:
                approx_placement = 1

        return approx_moves + approx_placement

    def _evaluate_influence(self, game_state: GameState) -> float:
        """Evaluate board influence"""
        board = game_state.board

        # Influence map: +1 for my stack, -1 for opponent stack
        # Decay by distance

        # Get all valid positions
        # We don't have a direct method to get all positions in BoardManager
        # exposed here easily without re-implementing.
        # Let's iterate over stacks and project influence.

        influence_map = {}

        for stack in board.stacks.values():
            if stack.controlling_player == self.player_number:
                value = 1.0
            else:
                value = -1.0
            # Base influence at stack position
            pos_key = stack.position.to_key()
            influence_map[pos_key] = (
                influence_map.get(pos_key, 0) + value * 2.0
            )

            # Project to neighbors (distance 1)
            neighbors = self._get_adjacent_positions(
                stack.position,
                game_state
            )
            for n in neighbors:
                n_key = n.to_key()
                influence_map[n_key] = (
                    influence_map.get(n_key, 0) + value * 1.0
                )

                # Project to distance 2 (simplified, only for hex or if needed)
                # For now distance 1 is enough for "control"

        # Sum up positive influence (my control) vs negative (opponent control)
        total_influence = sum(influence_map.values())

        return total_influence * 2.0  # Weight for influence

    def _evaluate_eliminated_rings(self, game_state: GameState) -> float:
        """Evaluate eliminated rings"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
            
        return my_player.eliminated_rings * self.WEIGHT_ELIMINATED_RINGS

    def _evaluate_line_potential(self, game_state: GameState) -> float:
        """Evaluate potential to form lines"""
        score = 0.0
        board = game_state.board

        # Use BoardManager logic to find directions
        from ..board_manager import BoardManager
        directions = BoardManager._get_all_directions(board.type)

        # Iterate through all markers of the player
        my_markers = [
            m for m in board.markers.values()
            if m.player == self.player_number
        ]

        for marker in my_markers:
            start_pos = marker.position

            for direction in directions:
                # Check for 2 or 3 markers in a row
                # We only check forward to avoid double counting (mostly)

                # Check length 2
                pos2 = BoardManager._add_direction(start_pos, direction, 1)
                key2 = pos2.to_key()

                if (key2 in board.markers and
                        board.markers[key2].player == self.player_number):
                    score += 1.0  # 2 in a row

                    # Check length 3
                    pos3 = BoardManager._add_direction(start_pos, direction, 2)
                    key3 = pos3.to_key()

                    if (key3 in board.markers and
                            board.markers[key3].player == self.player_number):
                        score += 2.0  # 3 in a row (cumulative with 2)

                        # Check length 4 (almost a line)
                        pos4 = BoardManager._add_direction(
                            start_pos, direction, 3
                        )
                        key4 = pos4.to_key()

                        if (key4 in board.markers and
                                board.markers[key4].player ==
                                self.player_number):
                            score += 5.0  # 4 in a row

        return score * self.WEIGHT_LINE_POTENTIAL

    def _evaluate_victory_proximity(self, game_state: GameState) -> float:
        """Evaluate how close we are to winning"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        base = _victory_proximity_base_for_player(game_state, my_player)
        return base * self.WEIGHT_VICTORY_PROXIMITY

    def _evaluate_opponent_victory_threat(
        self,
        game_state: GameState,
    ) -> float:
        """
        Evaluate how much closer the leading opponent is to victory
        than we are.

        This mirrors the self victory proximity computation and
        compares our proximity score to the maximum proximity score
        among all opponents. A positive gap is treated as a threat
        and converted into a penalty.
        """
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        self_prox = _victory_proximity_base_for_player(game_state, my_player)

        max_opp_prox = 0.0
        for p in game_state.players:
            if p.player_number == self.player_number:
                continue
            prox = _victory_proximity_base_for_player(game_state, p)
            if prox > max_opp_prox:
                max_opp_prox = prox

        raw_gap = max_opp_prox - self_prox
        relative_threat = max(0.0, raw_gap)

        return -relative_threat * self.WEIGHT_OPPONENT_VICTORY_THREAT

    def _evaluate_marker_count(self, game_state: GameState) -> float:
        """Evaluate number of markers on board"""
        my_markers = 0
        for marker in game_state.board.markers.values():
            if marker.player == self.player_number:
                my_markers += 1
                
        return my_markers * self.WEIGHT_MARKER_COUNT

    def _get_visible_stacks(
        self,
        position: Position,
        game_state: GameState,
    ) -> List[RingStack]:
        """
        Compute line-of-sight visible stacks from a position.
        """
        visible: List[RingStack] = []
        board = game_state.board
        board_type = board.type
        size = board.size

        directions = BoardGeometry.get_line_of_sight_directions(board_type)

        for dx, dy, dz in directions:
            curr_x = position.x
            curr_y = position.y
            curr_z = position.z or 0

            while True:
                curr_x += dx
                curr_y += dy
                curr_z += dz

                curr_pos = Position(x=curr_x, y=curr_y, z=curr_z)
                if not BoardGeometry.is_within_bounds(
                    curr_pos, board_type, size
                ):
                    break

                pos_key = curr_pos.to_key()
                stack = board.stacks.get(pos_key)
                if stack is not None:
                    visible.append(stack)
                    break

        return visible

    def _evaluate_vulnerability(self, game_state: GameState) -> float:
        """
        Evaluate vulnerability of our stacks to overtaking captures.
        Considers relative cap heights of stacks in clear line of sight.
        """
        score = 0.0
        my_stacks = [
            s
            for s in game_state.board.stacks.values()
            if s.controlling_player == self.player_number
        ]

        for stack in my_stacks:
            visible_stacks = self._get_visible_stacks(
                stack.position,
                game_state
            )

            for adj_stack in visible_stacks:
                if adj_stack.controlling_player != self.player_number:
                    # Capture power is based on cap height per compact rules
                    # §10.1, so we compare using cap height here.
                    if adj_stack.cap_height > stack.cap_height:
                        diff = adj_stack.cap_height - stack.cap_height
                        score -= diff * 1.0

        return score * self.WEIGHT_VULNERABILITY

    def _evaluate_overtake_potential(self, game_state: GameState) -> float:
        """
        Evaluate our ability to overtake opponent stacks.
        Considers relative cap heights of stacks in clear line of sight.
        """
        score = 0.0
        my_stacks = [
            s
            for s in game_state.board.stacks.values()
            if s.controlling_player == self.player_number
        ]

        for stack in my_stacks:
            visible_stacks = self._get_visible_stacks(
                stack.position,
                game_state
            )

            for adj_stack in visible_stacks:
                if adj_stack.controlling_player != self.player_number:
                    # Capture power is based on cap height per compact rules
                    # §10.1, so we compare using cap height here.
                    if stack.cap_height > adj_stack.cap_height:
                        diff = stack.cap_height - adj_stack.cap_height
                        score += diff * 1.0

        return score * self.WEIGHT_OVERTAKE_POTENTIAL

    def _evaluate_territory_closure(self, game_state: GameState) -> float:
        """
        Evaluate how close we are to enclosing a territory.
        Uses a simplified metric: number of markers vs board size/density.
        """
        # This is a complex heuristic to implement perfectly without
        # pathfinding. As a proxy, we look at marker density and clustering.

        my_markers = [m for m in game_state.board.markers.values()
                      if m.player == self.player_number]

        if not my_markers:
            return 0.0

        # Calculate "clustering" - average distance between markers
        # Closer markers are more likely to form a closed loop
        total_dist = 0.0
        count = 0

        # Sample a few pairs to estimate density if too many markers
        # Deterministic subsampling: take first 10
        if len(my_markers) < 10:
            markers_to_check = my_markers
        else:
            markers_to_check = my_markers[:10]

        for i, m1 in enumerate(markers_to_check):
            for m2 in markers_to_check[i+1:]:
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
        # We invert it for the score
        clustering_score = 10.0 / max(1.0, avg_dist)

        # Also reward total number of markers as a prerequisite for territory
        marker_count_score = len(my_markers) * 0.5

        return (
            (clustering_score + marker_count_score) *
            self.WEIGHT_TERRITORY_CLOSURE
        )

    # _evaluate_move is deprecated in favor of evaluate_position
    # on the simulated state
    
    def _get_center_positions(self, game_state: GameState) -> set:
        """Get center position keys for the board"""
        return BoardGeometry.get_center_positions(
            game_state.board.type,
            game_state.board.size
        )
    
    def _evaluate_line_connectivity(self, game_state: GameState) -> float:
        """
        Evaluate connectivity of markers for potential lines.
        Rewards markers that are close to each other in line-forming
        directions.
        """
        score = 0.0
        board = game_state.board
        from ..board_manager import BoardManager
        directions = BoardManager._get_all_directions(board.type)

        my_markers = [
            m for m in board.markers.values()
            if m.player == self.player_number
        ]

        for marker in my_markers:
            start_pos = marker.position
            for direction in directions:
                # Check distance 1 and 2 in each direction
                # If we have a marker at dist 2 but not 1, it's a "gap"
                # that can be filled
                pos1 = BoardManager._add_direction(start_pos, direction, 1)
                key1 = pos1.to_key()
                pos2 = BoardManager._add_direction(start_pos, direction, 2)
                key2 = pos2.to_key()

                has_m1 = (
                    key1 in board.markers and
                    board.markers[key1].player == self.player_number
                )
                has_m2 = (
                    key2 in board.markers and
                    board.markers[key2].player == self.player_number
                )

                if has_m1:
                    score += 1.0  # Connected neighbor
                if has_m2 and not has_m1:
                    # Gap of 1, potential to connect
                    # Check if gap is empty or has opponent marker (flippable)
                    if (key1 not in board.collapsed_spaces and
                            key1 not in board.stacks):
                        score += 0.5

        return score * self.WEIGHT_LINE_CONNECTIVITY

    def _evaluate_territory_safety(self, game_state: GameState) -> float:
        """
        Evaluate safety of potential territories.
        Penalizes if opponent has stacks near our marker clusters.
        """
        score = 0.0
        board = game_state.board

        # Identify clusters of markers (simplified)
        # For each of my markers, check distance to nearest opponent stack
        my_markers = [
            m for m in board.markers.values()
            if m.player == self.player_number
        ]
        opponent_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player != self.player_number
        ]

        if not my_markers or not opponent_stacks:
            return 0.0

        for marker in my_markers:
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

        return score * self.WEIGHT_TERRITORY_SAFETY

    def _evaluate_stack_mobility(self, game_state: GameState) -> float:
        """
        Evaluate mobility of individual stacks.
        A stack that is surrounded by collapsed spaces or board edges
        has low mobility.
        """
        score = 0.0
        my_stacks = [
            s for s in game_state.board.stacks.values()
            if s.controlling_player == self.player_number
        ]
 
        for stack in my_stacks:
            adjacent = self._get_adjacent_positions(stack.position, game_state)
            valid_moves_from_here = 0
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                # Check if blocked by collapsed space
                if adj_key in game_state.board.collapsed_spaces:
                    continue
                # Check if blocked by stack (unless capture possible)
                if adj_key in game_state.board.stacks:
                    target = game_state.board.stacks[adj_key]
                    if target.controlling_player != self.player_number:
                        # Capture power is based on cap height per compact
                        # rules §10.1, so we compare using cap height here.
                        if stack.cap_height >= target.cap_height:
                            valid_moves_from_here += 1
                    continue
 
                valid_moves_from_here += 1
 
            # Reward stacks with more freedom
            score += valid_moves_from_here
 
            # Penalty for completely blocked stacks (dead weight)
            if valid_moves_from_here == 0:
                score -= 5.0
 
        return score * self.WEIGHT_STACK_MOBILITY

    def _evaluate_forced_elimination_risk(
        self,
        game_state: GameState,
    ) -> float:
        """
        Penalise positions where we control many stacks but have very few
        real actions (moves or placements), indicating forced-elimination risk.
        """
        board = game_state.board
        my_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player == self.player_number
        ]
        controlled_stacks = len(my_stacks)
        if controlled_stacks == 0:
            return 0.0

        approx_actions = self._approx_real_actions_for_player(
            game_state,
            self.player_number,
        )
        ratio = approx_actions / max(1, controlled_stacks)

        if ratio >= 2.0:
            risk_factor = 0.0
        elif ratio >= 1.0:
            risk_factor = 2.0 - ratio
        else:
            risk_factor = 1.0 + (1.0 - ratio)

        return -risk_factor * self.WEIGHT_FORCED_ELIMINATION_RISK

    def _evaluate_lps_action_advantage(
        self,
        game_state: GameState,
    ) -> float:
        """
        Last-player-standing action advantage heuristic.

        In 3+ player games, reward being one of the few players with real
        actions left and penalise being the only player without actions.
        """
        players = game_state.players
        if len(players) <= 2:
            return 0.0

        actions = {
            p.player_number: self._approx_real_actions_for_player(
                game_state,
                p.player_number,
            )
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
                inactive_fraction = (
                    total_opponents - opp_with_action
                ) / float(total_opponents)
                advantage = inactive_fraction

        return advantage * self.WEIGHT_LPS_ACTION_ADVANTAGE

    def _evaluate_multi_leader_threat(
        self,
        game_state: GameState,
    ) -> float:
        """
        Multi-player leader threat heuristic.

        In 3+ player games, penalise positions where a single opponent is
        much closer to victory than the other opponents.
        """
        players = game_state.players
        if len(players) <= 2:
            return 0.0

        prox_by_player = {
            p.player_number: _victory_proximity_base_for_player(game_state, p)
            for p in players
        }

        opp_prox = [
            prox_by_player[p.player_number]
            for p in players
            if p.player_number != self.player_number
        ]
        if len(opp_prox) < 2:
            return 0.0

        opp_prox_sorted = sorted(opp_prox, reverse=True)
        opp_top1 = opp_prox_sorted[0]
        opp_top2 = opp_prox_sorted[1]

        leader_gap = max(0.0, opp_top1 - opp_top2)

        return -leader_gap * self.WEIGHT_MULTI_LEADER_THREAT
 
    def _get_adjacent_positions(
        self,
        position: Position,
        game_state: GameState
    ) -> List[Position]:
        """Get adjacent positions around a position."""
        return BoardGeometry.get_adjacent_positions(
            position,
            game_state.board.type,
            game_state.board.size
        )

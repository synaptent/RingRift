"""
Heuristic AI implementation for RingRift
Uses strategic heuristics to evaluate and select moves
"""

from typing import Optional, List, Dict
import random

from .base import BaseAI
from ..models import GameState, Move, RingStack, Position
from ..rules.geometry import BoardGeometry


class HeuristicAI(BaseAI):
    """AI that uses heuristics to select strategic moves"""
    
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
            selected = random.choice(valid_moves)
        else:
            # Evaluate each move and pick the best one
            best_move = None
            best_score = float('-inf')
            
            for move in valid_moves:
                # Simulate the move to get the resulting state
                next_state = self.rules_engine.apply_move(game_state, move)
                score = self.evaluate_position(next_state)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            selected = best_move if best_move else random.choice(valid_moves)
        
        self.move_count += 1
        return selected
    
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

        score = 0.0
        
        # Stack control evaluation
        score += self._evaluate_stack_control(game_state)
        
        # Territory evaluation
        score += self._evaluate_territory(game_state)
        
        # Rings in hand evaluation
        score += self._evaluate_rings_in_hand(game_state)
        
        # Center control evaluation
        score += self._evaluate_center_control(game_state)
        
        # Opponent threat evaluation
        score += self._evaluate_opponent_threats(game_state)
        
        # Mobility evaluation
        score += self._evaluate_mobility(game_state)
        
        # Eliminated rings evaluation
        score += self._evaluate_eliminated_rings(game_state)
        
        # Line potential evaluation
        score += self._evaluate_line_potential(game_state)
        
        # Victory proximity evaluation
        score += self._evaluate_victory_proximity(game_state)
        
        # Marker count evaluation
        score += self._evaluate_marker_count(game_state)

        # Vulnerability evaluation
        score += self._evaluate_vulnerability(game_state)

        # Overtake potential evaluation
        score += self._evaluate_overtake_potential(game_state)

        # Territory closure evaluation
        score += self._evaluate_territory_closure(game_state)

        # Influence evaluation
        score += self._evaluate_influence(game_state)

        # Line connectivity evaluation
        score += self._evaluate_line_connectivity(game_state)

        # Territory safety evaluation
        score += self._evaluate_territory_safety(game_state)

        # Stack mobility evaluation
        score += self._evaluate_stack_mobility(game_state)
        
        return score
    
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
        return {
            "total": self.evaluate_position(game_state),
            "stack_control": self._evaluate_stack_control(game_state),
            "territory": self._evaluate_territory(game_state),
            "rings_in_hand": self._evaluate_rings_in_hand(game_state),
            "center_control": self._evaluate_center_control(game_state),
            "opponent_threats": self._evaluate_opponent_threats(game_state),
            "mobility": self._evaluate_mobility(game_state),
            "eliminated_rings": self._evaluate_eliminated_rings(game_state),
            "line_potential": self._evaluate_line_potential(game_state),
            "victory_proximity": self._evaluate_victory_proximity(game_state),
            "marker_count": self._evaluate_marker_count(game_state),
            "vulnerability": self._evaluate_vulnerability(game_state),
            "overtake_potential": self._evaluate_overtake_potential(
                game_state
            ),
            "territory_closure": self._evaluate_territory_closure(game_state),
            "line_connectivity": self._evaluate_line_connectivity(game_state),
            "territory_safety": self._evaluate_territory_safety(game_state),
            "stack_mobility": self._evaluate_stack_mobility(game_state)
        }
    
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
                        # Opponent stack adjacent to ours is a threat
                        threat_level = (
                            adj_stack.stack_height - my_stack.stack_height
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
                    if (target.controlling_player != self.player_number and
                            stack.stack_height > target.stack_height):
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
                    if (target.controlling_player == self.player_number and
                            stack.stack_height > target.stack_height):
                        opp_mobility += 1
                else:
                    opp_mobility += 1

        score = (my_mobility - opp_mobility) * self.WEIGHT_MOBILITY
        return score

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

        # Ring elimination victory
        rings_needed = (
            game_state.victory_threshold - my_player.eliminated_rings
        )
        if rings_needed <= 0:
            return 1000.0  # Winning state

        # Territory victory
        territory_needed = (
            game_state.territory_victory_threshold - my_player.territory_spaces
        )
        if territory_needed <= 0:
            return 1000.0  # Winning state

        score = 0.0
        score += (1.0 / max(1, rings_needed)) * 50.0
        score += (1.0 / max(1, territory_needed)) * 50.0

        return score * self.WEIGHT_VICTORY_PROXIMITY

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
                    # If opponent stack is taller, we are vulnerable
                    if adj_stack.stack_height > stack.stack_height:
                        diff = adj_stack.stack_height - stack.stack_height
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
                    # If our stack is taller, we can overtake
                    if stack.stack_height > adj_stack.stack_height:
                        diff = stack.stack_height - adj_stack.stack_height
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
                        if stack.stack_height > target.stack_height:
                            valid_moves_from_here += 1
                    continue

                valid_moves_from_here += 1

            # Reward stacks with more freedom
            score += valid_moves_from_here

            # Penalty for completely blocked stacks (dead weight)
            if valid_moves_from_here == 0:
                score -= 5.0

        return score * self.WEIGHT_STACK_MOBILITY

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

"""
Heuristic AI implementation for RingRift
Uses strategic heuristics to evaluate and select moves
"""

from typing import Optional, List, Dict, Any
import random
from datetime import datetime
import uuid

from .base import BaseAI
from ..models import GameState, Move, AIConfig, Position, RingStack


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
        
        # Get all valid moves using the new GameEngine
        from ..game_engine import GameEngine
        valid_moves = GameEngine.get_valid_moves(game_state, self.player_number)
        
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
                next_state = GameEngine.apply_move(game_state, move)
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
        
        return score
    
    def get_evaluation_breakdown(self, game_state: GameState) -> Dict[str, float]:
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
            "overtake_potential": self._evaluate_overtake_potential(game_state),
            "territory_closure": self._evaluate_territory_closure(game_state)
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
                my_height += stack.stack_height
            else:
                opponent_stacks += 1
                opponent_height += stack.stack_height
        
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
                opponent_territory = max(opponent_territory, player.territory_spaces)
        
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
            adjacent = self._get_adjacent_positions(my_stack.position, game_state)
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                if adj_key in game_state.board.stacks:
                    adj_stack = game_state.board.stacks[adj_key]
                    if adj_stack.controlling_player != self.player_number:
                        # Opponent stack adjacent to ours is a threat
                        threat_level = adj_stack.stack_height - my_stack.stack_height
                        score -= threat_level * self.WEIGHT_OPPONENT_THREAT * 0.5
        
        return score
    
    def _evaluate_mobility(self, game_state: GameState) -> float:
        """Evaluate mobility (number of valid moves)"""
        # This is computationally expensive, so we approximate it
        # by counting stacks that are not blocked
        score = 0.0
        my_stacks = [s for s in game_state.board.stacks.values()
                     if s.controlling_player == self.player_number]
        
        for stack in my_stacks:
            adjacent = self._get_adjacent_positions(stack.position, game_state)
            valid_moves = 0
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                # Check if position is blocked by collapsed space or stack
                if adj_key not in game_state.board.collapsed_spaces and adj_key not in game_state.board.stacks:
                    valid_moves += 1
            
            score += valid_moves * 0.5
            
        return score * self.WEIGHT_MOBILITY

    def _evaluate_eliminated_rings(self, game_state: GameState) -> float:
        """Evaluate eliminated rings"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
            
        return my_player.eliminated_rings * self.WEIGHT_ELIMINATED_RINGS

    def _evaluate_line_potential(self, game_state: GameState) -> float:
        """Evaluate potential to form lines"""
        # Simplified check for markers in a row
        score = 0.0
        # TODO: Implement proper line detection logic
        # For now, just count markers as a proxy
        return score * self.WEIGHT_LINE_POTENTIAL

    def _evaluate_victory_proximity(self, game_state: GameState) -> float:
        """Evaluate how close we are to winning"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
            
        # Ring elimination victory
        rings_needed = game_state.victory_threshold - my_player.eliminated_rings
        if rings_needed <= 0:
            return 1000.0 # Winning state
            
        # Territory victory
        territory_needed = game_state.territory_victory_threshold - my_player.territory_spaces
        if territory_needed <= 0:
            return 1000.0 # Winning state
            
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

    def _evaluate_vulnerability(self, game_state: GameState) -> float:
        """
        Evaluate vulnerability of our stacks to overtaking captures.
        Considers relative cap heights of stacks in clear line of sight.
        """
        score = 0.0
        my_stacks = [s for s in game_state.board.stacks.values()
                     if s.controlling_player == self.player_number]
        
        from ..game_engine import GameEngine

        for stack in my_stacks:
            # Use GameEngine's visibility logic for true line-of-sight
            visible_stacks = GameEngine.get_visible_stacks(stack.position, game_state)
            
            for adj_stack in visible_stacks:
                if adj_stack.controlling_player != self.player_number:
                    # If opponent stack is taller, we are vulnerable
                    if adj_stack.stack_height > stack.stack_height:
                        # Higher penalty for larger height difference
                        diff = adj_stack.stack_height - stack.stack_height
                        score -= diff * 1.0
        
        return score * self.WEIGHT_VULNERABILITY

    def _evaluate_overtake_potential(self, game_state: GameState) -> float:
        """
        Evaluate our ability to overtake opponent stacks.
        Considers relative cap heights of stacks in clear line of sight.
        """
        score = 0.0
        my_stacks = [s for s in game_state.board.stacks.values()
                     if s.controlling_player == self.player_number]
        
        from ..game_engine import GameEngine

        for stack in my_stacks:
            # Use GameEngine's visibility logic for true line-of-sight
            visible_stacks = GameEngine.get_visible_stacks(stack.position, game_state)
            
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
        # This is a complex heuristic to implement perfectly without pathfinding.
        # As a proxy, we look at marker density and clustering.
        
        my_markers = [m for m in game_state.board.markers.values()
                      if m.player == self.player_number]
        
        if not my_markers:
            return 0.0
            
        # Calculate "clustering" - average distance between markers
        # Closer markers are more likely to form a closed loop
        total_dist = 0.0
        count = 0
        
        # Sample a few pairs to estimate density if too many markers
        markers_to_check = my_markers if len(my_markers) < 10 else random.sample(my_markers, 10)
        
        for i, m1 in enumerate(markers_to_check):
            for m2 in markers_to_check[i+1:]:
                dist = abs(m1.position.x - m2.position.x) + abs(m1.position.y - m2.position.y)
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
        
        return (clustering_score + marker_count_score) * self.WEIGHT_TERRITORY_CLOSURE

    # _evaluate_move is deprecated in favor of evaluate_position on the simulated state
    
    def _get_center_positions(self, game_state: GameState) -> set:
        """Get center position keys for the board"""
        center = set()
        board_type = game_state.board.type
        size = game_state.board.size
        
        if board_type.value == "square8":
            # Center 2x2 of 8x8 board
            for x in [3, 4]:
                for y in [3, 4]:
                    center.add(f"{x},{y}")
        
        elif board_type.value == "square19":
            # Center 3x3 of 19x19 board
            for x in [8, 9, 10]:
                for y in [8, 9, 10]:
                    center.add(f"{x},{y}")
        
        elif board_type.value == "hexagonal":
            # Center hexagon (distance 0-2 from origin)
            for x in range(-2, 3):
                for y in range(-2, 3):
                    z = -x - y
                    if abs(x) <= 2 and abs(y) <= 2 and abs(z) <= 2:
                        center.add(f"{x},{y},{z}")
        
        return center
    
    def _get_adjacent_positions(self, position: Position, game_state: GameState) -> List[Position]:
        """Get adjacent positions (simplified version from RandomAI)"""
        from .random_ai import RandomAI
        temp_random_ai = RandomAI(self.player_number, self.config)
        return temp_random_ai._get_adjacent_positions(position, game_state)

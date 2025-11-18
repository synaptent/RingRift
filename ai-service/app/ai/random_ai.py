"""
Random AI implementation for RingRift
Selects moves randomly from valid options
"""

from typing import Optional, List, Dict, Any
import random
from datetime import datetime
import uuid

from .base import BaseAI
from ..models import GameState, Move, AIConfig, Position


class RandomAI(BaseAI):
    """AI that selects random valid moves"""
    
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select a random valid move
        
        Args:
            game_state: Current game state
            
        Returns:
            Random valid move or None if no valid moves
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=200, max_ms=800)
        
        # Get all valid moves
        valid_moves = self._get_valid_moves_for_phase(game_state)
        
        if not valid_moves:
            return None
        
        # Select random move
        selected = random.choice(valid_moves)
        
        # Convert to Move object
        move = self._create_move_object(selected, game_state)
        
        self.move_count += 1
        return move
    
    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate position (random AI doesn't really evaluate, returns neutral)
        
        Args:
            game_state: Current game state
            
        Returns:
            0.0 (neutral evaluation)
        """
        # Random AI doesn't evaluate positions
        # Return small random value to simulate variance
        return random.uniform(-0.1, 0.1)
    
    def get_evaluation_breakdown(self, game_state: GameState) -> Dict[str, float]:
        """
        Get evaluation breakdown for random AI
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with random evaluation
        """
        return {
            "total": 0.0,
            "random_variance": random.uniform(-0.1, 0.1)
        }
    
    def _get_valid_moves_for_phase(self, game_state: GameState) -> List[Dict[str, Any]]:
        """
        Get valid moves based on current game phase
        
        Args:
            game_state: Current game state
            
        Returns:
            List of valid move dictionaries
        """
        phase = game_state.current_phase
        
        if phase.value == "ring_placement":
            return self._get_ring_placement_moves(game_state)
        elif phase.value == "movement":
            return self._get_movement_moves(game_state)
        elif phase.value == "capture":
            return self._get_capture_moves(game_state)
        else:
            # For other phases, return empty list
            return []
    
    def _get_ring_placement_moves(self, game_state: GameState) -> List[Dict[str, Any]]:
        """Get candidate ring placement moves for the current player.

        This helper intentionally mirrors the *shape* of legal placements
        rather than enforcing all RingRift rules itself; the authoritative
        validation is still performed by the backend RuleEngine. To keep
        behaviour aligned with the updated placement semantics, we:

        - Allow placement on both empty spaces and existing stacks.
        - Never generate placements on collapsed spaces.
        - Use the canonical "place_ring" move type so the backend does
          not need to special-case legacy "place" moves from the AI
          service.

        The exact number of rings placed (multi-ring vs single-ring) is
        decided on the server side via AIEngine.normalizeServiceMove,
        which consults the player’s ringsInHand and board occupancy.
        """
        valid_moves: List[Dict[str, Any]] = []

        board = game_state.board

        # Positions that are completely unavailable for placement:
        # collapsed territory spaces. Stacks are allowed targets for
        # stacking placements and are therefore not excluded here.
        blocked = set(board.collapsed_spaces.keys())

        # Generate all geometrically valid positions for this board and
        # filter out collapsed spaces only. The backend will apply
        # no-dead-placement and ring-count rules.
        all_positions = self._generate_all_positions(board.type, board.size)

        for pos in all_positions:
            pos_key = pos.to_key()
            if pos_key in blocked:
                continue

            valid_moves.append({
                "type": "place_ring",
                "to": pos,
                "from": None
            })

        return valid_moves
    
    def _get_movement_moves(self, game_state: GameState) -> List[Dict[str, Any]]:
        """
        Get valid movement moves
        
        Args:
            game_state: Current game state
            
        Returns:
            List of valid movement moves
        """
        valid_moves = []
        
        # Find all stacks controlled by this player
        for pos_key, stack in game_state.board.stacks.items():
            if stack.controlling_player == self.player_number:
                # Get adjacent positions (simplified - should use proper adjacency rules)
                from_pos = stack.position
                adjacent_positions = self._get_adjacent_positions(from_pos, game_state)
                
                for to_pos in adjacent_positions:
                    to_key = to_pos.to_key()
                    # Check if position is valid for movement
                    if to_key not in game_state.board.collapsed_spaces:
                        valid_moves.append({
                            "type": "move",
                            "from": from_pos,
                            "to": to_pos
                        })
        
        return valid_moves
    
    def _get_capture_moves(self, game_state: GameState) -> List[Dict[str, Any]]:
        """
        Get valid capture moves (marker removal during line formation)
        
        Args:
            game_state: Current game state
            
        Returns:
            List of valid capture moves
        """
        # TODO: Implement capture move logic
        # For now, return empty list
        return []
    
    def _generate_all_positions(self, board_type, size: int) -> List[Position]:
        """
        Generate all valid positions for a board type
        
        Args:
            board_type: Type of board
            size: Board size
            
        Returns:
            List of all valid positions
        """
        positions = []
        
        if board_type.value == "square8":
            for x in range(8):
                for y in range(8):
                    positions.append(Position(x=x, y=y))
        elif board_type.value == "square19":
            for x in range(19):
                for y in range(19):
                    positions.append(Position(x=x, y=y))
        elif board_type.value == "hexagonal":
            # Hexagonal board with cube coordinates
            radius = size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if abs(x) <= radius and abs(y) <= radius and abs(z) <= radius:
                        positions.append(Position(x=x, y=y, z=z))
        
        return positions
    
    def _get_adjacent_positions(self, position: Position, game_state: GameState) -> List[Position]:
        """
        Get adjacent positions to a given position
        
        Args:
            position: Current position
            game_state: Current game state
            
        Returns:
            List of adjacent positions
        """
        adjacent = []
        
        if game_state.board.type.value in ["square8", "square19"]:
            # Moore neighborhood (8 adjacent for square boards)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = Position(x=position.x + dx, y=position.y + dy)
                    if self._is_valid_position(new_pos, game_state):
                        adjacent.append(new_pos)
        
        elif game_state.board.type.value == "hexagonal":
            # Hexagonal adjacency (6 directions)
            hex_directions = [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]
            for dx, dy, dz in hex_directions:
                new_pos = Position(
                    x=position.x + dx,
                    y=position.y + dy,
                    z=position.z + dz if position.z is not None else None
                )
                if self._is_valid_position(new_pos, game_state):
                    adjacent.append(new_pos)
        
        return adjacent
    
    def _is_valid_position(self, position: Position, game_state: GameState) -> bool:
        """
        Check if a position is valid on the board
        
        Args:
            position: Position to check
            game_state: Current game state
            
        Returns:
            True if position is valid
        """
        board_type = game_state.board.type
        size = game_state.board.size
        
        if board_type.value == "square8":
            return 0 <= position.x < 8 and 0 <= position.y < 8
        elif board_type.value == "square19":
            return 0 <= position.x < 19 and 0 <= position.y < 19
        elif board_type.value == "hexagonal":
            if position.z is None:
                return False
            radius = size - 1
            return (abs(position.x) <= radius and 
                    abs(position.y) <= radius and 
                    abs(position.z) <= radius and
                    position.x + position.y + position.z == 0)
        
        return False
    
    def _create_move_object(self, move_dict: Dict[str, Any], game_state: GameState) -> Move:
        """
        Create a Move object from move dictionary
        
        Args:
            move_dict: Dictionary containing move information
            game_state: Current game state
            
        Returns:
            Move object
        """
        return Move(
            id=str(uuid.uuid4()),
            type=move_dict["type"],
            player=self.player_number,
            **{"from": move_dict.get("from")},
            to=move_dict["to"],
            timestamp=datetime.now(),
            thinkTime=random.randint(200, 800),
            moveNumber=len(game_state.move_history) + 1
        )

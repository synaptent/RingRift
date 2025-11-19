"""
Game Engine for RingRift AI Service
Provides move generation and state simulation logic
"""

from typing import List
import copy
from .models import GameState, Move, Position, BoardType, GamePhase, RingStack


class GameEngine:
    """
    Game engine implementation for AI service
    Provides valid move generation and state transition logic
    """
    
    @staticmethod
    def get_valid_moves(game_state: GameState, player_number: int) -> List[Move]:
        """
        Get all valid moves for a player in the current game state
        """
        # Only generate moves if it's the player's turn
        if game_state.current_player != player_number:
            return []

        phase = game_state.current_phase

        if phase == GamePhase.RING_PLACEMENT:
            return GameEngine._get_ring_placement_moves(game_state, player_number)
        elif phase == GamePhase.MOVEMENT:
            return GameEngine._get_movement_moves(game_state, player_number)
        elif phase == GamePhase.CAPTURE:
            # Capture moves are handled by the frontend/backend interaction usually
            # But for AI simulation we might need them.
            # For now, we'll focus on placement and movement as primary decision points.
            return []

        return []

    @staticmethod
    def apply_move(game_state: GameState, move: Move) -> GameState:
        """
        Apply a move to a game state and return the new state
        This is a simplified simulation for AI lookahead
        """
        # Create a deep copy of the game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)

        if move.type == "place_ring":
            GameEngine._apply_place_ring(new_state, move)
        elif move.type == "move_stack":
            GameEngine._apply_move_stack(new_state, move)
        elif move.type == "overtaking_capture":
            GameEngine._apply_overtaking_capture(new_state, move)
        elif move.type == "chain_capture":
            GameEngine._apply_chain_capture(new_state, move)
            
        # Update turn/phase logic would go here
        # For now, we just update the board state
        
        return new_state

    @staticmethod
    def _get_ring_placement_moves(game_state: GameState, player_number: int) -> List[Move]:
        """Get valid ring placement moves"""
        moves = []
        board = game_state.board

        # Check if player has rings in hand
        player = next((p for p in game_state.players if p.player_number == player_number), None)
        if not player or player.rings_in_hand <= 0:
            return []

        # Get all valid positions
        all_positions = GameEngine._generate_all_positions(board.type, board.size)

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces
            if pos_key in board.collapsed_spaces:
                continue

            # Can place on empty space
            if pos_key not in board.stacks:
                moves.append(Move(
                    id="simulated",
                    type="place_ring",
                    player=player_number,
                    to=pos,
                    timestamp=game_state.last_move_at,  # Placeholder
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                    placementCount=1,  # Default to 1 for simulation
                    placedOnStack=False
                ))
            else:
                # Can place on existing stack (stacking)
                # But only if it's not a marker? (Rules check needed)
                # In RingRift, you can place on existing stacks to build them up
                moves.append(Move(
                    id="simulated",
                    type="place_ring",
                    player=player_number,
                    to=pos,
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                    placementCount=1,
                    placedOnStack=True
                ))

        return moves

    @staticmethod
    def _get_movement_moves(game_state: GameState, player_number: int) -> List[Move]:
        """Get valid movement moves"""
        moves = []
        board = game_state.board

        # Iterate through all stacks controlled by the player
        for pos_key, stack in board.stacks.items():
            if stack.controlling_player == player_number:
                from_pos = stack.position

                # Get adjacent positions
                adjacent = GameEngine._get_adjacent_positions(from_pos, board.type, board.size)

                for to_pos in adjacent:
                    to_key = to_pos.to_key()

                    # Cannot move to collapsed space
                    if to_key in board.collapsed_spaces:
                        continue

                    # Logic for different move types based on target cell content
                    if to_key not in board.stacks:
                        # Move to empty space -> Move Stack
                        moves.append(Move(
                            id="simulated",
                            type="move_stack",
                            player=player_number,
                            from_pos=from_pos,
                            to=to_pos,
                            timestamp=game_state.last_move_at,
                            thinkTime=0,
                            moveNumber=len(game_state.move_history) + 1
                        ))
                    else:
                        target_stack = board.stacks[to_key]
                        # Move to occupied space
                        # If friendly stack -> Merge (not implemented in basic rules yet?) or invalid?
                        # If enemy stack -> Overtaking Capture check
                        if target_stack.controlling_player != player_number:
                            if stack.stack_height > target_stack.stack_height:
                                moves.append(Move(
                                    id="simulated",
                                    type="overtaking_capture",
                                    player=player_number,
                                    from_pos=from_pos,
                                    to=to_pos,
                                    timestamp=game_state.last_move_at,
                                    thinkTime=0,
                                    moveNumber=len(game_state.move_history) + 1
                                ))
        return moves

    @staticmethod
    def _apply_place_ring(game_state: GameState, move: Move):
        """Apply place ring move"""
        pos_key = move.to.to_key()
        board = game_state.board
        
        if pos_key not in board.stacks:
            # Create new stack
            board.stacks[pos_key] = RingStack(
                position=move.to,
                rings=[move.player],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=move.player
            )
        else:
            # Add to existing stack
            stack = board.stacks[pos_key]
            stack.rings.append(move.player)
            stack.stack_height += 1
            stack.cap_height += 1
            # Controlling player might change if we implemented full logic, 
            # but for placement it usually remains or becomes the placer if it was neutral?
            # In RingRift, placement usually adds to your own or neutral.
            # Simplified: Update controlling player to top ring
            stack.controlling_player = move.player
            
        # Decrement rings in hand
        for p in game_state.players:
            if p.player_number == move.player:
                p.rings_in_hand -= 1
                break

    @staticmethod
    def _apply_move_stack(game_state: GameState, move: Move):
        """Apply move stack move"""
        if not move.from_pos:
            return
            
        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()
        board = game_state.board
        
        if from_key in board.stacks:
            stack = board.stacks.pop(from_key)
            stack.position = move.to
            board.stacks[to_key] = stack

    @staticmethod
    def _apply_overtaking_capture(game_state: GameState, move: Move):
        """Apply overtaking capture move"""
        if not move.from_pos:
            return

        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()
        board = game_state.board

        if from_key in board.stacks and to_key in board.stacks:
            attacker = board.stacks.pop(from_key)
            defender = board.stacks.pop(to_key)
            
            # Simplified capture logic: Attacker replaces defender
            # In full rules, rings are combined/eliminated.
            # For AI simulation, we'll assume attacker takes the spot and grows?
            # Or just takes the spot.
            attacker.position = move.to
            board.stacks[to_key] = attacker
            
            # Update eliminated rings for defender?
            # This is complex to simulate perfectly without full rule engine.
            pass

    @staticmethod
    def _apply_chain_capture(game_state: GameState, move: Move):
        """Apply chain capture move"""
        # Chain captures are complex sequences of moves.
        # For AI simulation, we need to simulate the end result of the chain.
        # This requires knowing the full sequence of captures.
        # If the move object contains the sequence, we can iterate through it.
        # Otherwise, we might need to perform a search for the best chain.
        
        # Assuming move contains a 'chain_sequence' field or similar if it's a complex move
        # For now, we'll treat it as a single overtaking capture at the final destination
        # if we don't have the full sequence data.
        
        # TODO: Implement full chain capture simulation
        if not move.from_pos:
            return

        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()
        board = game_state.board

        if from_key in board.stacks and to_key in board.stacks:
            attacker = board.stacks.pop(from_key)
            # Remove defender stack
            board.stacks.pop(to_key)
            
            attacker.position = move.to
            board.stacks[to_key] = attacker

    @staticmethod
    def _generate_all_positions(board_type: BoardType, size: int) -> List[Position]:
        """Generate all valid positions"""
        positions = []
        if board_type == BoardType.SQUARE8:
            for x in range(8):
                for y in range(8):
                    positions.append(Position(x=x, y=y))
        elif board_type == BoardType.SQUARE19:
            for x in range(19):
                for y in range(19):
                    positions.append(Position(x=x, y=y))
        elif board_type == BoardType.HEXAGONAL:
            radius = size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if abs(x) <= radius and abs(y) <= radius and abs(z) <= radius:
                        positions.append(Position(x=x, y=y, z=z))
        return positions

    @staticmethod
    def _get_adjacent_positions(pos: Position, board_type: BoardType, size: int) -> List[Position]:
        """Get adjacent positions"""
        adjacent = []

        if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
            # Moore neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = pos.x + dx, pos.y + dy
                    # Bounds check
                    limit = 8 if board_type == BoardType.SQUARE8 else 19
                    if 0 <= new_x < limit and 0 <= new_y < limit:
                        adjacent.append(Position(x=new_x, y=new_y))

        elif board_type == BoardType.HEXAGONAL:
            hex_directions = [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]
            radius = size - 1
            for dx, dy, dz in hex_directions:
                if pos.z is None:
                    continue
                nx, ny, nz = pos.x + dx, pos.y + dy, pos.z + dz
                if abs(nx) <= radius and abs(ny) <= radius and abs(nz) <= radius:
                    adjacent.append(Position(x=nx, y=ny, z=nz))

        return adjacent

    @staticmethod
    def get_visible_stacks(pos: Position, game_state: GameState) -> List[RingStack]:
        """
        Get all stacks visible from a given position (line of sight).
        This is used for determining capture/overtake potential.
        """
        visible_stacks = []
        board = game_state.board
        board_type = board.type
        size = board.size

        directions = []
        if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
            # 8 directions for square board
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    directions.append((dx, dy, 0))
        elif board_type == BoardType.HEXAGONAL:
            # 6 directions for hexagonal board
            directions = [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]

        limit = 8 if board_type == BoardType.SQUARE8 else 19
        radius = size - 1

        for dx, dy, dz in directions:
            curr_x, curr_y = pos.x, pos.y
            curr_z = pos.z if pos.z is not None else 0
            
            # Raycast in this direction
            while True:
                curr_x += dx
                curr_y += dy
                curr_z += dz
                
                # Check bounds
                if board_type in [BoardType.SQUARE8, BoardType.SQUARE19]:
                    if not (0 <= curr_x < limit and 0 <= curr_y < limit):
                        break
                elif board_type == BoardType.HEXAGONAL:
                    if not (abs(curr_x) <= radius and abs(curr_y) <= radius and abs(curr_z) <= radius):
                        break
                
                curr_pos_key = f"{curr_x},{curr_y}"
                if board_type == BoardType.HEXAGONAL:
                    curr_pos_key += f",{curr_z}"
                
                # Check for stack
                if curr_pos_key in board.stacks:
                    visible_stacks.append(board.stacks[curr_pos_key])
                    break # Line of sight blocked by first stack
                
                # Check for marker (markers don't block line of sight in RingRift?
                # Actually they might, but for stack interactions usually we care about stacks.
                # Assuming markers don't block stack visibility for now, or if they do, add check here.)
                
        return visible_stacks
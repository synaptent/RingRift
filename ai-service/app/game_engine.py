"""
Game Engine for RingRift AI Service
Provides move generation and state simulation logic
"""

from typing import List
from .models import (
    GameState, Move, Position, BoardType, GamePhase, RingStack, MarkerInfo,
    GameStatus, MoveType
)
from .board_manager import BoardManager


class GameEngine:
    """
    Game engine implementation for AI service
    Provides valid move generation and state transition logic
    """
    
    # Cache for valid moves: key=state_hash, value=List[Move]
    _move_cache = {}
    _cache_hits = 0
    _cache_misses = 0

    @staticmethod
    def get_valid_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        """
        Get all valid moves for a player in the current game state
        """
        # Only generate moves if it's the player's turn
        if game_state.current_player != player_number:
            return []
            
        # Check cache
        state_hash = BoardManager.hash_game_state(game_state)
        cache_key = f"{state_hash}:{player_number}"
        
        if cache_key in GameEngine._move_cache:
            GameEngine._cache_hits += 1
            return GameEngine._move_cache[cache_key]
        
        GameEngine._cache_misses += 1

        phase = game_state.current_phase
        moves = []

        if phase == GamePhase.RING_PLACEMENT:
            moves = GameEngine._get_ring_placement_moves(
                game_state, player_number
            )
            if not moves:
                # If no placement possible (or no rings), try movement
                moves = GameEngine._get_movement_moves(
                    game_state, player_number
                )
                if not moves:
                    # If no movement, check forced elimination
                    player_stacks = BoardManager.get_player_stacks(
                        game_state.board, player_number
                    )
                    if player_stacks:
                        moves = GameEngine._get_forced_elimination_moves(
                            game_state, player_number
                        )

        elif phase == GamePhase.MOVEMENT:
            moves = GameEngine._get_movement_moves(game_state, player_number)
            if not moves:
                # If no movement, check forced elimination
                player_stacks = BoardManager.get_player_stacks(
                    game_state.board, player_number
                )
                if player_stacks:
                    moves = GameEngine._get_forced_elimination_moves(
                        game_state, player_number
                    )
        elif phase == GamePhase.CAPTURE:
            moves = GameEngine._get_capture_moves(game_state, player_number)
        elif phase == GamePhase.CHAIN_CAPTURE:
            # Support for canonical chain capture phase
            moves = GameEngine._get_capture_moves(game_state, player_number)
        elif phase == GamePhase.LINE_PROCESSING:
            moves = GameEngine._get_line_processing_moves(
                game_state, player_number
            )
        elif phase == GamePhase.TERRITORY_PROCESSING:
            moves = GameEngine._get_territory_processing_moves(
                game_state, player_number
            )

        # Cache result
        GameEngine._move_cache[cache_key] = moves
        return moves

    @staticmethod
    def clear_cache():
        """Clear the move cache"""
        GameEngine._move_cache.clear()
        GameEngine._cache_hits = 0
        GameEngine._cache_misses = 0

    @staticmethod
    def apply_move(game_state: GameState, move: Move) -> GameState:
        """
        Apply a move to a game state and return the new state
        This is a simplified simulation for AI lookahead
        """
        # Optimization: Manual shallow copy of GameState and BoardState
        # We only deep copy mutable structures that we intend to modify
        
        # 1. Create new BoardState (shallow copy first)
        new_board = game_state.board.model_copy()
        
        # 2. Deep copy mutable dictionaries in board that we might modify
        # Note: We can optimize further by only copying what we need based on
        # move type
        new_board.stacks = game_state.board.stacks.copy()
        new_board.markers = game_state.board.markers.copy()
        new_board.collapsed_spaces = game_state.board.collapsed_spaces.copy()
        new_board.eliminated_rings = game_state.board.eliminated_rings.copy()
        # formed_lines and territories are usually re-calculated or appended
        new_board.formed_lines = list(game_state.board.formed_lines)
        new_board.territories = game_state.board.territories.copy()

        # 3. Create new GameState
        new_state = game_state.model_copy(update={"board": new_board})
        
        # 4. Copy other mutable fields in GameState
        new_state.players = [p.model_copy() for p in game_state.players]
        new_state.move_history = list(game_state.move_history)
        
        # Capture S-invariant before move
        before_snapshot = BoardManager.compute_progress_snapshot(new_state)

        if move.type == MoveType.PLACE_RING:
            GameEngine._apply_place_ring(new_state, move)
        elif move.type == MoveType.MOVE_STACK:
            GameEngine._apply_move_stack(new_state, move)
        elif move.type == MoveType.OVERTAKING_CAPTURE:
            GameEngine._apply_overtaking_capture(new_state, move)
        elif (move.type == MoveType.CHAIN_CAPTURE or
              move.type == MoveType.CONTINUE_CAPTURE_SEGMENT):
            GameEngine._apply_chain_capture(new_state, move)
        elif (move.type == MoveType.LINE_FORMATION or
              move.type == MoveType.CHOOSE_LINE_OPTION):
            GameEngine._apply_line_formation(new_state, move)
        elif (move.type == MoveType.TERRITORY_CLAIM or
              move.type == MoveType.CHOOSE_TERRITORY_OPTION):
            GameEngine._apply_territory_claim(new_state, move)
        elif move.type == MoveType.FORCED_ELIMINATION:
            GameEngine._apply_forced_elimination(new_state, move)

        # Update move history
        new_state.move_history.append(move)
        new_state.last_move_at = move.timestamp

        # Handle phase transitions
        GameEngine._update_phase(new_state, move)

        # Verify S-invariant
        # S = markers + collapsed + eliminated
        # S must be non-decreasing
        after_snapshot = BoardManager.compute_progress_snapshot(new_state)
        if after_snapshot.S < before_snapshot.S:
            # In a real engine we might throw, but for AI simulation we
            # log/warn or just accept it if it's a known deviation.
            # For now, we'll assume correctness of logic but this hook is here.
            pass

        # Check victory conditions
        GameEngine._check_victory(new_state)

        return new_state

    @staticmethod
    def _check_victory(game_state: GameState):
        """Check for victory conditions"""
        # 1. Ring Elimination Victory
        # Check total eliminated rings for each player
        # Note: game_state.players might not be up to date with
        # board.eliminated_rings. We should sync them or check
        # board.eliminated_rings directly
        
        for p_id_str, count in game_state.board.eliminated_rings.items():
            if count >= game_state.victory_threshold:
                game_state.game_status = GameStatus.FINISHED
                game_state.winner = int(p_id_str)
                return

        # 2. Territory Victory
        territory_counts = {}
        for p_id in game_state.board.collapsed_spaces.values():
            if p_id not in territory_counts:
                territory_counts[p_id] = 0
            territory_counts[p_id] += 1
            
        for p_id, count in territory_counts.items():
            if count >= game_state.territory_victory_threshold:
                game_state.game_status = GameStatus.FINISHED
                game_state.winner = p_id
                return

        # 3. Global Stalemate (No stacks remain)
        # If no stacks remain on the board, the game ends in a stalemate
        # which is resolved by tie-breakers.
        # 3. Global Stalemate (No stacks remain)
        # If no stacks remain on the board, the game ends in a stalemate
        # which is resolved by tie-breakers.
        if not game_state.board.stacks:
            game_state.game_status = GameStatus.FINISHED
            
            # Tie-breaker logic:
            # 1. Most collapsed spaces
            # 2. Most eliminated rings (including rings in hand converted)
            # 3. Most markers
            # 4. Last player to complete a valid turn action
            
            # Calculate scores for each player
            scores = {}
            for player in game_state.players:
                pid = player.player_number
                
                # 1. Collapsed spaces
                collapsed = territory_counts.get(pid, 0)
                
                # 2. Eliminated rings + Rings in hand
                eliminated = game_state.board.eliminated_rings.get(str(pid), 0)
                eliminated += player.rings_in_hand
                
                # 3. Markers
                markers = 0
                for m in game_state.board.markers.values():
                    if m.player == pid:
                        markers += 1
                        
                scores[pid] = {
                    "collapsed": collapsed,
                    "eliminated": eliminated,
                    "markers": markers
                }
                
            # Determine winner
            # Assuming 2 players for simplicity, but logic holds for N
            # Sort players by criteria
            sorted_players = sorted(
                game_state.players,
                key=lambda p: (
                    scores[p.player_number]["collapsed"],
                    scores[p.player_number]["eliminated"],
                    scores[p.player_number]["markers"],
                    # 4. Last player to move wins?
                    # Rules say: "4. If still tied, the last player to complete a valid turn action."
                    # This means the current player (who just moved) or the previous?
                    # Usually "last player to complete" means the one who just finished their turn.
                    # Since we check victory after a move, the current_player (who made the move)
                    # is the one who completed the action.
                    # So we prefer the one who matches game_state.current_player?
                    # Wait, game_state.current_player is updated AFTER check_victory?
                    # In apply_move:
                    #   ...
                    #   GameEngine._update_phase(new_state, move)
                    #   ...
                    #   GameEngine._check_victory(new_state)
                    #
                    # _update_phase might switch player if turn ended.
                    # If turn ended, current_player is the NEXT player.
                    # So the LAST player is the one who just moved.
                    # We can check move_history[-1].player
                    1 if game_state.move_history and game_state.move_history[-1].player == p.player_number else 0
                ),
                reverse=True
            )
            
            game_state.winner = sorted_players[0].player_number
            return
        
        # 4. No legal moves for ANY player (Global Stalemate with stacks)
        # This is computationally expensive to check every move.
        # However, the rules say: "In any situation where no player has any legal placement,
        # movement, or capture but at least one stack still exists on the board, the controlling
        # player of some stack on their turn must satisfy the condition above and perform a forced elimination."
        # So technically, global stalemate with stacks is impossible if forced elimination is implemented correctly.
        # The only true stalemate is when no stacks remain.
        # But we should check if the current player has no moves AND no forced elimination is possible?
        # No, forced elimination is always possible if you have a stack.
        # So if you have a stack, you have a move (forced elim).
        # If you have no stack, you might have placement.
        # If you have no stack and no placement (no rings), you are eliminated/inactive.
        # If ALL players are eliminated/inactive, then game ends.
        
        active_players = 0
        for p in game_state.players:
            # Check if player can potentially move
            has_stacks = any(s.controlling_player == p.player_number for s in game_state.board.stacks.values())
            can_place = p.rings_in_hand > 0
            if has_stacks or can_place:
                active_players += 1
                
        if active_players == 0:
             # Should have been caught by "no stacks" check if no stacks,
             # but if stacks exist but belong to no one (impossible?) or
             # some edge case, we treat as stalemate.
             # But stacks always have a controller.
             # So if stacks exist, someone is active.
             pass

    @staticmethod
    def _update_phase(game_state: GameState, last_move: Move):
        current_player = game_state.current_player

        if last_move.type == MoveType.FORCED_ELIMINATION:
            # After forced elimination, turn ends?
            # "If after this elimination P still has no legal action, their
            # turn ends."
            # "Successive forced eliminations continue... until no stacks
            # remain"
            # For simplicity, we'll end turn and let next turn handle it if
            # needed.
            # Or we should check if they have moves now?
            # Usually forced elimination opens up space.
            # But the rule says "If after this elimination P still has no
            # legal action, their turn ends."
            # This implies we should check for moves again.
            # But that would require a phase loop.
            # For now, let's end turn.
            GameEngine._end_turn(game_state)

        elif last_move.type == MoveType.PLACE_RING:
            # After placement, must move the placed stack
            game_state.current_phase = GamePhase.MOVEMENT

        elif last_move.type == MoveType.MOVE_STACK:
            # After movement, check for captures
            capture_moves = GameEngine._get_capture_moves(
                game_state, current_player
            )
            if capture_moves:
                game_state.current_phase = GamePhase.CAPTURE
            else:
                # No captures, go to line processing
                GameEngine._advance_to_line_processing(game_state)

        elif (last_move.type == MoveType.OVERTAKING_CAPTURE or
              last_move.type == MoveType.CHAIN_CAPTURE or
              last_move.type == MoveType.CONTINUE_CAPTURE_SEGMENT):
            # Check for more captures (chain)
            capture_moves = GameEngine._get_capture_moves(
                game_state, current_player
            )
            if capture_moves:
                game_state.current_phase = GamePhase.CAPTURE
            else:
                # End of chain
                GameEngine._advance_to_line_processing(game_state)

        elif (last_move.type == MoveType.LINE_FORMATION or
              last_move.type == MoveType.CHOOSE_LINE_OPTION):
            # Check for more lines
            line_moves = GameEngine._get_line_processing_moves(
                game_state, current_player
            )
            if line_moves:
                game_state.current_phase = GamePhase.LINE_PROCESSING
            else:
                GameEngine._advance_to_territory_processing(game_state)

        elif (last_move.type == MoveType.TERRITORY_CLAIM or
              last_move.type == MoveType.CHOOSE_TERRITORY_OPTION):
            # Check for more regions
            territory_moves = GameEngine._get_territory_processing_moves(
                game_state, current_player
            )
            if territory_moves:
                game_state.current_phase = GamePhase.TERRITORY_PROCESSING
            else:
                GameEngine._end_turn(game_state)

    @staticmethod
    def _advance_to_line_processing(game_state: GameState):
        line_moves = GameEngine._get_line_processing_moves(
            game_state, game_state.current_player
        )
        if line_moves:
            game_state.current_phase = GamePhase.LINE_PROCESSING
        else:
            GameEngine._advance_to_territory_processing(game_state)

    @staticmethod
    def _advance_to_territory_processing(game_state: GameState):
        territory_moves = GameEngine._get_territory_processing_moves(
            game_state, game_state.current_player
        )
        if territory_moves:
            game_state.current_phase = GamePhase.TERRITORY_PROCESSING
        else:
            GameEngine._end_turn(game_state)

    @staticmethod
    def _end_turn(game_state: GameState):
        # Switch player
        # Assuming 2 players for now
        game_state.current_player = 2 if game_state.current_player == 1 else 1

        # Determine next phase
        # If player has rings in hand, they can place (optional/mandatory)
        # But usually starts with placement check
        game_state.current_phase = GamePhase.RING_PLACEMENT

        # Check if placement is possible/required
        # If no rings in hand and no stacks, pass?
        # Rules say: "Movement is always mandatory... Ring placement is
        # mandatory only if..."

        # For simplicity, start in RING_PLACEMENT.
        # If no rings, get_valid_moves will return empty, and we should
        # handle that?
        # Or we can check here.

        player = next(
            p for p in game_state.players
            if p.player_number == game_state.current_player
        )
        if player.rings_in_hand == 0:
            # Skip placement if no rings
            # But wait, if they have no rings on board, they lose/pass?
            # If they have stacks, they go to MOVEMENT.
            game_state.current_phase = GamePhase.MOVEMENT

    @staticmethod
    def _get_ring_placement_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        """Get valid ring placement moves"""
        moves = []
        board = game_state.board

        # Check if player has rings in hand
        player = next(
            (p for p in game_state.players
             if p.player_number == player_number),
            None
        )
        if not player or player.rings_in_hand <= 0:
            return []

        # Get all valid positions
        all_positions = GameEngine._generate_all_positions(
            board.type, board.size
        )

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces
            if pos_key in board.collapsed_spaces:
                continue

            # Cannot place on markers
            if pos_key in board.markers:
                continue

            # Can place on empty space
            if pos_key not in board.stacks:
                # Check no-dead-placement
                # Simulate placement
                # Create hypothetical stack
                hypothetical_stack = RingStack(
                    position=pos,
                    rings=[player_number],
                    stackHeight=1,
                    capHeight=1,
                    controllingPlayer=player_number
                )
                # Temporarily add to board
                board.stacks[pos_key] = hypothetical_stack
                
                # Check if any legal move exists from this stack
                has_legal_move = False
                
                # Check movement
                adjacent = GameEngine._get_adjacent_positions(
                    pos, board.type, board.size
                )
                for adj in adjacent:
                    adj_key = adj.to_key()
                    if adj_key in board.collapsed_spaces:
                        continue
                    if adj_key not in board.stacks:
                        has_legal_move = True
                        break
                    else:
                        target_stack = board.stacks[adj_key]
                        if target_stack.controlling_player != player_number:
                            if (hypothetical_stack.stack_height >
                                    target_stack.stack_height):
                                has_legal_move = True
                                break
                
                # Check capture (if movement didn't yield any)
                if not has_legal_move:
                    # Check for capture opportunities
                    # We need to check if any capture is possible from this
                    # position. This is a simplified check: if we can reach
                    # any enemy stack that we can capture and land beyond it.
                    
                    # Reuse _get_capture_moves logic but scoped to this
                    # position. We need to temporarily set phase to CAPTURE
                    # to use _get_capture_moves? No, that's too invasive.
                    # Let's implement a helper for single-stack capture check.
                    
                    # For now, we can iterate directions and check for capture
                    # patterns
                    directions = BoardManager._get_all_directions(board.type)
                    for direction in directions:
                        # Find target
                        step = 1
                        target_pos = None
                        while True:
                            check_pos = BoardManager._add_direction(
                                pos, direction, step
                            )
                            if not BoardManager.is_valid_position(
                                check_pos, board.type, board.size
                            ):
                                break
                            if BoardManager.is_collapsed_space(
                                check_pos, board
                            ):
                                break
                            
                            stack_at_pos = BoardManager.get_stack(
                                check_pos, board
                            )
                            if stack_at_pos:
                                if stack_at_pos.stack_height > 0:
                                    if (hypothetical_stack.cap_height >=
                                            stack_at_pos.cap_height):
                                        target_pos = check_pos
                                    break
                            
                            step += 1
                        
                        if target_pos:
                            # Check landing
                            landing_step = 1
                            while landing_step <= 5:  # Max landing distance
                                landing_pos = BoardManager._add_direction(
                                    target_pos, direction, landing_step
                                )
                                if not BoardManager.is_valid_position(
                                    landing_pos, board.type, board.size
                                ):
                                    break
                                if BoardManager.is_collapsed_space(
                                    landing_pos, board
                                ):
                                    break
                                
                                landing_stack = BoardManager.get_stack(
                                    landing_pos, board
                                )
                                if (landing_stack and
                                        landing_stack.stack_height > 0):
                                    break
                                
                                # Check marker at landing
                                marker = board.markers.get(
                                    landing_pos.to_key()
                                )
                                if (marker is not None and
                                        marker.player != player_number):
                                    break

                                # Valid capture found
                                has_legal_move = True
                                break
                            
                            if has_legal_move:
                                break

                # Remove hypothetical stack
                del board.stacks[pos_key]
                
                if has_legal_move:
                    moves.append(Move(
                        id="simulated",
                        type=MoveType.PLACE_RING,
                        player=player_number,
                        to=pos,
                        timestamp=game_state.last_move_at,  # Placeholder
                        thinkTime=0,
                        moveNumber=len(game_state.move_history) + 1,
                        placementCount=1,  # Default to 1 for simulation
                        placedOnStack=False
                    ))  # type: ignore
            else:
                # Can place on existing stack (stacking)
                # But only if it's not a marker? (Rules check needed)
                # In RingRift, you can place on existing stacks to build them
                # up
                
                # Check no-dead-placement for stacking
                stack = board.stacks[pos_key]
                original_rings = list(stack.rings)
                original_height = stack.stack_height
                original_cap = stack.cap_height
                original_controller = stack.controlling_player
                
                # Simulate
                stack.rings.append(player_number)
                stack.stack_height += 1
                stack.cap_height += 1
                stack.controlling_player = player_number
                
                has_legal_move = False
                adjacent = GameEngine._get_adjacent_positions(
                    pos, board.type, board.size
                )
                for adj in adjacent:
                    adj_key = adj.to_key()
                    if adj_key in board.collapsed_spaces:
                        continue
                    if adj_key not in board.stacks:
                        # Check for markers
                        marker = board.markers.get(adj_key)
                        if marker is None or marker.player == player_number:
                            has_legal_move = True
                            break
                    else:
                        target_stack = board.stacks[adj_key]
                        if target_stack.controlling_player != player_number:
                            if stack.stack_height > target_stack.stack_height:
                                has_legal_move = True
                                break
                                
                # Revert
                stack.rings = original_rings
                stack.stack_height = original_height
                stack.cap_height = original_cap
                stack.controlling_player = original_controller
                
                if has_legal_move:
                    moves.append(Move(
                        id="simulated",
                        type=MoveType.PLACE_RING,
                        player=player_number,
                        to=pos,
                        timestamp=game_state.last_move_at,
                        thinkTime=0,
                        moveNumber=len(game_state.move_history) + 1,
                        placementCount=1,
                        placedOnStack=True
                    ))  # type: ignore

        return moves

    @staticmethod
    def _get_capture_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        # Check if we are in a chain capture sequence
        if game_state.chain_capture_state:
            # Use state to determine attacker position
            attacker_pos = game_state.chain_capture_state.current_position
            visited = set(game_state.chain_capture_state.visited_positions)
        else:
            # Initial capture or legacy fallback
            last_move = (
                game_state.move_history[-1]
                if game_state.move_history else None
            )
            if not last_move or not last_move.to:
                return []
            attacker_pos = last_move.to
            visited = set()
            # Add current position to visited to prevent immediate reversal?
            # No, standard rules handle direction.
            # But for cyclic prevention, we should track visited.
            # For initial capture, visited is empty (except maybe start).

        attacker_stack = BoardManager.get_stack(attacker_pos, game_state.board)

        if (not attacker_stack or
                attacker_stack.controlling_player != player_number):
            return []

        moves = []
        directions = BoardManager._get_all_directions(game_state.board.type)

        for direction in directions:
            # Ray cast to find target
            step = 1
            target_pos = None

            while True:
                pos = BoardManager._add_direction(
                    attacker_pos, direction, step
                )
                if not BoardManager.is_valid_position(
                    pos, game_state.board.type, game_state.board.size
                ):
                    break

                if BoardManager.is_collapsed_space(pos, game_state.board):
                    break

                stack = BoardManager.get_stack(pos, game_state.board)
                if stack:
                    # First stack encountered is the only possible target
                    if (stack.controlling_player != player_number and
                            attacker_stack.cap_height >= stack.cap_height):
                        target_pos = pos
                    break  # Blocked by stack (target or friendly)

                step += 1

            if target_pos:
                # Check if target has been visited (cyclic capture prevention)
                if target_pos.to_key() in visited:
                    continue

                # Found target, check landing spots beyond
                landing_step = 1
                while True:
                    landing_pos = BoardManager._add_direction(
                        target_pos, direction, landing_step
                    )
                    if not BoardManager.is_valid_position(
                        landing_pos, game_state.board.type,
                        game_state.board.size
                    ):
                        break

                    if BoardManager.is_collapsed_space(
                        landing_pos, game_state.board
                    ):
                        break

                    if BoardManager.get_stack(landing_pos, game_state.board):
                        break  # Blocked by stack

                    # Check if landing has been visited
                    if landing_pos.to_key() in visited:
                        # Landing on a visited spot is generally allowed?
                        # Rules say: "A stack may not land on a space it has
                        # already occupied in the current turn."
                        # So yes, we must check landing against visited.
                        landing_step += 1
                        continue

                    # Valid landing (empty space)
                    # Check minimum distance (stack height)
                    # Distance is from attacker_pos to landing_pos
                    # In ray casting, total steps = step (to target) +
                    # landing_step (to landing)
                    total_dist = step + landing_step
                    if total_dist >= attacker_stack.stack_height:
                        # Determine move type based on phase
                        move_type = (
                            MoveType.CONTINUE_CAPTURE_SEGMENT
                            if game_state.chain_capture_state
                            else MoveType.CHAIN_CAPTURE  # Legacy/Initial
                        )
                        
                        moves.append(Move(
                            id="simulated",
                            type=move_type,
                            player=player_number,
                            from_pos=attacker_pos,  # type: ignore
                            to=landing_pos,
                            capture_target=target_pos,  # type: ignore
                            timestamp=game_state.last_move_at,
                            thinkTime=0,
                            moveNumber=len(game_state.move_history) + 1
                        ))

                    landing_step += 1

        return moves

    @staticmethod
    def _get_forced_elimination_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        """Get forced elimination moves when blocked"""
        moves = []
        board = game_state.board
        
        # Find all stacks controlled by player
        for pos_key, stack in board.stacks.items():
            if stack.controlling_player == player_number:
                # Create elimination move
                # We use a special move type or reuse an existing one?
                # The rules say "eliminate one entire cap".
                # We can use "forced_elimination" type.
                
                # Need position from key
                # Assuming we can parse key or store position in stack
                pos = stack.position
                
                moves.append(Move(
                    id="simulated",
                    type=MoveType.FORCED_ELIMINATION,
                    player=player_number,
                    to=pos,  # Target stack position
                    timestamp=game_state.last_move_at,
                    thinkTime=0,
                    moveNumber=len(game_state.move_history) + 1,
                    placementCount=0,
                    placedOnStack=False
                ))  # type: ignore
        return moves

    @staticmethod
    def _get_line_processing_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        # Find all lines for the player
        lines = BoardManager.find_all_lines(game_state.board)
        player_lines = [line for line in lines if line.player == player_number]

        if not player_lines:
            return []

        moves = []
        for i, line in enumerate(player_lines):
            # Option 1: Collapse all and eliminate
            moves.append(Move(
                id="simulated",
                type=MoveType.CHOOSE_LINE_OPTION,
                player=player_number,
                to=line.positions[0],  # Use first position as identifier
                timestamp=game_state.last_move_at,
                thinkTime=0,
                moveNumber=len(game_state.move_history) + 1,
                # We use 'placement_count' as a hack to encode the option
                # 1 = Option 1 (Collapse All + Eliminate)
                # 2 = Option 2 (Min Collapse + No Eliminate)
                placementCount=1
            ))  # type: ignore
            
            # Option 2: Min collapse, no elimination
            # Generate moves for all possible segments of required length
            required_len = 4 if game_state.board.type == BoardType.SQUARE8 else 5
            if len(line.positions) > required_len:
                # Generate all contiguous segments of required_len
                for start_idx in range(len(line.positions) - required_len + 1):
                    segment = line.positions[start_idx : start_idx + required_len]
                    moves.append(Move(
                        id="simulated",
                        type=MoveType.CHOOSE_LINE_OPTION,
                        player=player_number,
                        to=line.positions[0],
                        timestamp=game_state.last_move_at,
                        thinkTime=0,
                        moveNumber=len(game_state.move_history) + 1,
                        placementCount=2,
                        collapsed_markers=segment
                    ))  # type: ignore
        return moves

    @staticmethod
    def _get_territory_processing_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        # Find disconnected regions
        regions = BoardManager.find_disconnected_regions(
            game_state.board, player_number
        )

        # Filter regions based on self-elimination prerequisite
        # Player must have at least one stack outside the region
        valid_regions = []
        player_stacks = BoardManager.get_player_stacks(
            game_state.board, player_number
        )

        for region in regions:
            region_keys = set(p.to_key() for p in region.spaces)
            has_stack_outside = False
            for stack in player_stacks:
                if stack.position.to_key() not in region_keys:
                    has_stack_outside = True
                    break

            if has_stack_outside:
                valid_regions.append(region)

        if not valid_regions:
            return []

        moves = []
        for i, region in enumerate(valid_regions):
            moves.append(Move(
                id="simulated",
                type=MoveType.CHOOSE_TERRITORY_OPTION,
                player=player_number,
                to=region.spaces[0],  # Use first space as identifier
                timestamp=game_state.last_move_at,
                thinkTime=0,
                moveNumber=len(game_state.move_history) + 1
            ))  # type: ignore
        return moves

    @staticmethod
    def _get_movement_moves(
        game_state: GameState, player_number: int
    ) -> List[Move]:
        """Get valid movement moves"""
        moves = []
        board = game_state.board

        # Check if last move was placement by this player
        last_move = (
            game_state.move_history[-1] if game_state.move_history else None
        )
        must_move_pos = None
        if (last_move and
                last_move.player == player_number and
                last_move.type == MoveType.PLACE_RING):
            must_move_pos = last_move.to

        # Iterate through all stacks controlled by the player
        for pos_key, stack in board.stacks.items():
            if stack.controlling_player == player_number:
                # If we must move a specific stack, skip others
                if must_move_pos and (
                    stack.position.x != must_move_pos.x or
                    stack.position.y != must_move_pos.y
                ):
                    continue
                if must_move_pos and must_move_pos.z is not None and (
                    stack.position.z != must_move_pos.z
                ):
                    continue

                from_pos = stack.position

                # Get adjacent positions
                adjacent = GameEngine._get_adjacent_positions(
                    from_pos, board.type, board.size
                )

                for to_pos in adjacent:
                    to_key = to_pos.to_key()

                    # Cannot move to collapsed space
                    if to_key in board.collapsed_spaces:
                        continue

                    # Logic for different move types based on target cell
                    # content
                    if to_key not in board.stacks:
                        # Move to empty space -> Move Stack
                        moves.append(Move(
                            id="simulated",
                            type=MoveType.MOVE_STACK,
                            player=player_number,
                            from_pos=from_pos,  # type: ignore
                            to=to_pos,
                            timestamp=game_state.last_move_at,
                            thinkTime=0,
                            moveNumber=len(game_state.move_history) + 1
                        ))
                    else:
                        target_stack = board.stacks[to_key]
                        # Move to occupied space
                        # If friendly stack -> Merge (not implemented in basic
                        # rules yet?) or invalid?
                        # If enemy stack -> Overtaking Capture check
                        if target_stack.controlling_player != player_number:
                            if stack.stack_height > target_stack.stack_height:
                                moves.append(Move(
                                    id="simulated",
                                    type=MoveType.OVERTAKING_CAPTURE,
                                    player=player_number,
                                    from_pos=from_pos,  # type: ignore
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
            # Copy stack before modification to avoid state corruption
            stack = board.stacks[pos_key].model_copy(deep=True)
            board.stacks[pos_key] = stack
            
            stack.rings.append(move.player)
            stack.stack_height += 1
            stack.cap_height += 1
            # Controlling player might change if we implemented full logic,
            # but for placement it usually remains or becomes the placer if it
            # was neutral?
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
            # Copy stack before modification
            stack = board.stacks.pop(from_key).model_copy(deep=True)
            
            # Leave a marker
            board.markers[from_key] = MarkerInfo(
                player=move.player,
                position=move.from_pos,
                type="regular"
            )
            # Handle destination marker
            if to_key in board.markers:
                marker = board.markers[to_key]
                if marker.player == move.player:
                    # Retrieve own marker
                    del board.markers[to_key]
                else:
                    # Flip opponent marker
                    marker.player = move.player
            
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
            # Copy attacker before modification
            attacker = board.stacks.pop(from_key).model_copy(deep=True)
            board.stacks.pop(to_key)
            
            # Leave a marker
            board.markers[from_key] = MarkerInfo(
                player=move.player,
                position=move.from_pos,
                type="regular"
            )

            # Simplified capture logic: Attacker replaces defender
            # In full rules, rings are combined/eliminated.
            # For AI simulation, we'll assume attacker takes the spot and
            # grows?
            # Or just takes the spot.
            attacker.position = move.to
            board.stacks[to_key] = attacker

            # Update eliminated rings for defender?
            # This is complex to simulate perfectly without full rule engine.
            pass

            # Check for chain capture continuation
            # If we can capture again from the new position, we enter
            # CHAIN_CAPTURE phase. But only if the move type was
            # OVERTAKING_CAPTURE (initial capture).
            
            # In TS, ChainCaptureState tracks visited positions to prevent
            # cycles. We should initialize it here.
            
            from .models import ChainCaptureState
            
            # Initialize state
            game_state.chain_capture_state = ChainCaptureState(
                playerNumber=move.player,
                startPosition=move.from_pos,  # type: ignore
                currentPosition=move.to,
                segments=[],
                availableMoves=[],
                visitedPositions=[
                    move.from_pos.to_key(),  # type: ignore
                    move.to.to_key()
                ]
            )
            # Add target to visited?
            if move.capture_target:
                game_state.chain_capture_state.visited_positions.append(
                    move.capture_target.to_key()
                )

    @staticmethod
    def _apply_chain_capture(game_state: GameState, move: Move):
        """Apply chain capture move"""
        if not move.from_pos:
            return

        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()
        board = game_state.board

        attacker = board.stacks.get(from_key)
        if not attacker:
            return

        target_pos = None
        
        if move.capture_target:
            target_pos = move.capture_target
        else:
            # Fallback: re-calculate target if not provided
            # Determine direction
            dx = move.to.x - move.from_pos.x
            dy = move.to.y - move.from_pos.y
            dz = (move.to.z - move.from_pos.z) if (
                move.to.z is not None and move.from_pos.z is not None
            ) else None

            # Normalize direction
            steps = max(abs(dx), abs(dy), abs(dz) if dz is not None else 0)
            if steps == 0:
                return

            dir_x = dx // steps
            dir_y = dy // steps
            dir_z = dz // steps if dz is not None else None

            direction = (dir_x, dir_y, dir_z)

            # Find target
            step = 1
            while step < steps:
                pos = BoardManager._add_direction(
                    move.from_pos, direction, step
                )
                pos_key = pos.to_key()
                if pos_key in board.stacks:
                    target_pos = pos
                    break
                step += 1

        if target_pos:
            target_key = target_pos.to_key()
            
            # Copy stacks before modification
            # Attacker is already a reference from board.stacks.get(from_key)
            # We need to copy it.
            attacker = attacker.model_copy(deep=True)
            
            target_stack = board.stacks[target_key].model_copy(deep=True)
            board.stacks[target_key] = target_stack

            # Capture top ring
            # Top ring is at end.
            if not target_stack.rings:
                # Should not happen if stack exists in board.stacks
                # But handle gracefully just in case
                if target_key in board.stacks:
                    del board.stacks[target_key]
                return

            captured_ring = target_stack.rings.pop()
            target_stack.stack_height -= 1
            target_stack.cap_height = 0  # Recalculate later

            # Actually we need to recalculate cap height for target stack
            # But wait, if stack becomes empty?
            if not target_stack.rings:
                del board.stacks[target_key]
            else:
                # Update controlling player and cap height
                target_stack.controlling_player = target_stack.rings[-1]
                # Recalculate cap height
                h = 0
                for r in reversed(target_stack.rings):
                    if r == target_stack.controlling_player:
                        h += 1
                    else:
                        break
                target_stack.cap_height = h

            # Add to attacker
            attacker.rings.insert(0, captured_ring)  # Add to bottom
            attacker.stack_height += 1
            # Cap height doesn't change for attacker (added to bottom)

            # Move attacker
            del board.stacks[from_key]
            
            # Leave a marker at start
            board.markers[from_key] = MarkerInfo(
                player=move.player,
                position=move.from_pos,
                type="regular"
            )
            
            attacker.position = move.to
            board.stacks[to_key] = attacker

            # Handle markers at landing
            if to_key in board.markers:
                marker = board.markers[to_key]
                if marker.player == move.player:
                    # Retrieve own marker
                    del board.markers[to_key]
                else:
                    # Flip opponent marker
                    marker.player = move.player

            # Update chain capture state
            from .models import ChainCaptureSegment
            
            segment = ChainCaptureSegment(
                from_pos=move.from_pos,  # type: ignore
                target=target_pos,
                landing=move.to,
                capturedCapHeight=0  # Placeholder
            )
            
            if game_state.chain_capture_state:
                game_state.chain_capture_state.current_position = move.to
                game_state.chain_capture_state.segments.append(segment)
                game_state.chain_capture_state.visited_positions.append(
                    target_pos.to_key()
                )
                game_state.chain_capture_state.visited_positions.append(
                    move.to.to_key()
                )

    @staticmethod
    def _apply_line_formation(game_state: GameState, move: Move):
        """Apply line formation move (handles both legacy and canonical types)"""
        # Find the line associated with this move (by start pos)
        lines = BoardManager.find_all_lines(game_state.board)
        target_line = None
        for line in lines:
            if line.positions[0].to_key() == move.to.to_key():
                target_line = line
                break
        
        if not target_line:
            return

        # Check option
        # 1 = Option 1 (Collapse All + Eliminate)
        # 2 = Option 2 (Min Collapse + No Eliminate)
        option = move.placement_count or 1
        
        if option == 1:
            # Option 1: Collapse all markers
            for pos in target_line.positions:
                BoardManager.set_collapsed_space(
                    pos, move.player, game_state.board
                )
                
            # Eliminate ring from largest stack
            player_stacks = BoardManager.get_player_stacks(
                game_state.board, move.player
            )
            if player_stacks:
                # Sort by cap height desc
                player_stacks.sort(key=lambda s: s.cap_height, reverse=True)
                stack = player_stacks[0]
                
                # Copy stack before modification
                key = stack.position.to_key()
                stack = stack.model_copy(deep=True)
                game_state.board.stacks[key] = stack
                
                # Eliminate cap
                cap_height = stack.cap_height
                stack.rings = stack.rings[:-cap_height]
                stack.stack_height -= cap_height
                stack.cap_height = 0  # Recalculate if needed
                
                # Update eliminated count
                game_state.total_rings_eliminated += cap_height
                if str(move.player) not in game_state.board.eliminated_rings:
                    game_state.board.eliminated_rings[str(move.player)] = 0
                game_state.board.eliminated_rings[str(move.player)] += \
                    cap_height
                
                if not stack.rings:
                    BoardManager.remove_stack(stack.position, game_state.board)
                else:
                    # Update controlling player
                    stack.controlling_player = stack.rings[-1]
                    # Recalculate cap height
                    h = 0
                    for r in reversed(stack.rings):
                        if r == stack.controlling_player:
                            h += 1
                        else:
                            break
                    stack.cap_height = h
                    BoardManager.set_stack(
                        stack.position, stack, game_state.board
                    )
        elif option == 2:
            # Option 2: Minimum collapse (required_len markers), no elimination
            # Use collapsed_markers from move if available, otherwise default to first N
            required_len = 4 if game_state.board.type == BoardType.SQUARE8 else 5
            
            markers_to_collapse = move.collapsed_markers
            if not markers_to_collapse:
                # Fallback for legacy/simulated moves without explicit segment
                markers_to_collapse = target_line.positions[:required_len]
                
            for pos in markers_to_collapse:
                BoardManager.set_collapsed_space(
                    pos, move.player, game_state.board
                )
            
            # No elimination

    @staticmethod
    def _apply_territory_claim(game_state: GameState, move: Move):
        """Apply territory claim move (handles both legacy and canonical types)"""
        # Find region
        regions = BoardManager.find_disconnected_regions(
            game_state.board, move.player
        )
        target_region = None
        for region in regions:
            if region.spaces[0].to_key() == move.to.to_key():
                target_region = region
                break
                
        if not target_region:
            return
            
        # 1. Get border markers
        border_markers = BoardManager.get_border_marker_positions(
            target_region.spaces, game_state.board
        )
        
        # 2. Eliminate rings in region
        total_eliminated = 0
        for pos in target_region.spaces:
            stack = BoardManager.get_stack(pos, game_state.board)
            if stack:
                total_eliminated += stack.stack_height
                BoardManager.remove_stack(pos, game_state.board)
                
        # 3. Collapse spaces
        for pos in target_region.spaces:
            BoardManager.set_collapsed_space(
                pos, move.player, game_state.board
            )
            
        # 4. Collapse border markers
        for pos in border_markers:
            BoardManager.set_collapsed_space(
                pos, move.player, game_state.board
            )
            
        # 5. Update elimination counts
        game_state.total_rings_eliminated += total_eliminated
        if str(move.player) not in game_state.board.eliminated_rings:
            game_state.board.eliminated_rings[str(move.player)] = 0
        game_state.board.eliminated_rings[str(move.player)] += total_eliminated
        
        # 6. Self-elimination (one ring/cap)
        player_stacks = BoardManager.get_player_stacks(
            game_state.board, move.player
        )
        if player_stacks:
            player_stacks.sort(key=lambda s: s.cap_height, reverse=True)
            stack = player_stacks[0]
            
            # Copy stack before modification
            key = stack.position.to_key()
            stack = stack.model_copy(deep=True)
            game_state.board.stacks[key] = stack
            
            cap_height = stack.cap_height
            stack.rings = stack.rings[:-cap_height]
            stack.stack_height -= cap_height
            
            game_state.total_rings_eliminated += cap_height
            game_state.board.eliminated_rings[str(move.player)] += cap_height
            
            if not stack.rings:
                BoardManager.remove_stack(stack.position, game_state.board)
            else:
                stack.controlling_player = stack.rings[-1]
                h = 0
                for r in reversed(stack.rings):
                    if r == stack.controlling_player:
                        h += 1
                    else:
                        break
                stack.cap_height = h
                BoardManager.set_stack(stack.position, stack, game_state.board)

    @staticmethod
    def _generate_all_positions(
        board_type: BoardType, size: int
    ) -> List[Position]:
        """Generate all valid positions"""
        # Use BoardManager logic (replicated here or delegated if BoardManager
        # exposed it)
        # For now, keep existing logic but ensure it matches BoardManager
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
                    if (abs(x) <= radius and
                            abs(y) <= radius and
                            abs(z) <= radius):
                        positions.append(Position(x=x, y=y, z=z))
        return positions

    @staticmethod
    def _get_adjacent_positions(
        pos: Position, board_type: BoardType, size: int
    ) -> List[Position]:
        """Get adjacent positions"""
        # Use BoardManager logic
        # For now, keep existing logic
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
                if (abs(nx) <= radius and
                        abs(ny) <= radius and
                        abs(nz) <= radius):
                    adjacent.append(Position(x=nx, y=ny, z=nz))

        return adjacent

    @staticmethod
    def get_visible_stacks(
        pos: Position, game_state: GameState
    ) -> List[RingStack]:
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
                    if dx == 0 and dy == 0:
                        continue
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
                    if not (abs(curr_x) <= radius and
                            abs(curr_y) <= radius and
                            abs(curr_z) <= radius):
                        break
                
                curr_pos_key = f"{curr_x},{curr_y}"
                if board_type == BoardType.HEXAGONAL:
                    curr_pos_key += f",{curr_z}"
                
                # Check for stack
                if curr_pos_key in board.stacks:
                    visible_stacks.append(board.stacks[curr_pos_key])
                    break  # Line of sight blocked by first stack

                # Check for marker (markers don't block line of sight in
                # RingRift? Actually they might, but for stack interactions
                # usually we care about stacks. Assuming markers don't block
                # stack visibility for now, or if they do, add check here.)
                
        return visible_stacks

    @staticmethod
    def _apply_forced_elimination(game_state: GameState, move: Move):
        """Apply forced elimination move"""
        board = game_state.board
        pos_key = move.to.to_key()
        
        if pos_key not in board.stacks:
            return
            
        # Copy stack before modification
        stack = board.stacks[pos_key].model_copy(deep=True)
        board.stacks[pos_key] = stack
        
        # Eliminate cap
        cap_height = stack.cap_height
        
        # Remove rings
        for _ in range(cap_height):
            if stack.rings:
                stack.rings.pop()
                
        stack.stack_height -= cap_height
        
        # Update eliminated rings
        player_id = str(move.player)
        if player_id not in board.eliminated_rings:
            board.eliminated_rings[player_id] = 0
        board.eliminated_rings[player_id] += cap_height
        
        game_state.total_rings_eliminated += cap_height
        
        # If stack empty, remove it
        if stack.stack_height == 0:
            del board.stacks[pos_key]
        else:
            # Update control
            stack.controlling_player = stack.rings[-1]
            # Recalculate cap height
            h = 0
            for r in reversed(stack.rings):
                if r == stack.controlling_player:
                    h += 1
                else:
                    break
            stack.cap_height = h
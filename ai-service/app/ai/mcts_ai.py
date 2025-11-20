"""
MCTS AI implementation for RingRift
Uses Monte Carlo Tree Search for move selection
"""

from typing import Optional, List, Dict, Any
import random
from datetime import datetime
import uuid
import math
import time
import copy

from .base import BaseAI
from .heuristic_ai import HeuristicAI
from .neural_net import NeuralNetAI
from ..models import GameState, Move, AIConfig, Position

class MCTSNode:
    def __init__(self, game_state: GameState, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.amaf_wins = 0
        self.amaf_visits = 0
        self.untried_moves = [] # Should be populated with valid moves

    def uct_select_child(self):
        """Select child using PUCT formula with RAVE"""
        # PUCT = Q + c_puct * P * sqrt(N) / (1 + n)
        # RAVE: Value = (1 - beta) * Q + beta * AMAF
        
        c_puct = 1.0 # Exploration constant
        rave_k = 1000.0 # RAVE equivalence parameter
        
        def puct_value(child):
            # Calculate Q (MC value)
            if child.visits == 0:
                q_value = 0.0
            else:
                q_value = child.wins / child.visits
                
            # Calculate AMAF (RAVE value)
            if child.amaf_visits == 0:
                amaf_value = 0.0
            else:
                amaf_value = child.amaf_wins / child.amaf_visits
                
            # Calculate beta for RAVE
            beta = math.sqrt(rave_k / (3 * self.visits + rave_k))
            
            # Combined value
            combined_value = (1 - beta) * q_value + beta * amaf_value
            
            # Prior probability P(s, a)
            prior = getattr(child, 'prior', 1.0 / len(self.children))
            
            u_value = c_puct * prior * math.sqrt(self.visits) / (1 + child.visits)
            return combined_value + u_value
            
        return max(self.children, key=puct_value)

    def add_child(self, move, game_state, prior=None):
        """Add a new child node"""
        child = MCTSNode(game_state, parent=self, move=move)
        if prior is not None:
            child.prior = prior
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result, played_moves=None):
        """Update node stats"""
        self.visits += 1
        self.wins += result
        
        if played_moves and self.move:
            # Update AMAF stats if this node's move was played in the simulation
            # Check if self.move is in played_moves
            # Move equality check might need to be robust
            # For simplicity, check type and to/from
            for m in played_moves:
                if (m.type == self.move.type and
                    m.to.x == self.move.to.x and m.to.y == self.move.to.y):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break

class MCTSAI(HeuristicAI):
    """AI that uses Monte Carlo Tree Search"""
    
    def __init__(self, player_number: int, config: AIConfig):
        super().__init__(player_number, config)
        # Try to load neural net for evaluation
        try:
            self.neural_net = NeuralNetAI(player_number, config)
        except Exception:
            self.neural_net = None

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using MCTS
        
        Args:
            game_state: Current game state
            
        Returns:
            Best move or None if no valid moves
        """
        move, _ = self.select_move_and_policy(game_state)
        return move

    def select_move_and_policy(self, game_state: GameState) -> tuple[Optional[Move], Optional[Dict[Move, float]]]:
        """
        Select the best move using MCTS and return the policy distribution
        
        Args:
            game_state: Current game state
            
        Returns:
            Tuple of (Best move, Policy distribution)
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=500, max_ms=2000)
        
        from ..game_engine import GameEngine
        
        # Get all valid moves
        valid_moves = GameEngine.get_valid_moves(game_state, self.player_number)
        
        if not valid_moves:
            return None, None
            
        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = random.choice(valid_moves)
            # Create a simple policy for random move
            policy = {m: (1.0 if m == selected else 0.0) for m in valid_moves}
            return selected, policy
        else:
            # MCTS parameters
            # Time limit based on difficulty
            # Increase time limit for better search depth
            if self.config.think_time is not None and self.config.think_time > 0:
                time_limit = self.config.think_time / 1000.0
            else:
                time_limit = 1.0 + (self.config.difficulty * 0.5) # 1.5s to 6.0s
            
            # Tree Reuse: Check if we have a subtree for the current state
            root = None
            if hasattr(self, 'last_root') and self.last_root is not None:
                # Try to find the child corresponding to the opponent's move
                # We need to know what move led to the current state from the previous state
                # But we don't track opponent moves explicitly here.
                # However, we can check if any child of last_root matches the current state.
                # This is expensive if we do full state comparison.
                # A better way is to just reset for now, or implement move tracking in the AI instance.
                # Given the stateless nature of the API request, we can't easily reuse the tree
                # unless we cache it by game ID.
                # The current architecture instantiates a new AI for each request (or retrieves from cache).
                # If retrieved from cache, 'self' is persistent.
                
                # Let's assume we can find the child.
                # But we don't know the opponent's move.
                # We can try to match the board state.
                pass
                
            if root is None:
                root = MCTSNode(game_state)
                root.untried_moves = valid_moves
            
            end_time = time.time() + time_limit
            
            # Batched Inference Setup
            # We collect leaf nodes to evaluate in a batch
            batch_size = 8 # Small batch for CPU/latency balance
            
            # MCTS implementation with PUCT
            while time.time() < end_time:
                # Selection Phase - Collect a batch of leaves
                leaves = []
                
                for _ in range(batch_size):
                    node = root
                    state = copy.deepcopy(node.game_state)
                    played_moves = []
                    
                    # Selection
                    while not node.untried_moves and node.children:
                        node = node.uct_select_child()
                        state = GameEngine.apply_move(state, node.move)
                        played_moves.append(node.move)
                    
                    # Expansion
                    if node.untried_moves:
                        m = random.choice(node.untried_moves)
                        state = GameEngine.apply_move(state, m)
                        
                        # Get prior from parent's policy if available
                        prior = None
                        if hasattr(node, 'policy_map') and m in node.policy_map:
                            prior = node.policy_map[m]
                            
                        node = node.add_child(m, state, prior=prior)
                        played_moves.append(m)
                    
                    leaves.append((node, state, played_moves))
                    
                    # Check time to avoid overrunning too much
                    if time.time() >= end_time:
                        break
                
                if not leaves:
                    break
                    
                # Evaluation Phase (Batched)
                if self.neural_net:
                    # Prepare batch
                    states = [l[1] for l in leaves]
                    
                    # Use batched evaluation
                    values, policies = self.neural_net.evaluate_batch(states)
                        
                    # Process results
                    for i in range(len(leaves)):
                        value = values[i]
                        policy = policies[i]
                        node, state, played_moves = leaves[i]
                        
                        # Store policy priors
                        valid_moves_state = GameEngine.get_valid_moves(state, state.current_player)
                        if valid_moves_state:
                            node.untried_moves = valid_moves_state
                            node.policy_map = {}
                            
                            total_prob = 0.0
                            for move in valid_moves_state:
                                idx = self.neural_net.encode_move(move, state.board.size)
                                if 0 <= idx < len(policy):
                                    prob = float(policy[idx])
                                    node.policy_map[move] = prob
                                    total_prob += prob
                            
                            if total_prob > 0:
                                for move in node.policy_map:
                                    node.policy_map[move] /= total_prob
                            else:
                                uniform = 1.0 / len(valid_moves_state)
                                for move in valid_moves_state:
                                    node.policy_map[move] = uniform
                        
                        # Backpropagation
                        # Value is from perspective of state.current_player (who is about to move in 'state')
                        # But 'state' is the result of the move leading to 'node'.
                        # So state.current_player is the opponent of the player who made the move.
                        
                        # Standard AlphaZero:
                        # v is value for state.current_player.
                        # Parent node (who made the move) should receive -v.
                        
                        current_val = value
                        curr_node = node
                        
                        while curr_node is not None:
                            curr_node.update(current_val, played_moves)
                            current_val = -current_val
                            curr_node = curr_node.parent
                            
                else:
                    # Fallback to Heuristic Rollout (Sequential)
                    for node, state, played_moves in leaves:
                        rollout_depth = 3
                        rollout_state = state
                        
                        for _ in range(rollout_depth):
                            if rollout_state.game_status == "finished":
                                break
                                
                            moves = GameEngine.get_valid_moves(rollout_state, rollout_state.current_player)
                            if not moves:
                                break
                                
                            # Weighted selection
                            weights = []
                            for m in moves:
                                w = 1.0
                                if m.type == "territory_claim": w = 50.0
                                elif m.type == "line_formation": w = 25.0
                                elif m.type == "chain_capture": w = 10.0
                                elif m.type == "overtaking_capture": w = 5.0
                                elif m.type == "move_stack": w = 2.0
                                weights.append(w)
                                
                            selected_move = random.choices(moves, weights=weights, k=1)[0]
                            rollout_state = GameEngine.apply_move(rollout_state, selected_move)
                            
                        result = self.evaluate_position(rollout_state)
                        
                        # Backpropagation
                        # Heuristic evaluation is usually from perspective of player who just moved?
                        # Or absolute?
                        # evaluate_position returns score for self.player_number (the AI instance)
                        # But rollout_state.current_player might be different.
                        
                        # Let's assume evaluate_position returns score for the player whose turn it is in rollout_state?
                        # No, HeuristicAI.evaluate_position(state) usually returns score for 'self.player_number'.
                        # This is problematic for MCTS if we don't know who 'self' is relative to the node.
                        
                        # We need a perspective-neutral evaluation or flip it.
                        # For now, let's assume result is from perspective of root player (self.player_number).
                        
                        # If result is for root player:
                        # Root node: update(result)
                        # Child of root (opponent moved): update(-result)?
                        # This depends on how MCTSNode.update works.
                        # If MCTSNode stores "wins for the player who moved to get here", then:
                        # If root player moved to get to Root (conceptually), then result.
                        # If opponent moved to get to Child, then -result.
                        
                        # Let's stick to the alternating sign convention which is robust for zero-sum.
                        # We need 'result' to be from the perspective of the leaf node's player-to-move.
                        
                        # If evaluate_position returns score for self.player_number (Root Player):
                        # And leaf node state has current_player = P.
                        # If P == Root Player, val = result.
                        # If P != Root Player, val = -result.
                        
                        val_for_leaf_player = result if rollout_state.current_player == self.player_number else -result
                        
                        current_val = val_for_leaf_player
                        curr_node = node
                        while curr_node is not None:
                            curr_node.update(current_val, played_moves)
                            current_val = -current_val
                            curr_node = curr_node.parent
            
            # Select best move based on visits
            if root.children:
                total_visits = sum(c.visits for c in root.children)
                policy = {}
                if total_visits > 0:
                    for child in root.children:
                        policy[child.move] = child.visits / total_visits
                else:
                    uniform = 1.0 / len(root.children)
                    for child in root.children:
                        policy[child.move] = uniform
                
                # Robust child selection
                best_child = max(root.children, key=lambda c: c.visits)
                selected = best_child.move
                
                # Save subtree for reuse (if we were to implement persistent tree)
                # self.last_root = best_child
                # self.last_root.parent = None # Detach to allow GC of old tree
            else:
                selected = random.choice(valid_moves)
                policy = {m: (1.0 if m == selected else 0.0) for m in valid_moves}
        
        self.move_count += 1
        return selected, policy
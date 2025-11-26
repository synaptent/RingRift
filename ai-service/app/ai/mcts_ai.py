"""
MCTS AI implementation for RingRift
Uses Monte Carlo Tree Search for move selection
"""

import logging
from typing import Optional, Dict, Any, cast
import math
import time

import numpy as np
import torch

from .bounded_transposition_table import BoundedTranspositionTable
from .heuristic_ai import HeuristicAI
from .neural_net import (
    NeuralNetAI,
    INVALID_MOVE_INDEX,
    ActionEncoderHex,
    HexNeuralNet,
)
from ..models import GameState, Move, AIConfig, BoardType
from ..utils.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


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
        self.untried_moves = []  # Should be populated with valid moves
        self.prior = 0.0
        self.policy_map = {}

    def uct_select_child(self):
        """Select child using PUCT formula with RAVE"""
        # PUCT = Q + c_puct * P * sqrt(N) / (1 + n)
        # RAVE: Value = (1 - beta) * Q + beta * AMAF

        c_puct = 1.0  # Exploration constant
        rave_k = 1000.0  # RAVE equivalence parameter

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
            prior = getattr(child, "prior", 1.0 / len(self.children))

            u_value = (
                c_puct * prior * math.sqrt(self.visits) / (1 + child.visits)
            )
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
            # Update AMAF stats if this node's move was played in the
            # simulation. Check if self.move is in played_moves.
            # Move equality check might need to be robust.
            # For simplicity, check type and to/from.
            for m in played_moves:
                if (
                    m.type == self.move.type
                    and m.to.x == self.move.to.x
                    and m.to.y == self.move.to.y
                ):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break


class MCTSAI(HeuristicAI):
    """AI that uses Monte Carlo Tree Search"""
    
    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        memory_config: Optional[MemoryConfig] = None,
    ):
        super().__init__(player_number, config)
        # Try to load neural net for evaluation
        try:
            self.neural_net = NeuralNetAI(player_number, config)
        except Exception:
            self.neural_net = None

        # Optional hex-specific encoder and network (used for hex boards).
        self.hex_encoder: Optional[ActionEncoderHex]
        self.hex_model: Optional[HexNeuralNet]
        if self.neural_net is not None:
            try:
                # in_channels and global_features must match _extract_features.
                self.hex_encoder = ActionEncoderHex()
                self.hex_model = HexNeuralNet(
                    in_channels=10,
                    global_features=10,
                )
            except Exception:
                self.hex_encoder = None
                self.hex_model = None
        else:
            self.hex_encoder = None
            self.hex_model = None

        # Memory configuration for bounded structures
        self.memory_config = memory_config or MemoryConfig.from_env()
        
        # Transposition table to cache neural network evaluations
        # Key: state_hash, Value: (value, policy)
        # Entry size estimate: ~400 bytes (MCTS stores larger policies)
        tt_limit = self.memory_config.get_transposition_table_limit_bytes()
        self.transposition_table = BoundedTranspositionTable.from_memory_limit(
            tt_limit,
            entry_size_estimate=400,
        )

    def simulate_thinking(self, min_ms: int = 100, max_ms: int = 2000) -> None:
        """Override BaseAI.simulate_thinking.

        For search-based AIs we interpret config.think_time as the total
        search budget, so MCTSAI does not add any additional sleep on top
        of its Monte Carlo loop.
        """
        return

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select the best move using MCTS"""
        move, _ = self.select_move_and_policy(game_state)
        return move

    def select_move_and_policy(
        self,
        game_state: GameState,
    ) -> tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Select the best move using MCTS and return the policy distribution.

        Returns a tuple of (Best move, Policy distribution).
        """
        # For MCTS we treat config.think_time as a search-time budget; no
        # extra sleep is introduced here.

        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None, None

        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
            # Create a simple policy for random move
            policy = {
                str(m): (1.0 if m == selected else 0.0) for m in valid_moves
            }
            return selected, policy

        # MCTS parameters
        # Time limit based on difficulty
        # Increase time limit for better search depth
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            # 1.5s to 6.0s
            time_limit = 1.0 + (self.config.difficulty * 0.5)

        # Tree Reuse: Check if we have a subtree for the current state
        root = None
        if hasattr(self, "last_root") and self.last_root is not None:
            # Find child corresponding to the last move played
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                for child in self.last_root.children:
                    if (
                        child.move.type == last_move.type
                        and child.move.to.x == last_move.to.x
                        and child.move.to.y == last_move.to.y
                    ):
                        # Found subtree
                        root = child
                        root.parent = None  # Detach
                        break

        if root is None:
            root = MCTSNode(game_state)
            root.untried_moves = valid_moves

        end_time = time.time() + time_limit

        # Batched Inference Setup
        # We collect leaf nodes to evaluate in a batch
        batch_size = 8  # Small batch for CPU/latency balance
        
        # MCTS implementation with PUCT
        while time.time() < end_time:
            # Selection Phase - Collect a batch of leaves
            leaves = []
            
            for _ in range(batch_size):
                node = root
                # No deepcopy needed as apply_move returns new state
                state = node.game_state
                played_moves = []

                # Selection
                while not node.untried_moves and node.children:
                    node = node.uct_select_child()
                    state = self.rules_engine.apply_move(state, node.move)
                    played_moves.append(node.move)

                # Expansion
                if node.untried_moves:
                    m = cast(Move, self.get_random_element(node.untried_moves))
                    state = self.rules_engine.apply_move(state, m)

                    # Get prior from parent's policy if available
                    prior = None
                    m_key = str(m)
                    if m_key in node.policy_map:
                        prior = node.policy_map[m_key]

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
                # Prepare batch - check transposition table first
                states = [leaf[1] for leaf in leaves]
                
                # Check cache for already-evaluated states
                cached_results = []
                uncached_indices = []
                uncached_states = []
                
                for i, state in enumerate(states):
                    state_hash = hash(state)
                    cached = self.transposition_table.get(state_hash)
                    if cached is not None:
                        cached_results.append((i, cached[0], cached[1]))
                    else:
                        uncached_indices.append(i)
                        uncached_states.append(state)
                
                # Initialize result arrays with proper types
                values: list = [0.0] * len(states)
                policies: list = [None] * len(states)
                
                # Fill in cached results
                for idx, val, pol in cached_results:
                    values[idx] = val
                    policies[idx] = pol

                # Decide whether to use the hex-specific network based on
                # the current board geometry. All states in a batch share
                # the same board.type/size (enforced by evaluate_batch).
                use_hex_nn = (
                    self.hex_model is not None
                    and self.hex_encoder is not None
                    and states
                    and states[0].board.type == BoardType.HEXAGONAL
                )
                
                # Only evaluate uncached states
                if uncached_states:
                    if use_hex_nn:
                        # Build feature and global tensors compatible with
                        # HexNeuralNet using NeuralNetAI's encoder.
                        feature_batches = []
                        globals_batches = []
                        for s in uncached_states:
                            feats, globs = self.neural_net._extract_features(s)
                            feature_batches.append(feats)
                            globals_batches.append(globs)

                        tensor_input = torch.FloatTensor(
                            np.array(feature_batches)
                        )
                        globals_input = torch.FloatTensor(
                            np.array(globals_batches)
                        )

                        # hex_model is not None when use_hex_nn is True
                        hex_model = cast(HexNeuralNet, self.hex_model)
                        with torch.no_grad():
                            values_tensor, policy_logits = hex_model(
                                tensor_input, globals_input, hex_mask=None
                            )
                            policy_probs = torch.softmax(policy_logits, dim=1)

                        eval_values = (
                            values_tensor.cpu().numpy().flatten().tolist()
                        )
                        eval_policies = policy_probs.cpu().numpy()
                    else:
                        # Use the square/shared network path.
                        eval_values, eval_policies = (
                            self.neural_net.evaluate_batch(uncached_states)
                        )
                    
                    # Store uncached results and cache them
                    for j, orig_idx in enumerate(uncached_indices):
                        values[orig_idx] = eval_values[j]
                        policies[orig_idx] = eval_policies[j]
                        
                        # Cache the result
                        state_hash = hash(uncached_states[j])
                        self.transposition_table.put(
                            state_hash,
                            (eval_values[j], eval_policies[j]),
                        )

                # Process results
                for i in range(len(leaves)):
                    value = values[i]
                    policy = policies[i]
                    node, state, played_moves = leaves[i]

                    # Skip if policy wasn't computed (shouldn't happen)
                    if policy is None:
                        continue

                    # Store policy priors
                    valid_moves_state = self.rules_engine.get_valid_moves(
                        state,
                        state.current_player,
                    )
                    if valid_moves_state:
                        node.untried_moves = valid_moves_state
                        node.policy_map = {}

                        total_prob = 0.0
                        for move in valid_moves_state:
                            if use_hex_nn and self.hex_encoder is not None:
                                idx = self.hex_encoder.encode_move(
                                    move, state.board
                                )
                            elif self.neural_net is not None:
                                # Encode using canonical coordinates
                                # derived from current board geometry.
                                # Moves outside the fixed 19×19 policy
                                # grid return INVALID_MOVE_INDEX.
                                idx = self.neural_net.encode_move(
                                    move, state.board
                                )
                            else:
                                idx = INVALID_MOVE_INDEX

                            if (
                                idx != INVALID_MOVE_INDEX
                                and 0 <= idx < len(policy)
                            ):
                                # policy is numpy array, idx is int
                                prob = float(cast(Any, policy)[idx])
                                # Use string key for dict
                                node.policy_map[str(move)] = prob
                                total_prob += prob

                        if total_prob > 0:
                            for move_key in node.policy_map:
                                node.policy_map[move_key] /= total_prob
                        else:
                            uniform = 1.0 / len(valid_moves_state)
                            for move in valid_moves_state:
                                node.policy_map[str(move)] = uniform

                    # Backpropagation
                    current_val = float(value) if value is not None else 0.0
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

                        moves = self.rules_engine.get_valid_moves(
                            rollout_state,
                            rollout_state.current_player,
                        )
                        if not moves:
                            break

                        # Light playout: weight moves by domain knowledge
                        # to avoid random-walk bias
                        weights = []
                        for m in moves:
                            w = 1.0
                            if m.type == "territory_claim":
                                w = 100.0
                            elif m.type == "line_formation":
                                w = 50.0
                            elif m.type == "chain_capture":
                                w = 20.0
                            elif m.type == "overtaking_capture":
                                w = 10.0
                            elif m.type == "move_stack":
                                # Prefer stacks (merges) or captures
                                # If to-space has stack, it's a merge
                                to_key = m.to.to_key()
                                if to_key in rollout_state.board.stacks:
                                    w = 5.0
                                else:
                                    w = 2.0
                            elif m.type == "place_ring":
                                # Prefer placing near own stacks
                                w = 1.5
                            weights.append(w)

                            selected_move = self.rng.choices(
                                moves,
                                weights=weights,
                                k=1,
                            )[0]
                            rollout_state = self.rules_engine.apply_move(
                                rollout_state,
                                selected_move,
                            )

                    result = self.evaluate_position(rollout_state)

                    # Backpropagation
                    if rollout_state.current_player == self.player_number:
                        val_for_leaf_player = result
                    else:
                        val_for_leaf_player = -result

                    current_val = val_for_leaf_player
                    curr_node = node
                    while curr_node is not None:
                        curr_node.update(current_val, played_moves)
                        current_val = -current_val
                        curr_node = curr_node.parent
        
        # Log transposition table stats at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            tt_stats = self.transposition_table.stats()
            logger.debug(
                "MCTS transposition table stats: entries=%d/%d, "
                "hits=%d, misses=%d, hit_rate=%.2f%%, evictions=%d, "
                "est_memory=%.2fMB",
                tt_stats["entries"],
                tt_stats["max_entries"],
                tt_stats["hits"],
                tt_stats["misses"],
                tt_stats["hit_rate"] * 100,
                tt_stats["evictions"],
                tt_stats["estimated_memory_mb"],
            )
        
        # Select best move based on visits
        if root.children:
            total_visits = sum(c.visits for c in root.children)
            policy: Dict[str, float] = {}
            if total_visits > 0:
                for child in root.children:
                    # Use string key for dict
                    policy[str(child.move)] = child.visits / total_visits
            else:
                uniform = 1.0 / len(root.children)
                for child in root.children:
                    policy[str(child.move)] = uniform
            
            # Robust child selection
            best_child = max(root.children, key=lambda c: c.visits)
            selected = best_child.move
            
            # Save subtree for reuse
            self.last_root = best_child
            self.last_root.parent = None  # Detach to allow GC of old tree
        else:
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0)
                for m in valid_moves
            }

        self.move_count += 1
        return selected, policy

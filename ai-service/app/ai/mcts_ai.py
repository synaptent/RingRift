"""
MCTS AI implementation for RingRift
Uses Monte Carlo Tree Search for move selection

When `config.use_incremental_search` is True (the default), MCTSAI uses
the make/unmake pattern on MutableGameState for faster search by avoiding
object allocation overhead. When False, it falls back to the legacy
immutable state cloning via apply_move().
"""

import logging
from typing import Optional, Dict, Any, cast, List, Tuple
import math
import time

import numpy as np
import psutil
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
from ..rules.mutable_state import MutableGameState, MoveUndo
from ..utils.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class MCTSNode:
    """MCTS tree node for legacy (immutable) search.
    
    Stores a reference to the game state for each node in the tree.
    """
    def __init__(self, game_state: GameState, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children: List["MCTSNode"] = []
        self.wins = 0
        self.visits = 0
        self.amaf_wins = 0
        self.amaf_visits = 0
        self.untried_moves: List[Move] = []
        self.prior = 0.0
        self.policy_map: Dict[str, float] = {}

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


class MCTSNodeLite:
    """Lightweight MCTS tree node for incremental (mutable) search.
    
    Does NOT store game state - state is maintained externally via
    make/unmake pattern. This reduces memory footprint significantly.
    """
    __slots__ = [
        'parent', 'move', 'children', 'wins', 'visits',
        'amaf_wins', 'amaf_visits', 'untried_moves', 'prior', 'policy_map'
    ]
    
    def __init__(
        self,
        parent: Optional["MCTSNodeLite"] = None,
        move: Optional[Move] = None
    ):
        self.parent = parent
        self.move = move
        self.children: List["MCTSNodeLite"] = []
        self.wins = 0.0
        self.visits = 0
        self.amaf_wins = 0.0
        self.amaf_visits = 0
        self.untried_moves: List[Move] = []
        self.prior = 0.0
        self.policy_map: Dict[str, float] = {}
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0
    
    def uct_select_child(self) -> "MCTSNodeLite":
        """Select child using PUCT formula with RAVE."""
        c_puct = 1.0
        rave_k = 1000.0
        
        def puct_value(child: "MCTSNodeLite") -> float:
            if child.visits == 0:
                q_value = 0.0
            else:
                q_value = child.wins / child.visits
            
            if child.amaf_visits == 0:
                amaf_value = 0.0
            else:
                amaf_value = child.amaf_wins / child.amaf_visits
            
            beta = math.sqrt(rave_k / (3 * self.visits + rave_k))
            combined_value = (1 - beta) * q_value + beta * amaf_value
            
            num_children = max(1, len(self.children))
            prior = child.prior if child.prior > 0 else 1.0 / num_children
            sqrt_visits = math.sqrt(self.visits)
            u_value = c_puct * prior * sqrt_visits / (1 + child.visits)
            
            return combined_value + u_value
        
        return max(self.children, key=puct_value)
    
    def add_child(
        self,
        move: Move,
        prior: Optional[float] = None
    ) -> "MCTSNodeLite":
        """Add a new child node."""
        child = MCTSNodeLite(parent=self, move=move)
        if prior is not None:
            child.prior = prior
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        self.children.append(child)
        return child
    
    def update(
        self,
        result: float,
        played_moves: Optional[List[Move]] = None
    ) -> None:
        """Update node stats."""
        self.visits += 1
        self.wins += result
        
        if played_moves and self.move:
            for m in played_moves:
                if (
                    m.type == self.move.type
                    and m.to.x == self.move.to.x
                    and m.to.y == self.move.to.y
                ):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break


class DynamicBatchSizer:
    """Dynamically adjusts MCTS batch size based on available memory.
    
    This class monitors system memory and calculates optimal batch sizes
    for MCTS simulations, allowing more simulations when memory permits
    and gracefully reducing when memory is constrained.
    """
    
    def __init__(
        self,
        memory_config: Optional[MemoryConfig] = None,
        batch_size_min: int = 100,
        batch_size_max: int = 1600,
        memory_safety_margin: float = 0.8,
        node_size_estimate: int = 500,
    ):
        """Initialize the dynamic batch sizer.
        
        Args:
            memory_config: Memory configuration for limits
            batch_size_min: Minimum batch size (default: 100)
            batch_size_max: Maximum batch size (default: 1600)
            memory_safety_margin: Only use this fraction of available budget
                (default: 0.8 = 80%)
            node_size_estimate: Estimated bytes per MCTS node (default: 500)
        """
        self.memory_config = memory_config or MemoryConfig.from_env()
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.memory_safety_margin = memory_safety_margin
        self.node_size_estimate = node_size_estimate
        
        # Track actual memory usage to refine estimates
        self._memory_samples: List[Tuple[int, int]] = []
        self._last_batch_size = batch_size_max
        self._adjustment_count = 0
    
    def get_optimal_batch_size(self, current_node_count: int = 0) -> int:
        """Calculate optimal batch size based on available memory.
        
        Args:
            current_node_count: Current number of nodes in the tree
            
        Returns:
            Optimal batch size within configured limits
        """
        # Get available memory
        mem_info = psutil.virtual_memory()
        available_bytes = mem_info.available
        
        # Get inference budget from config
        inference_budget = self.memory_config.get_inference_limit_bytes()
        
        # Apply safety margin
        safe_budget = int(inference_budget * self.memory_safety_margin)
        
        # Use minimum of available memory and safe budget
        usable_memory = min(available_bytes, safe_budget)
        
        # Estimate memory already used by current tree
        current_tree_memory = current_node_count * self.node_size_estimate
        
        # Calculate remaining memory for new nodes
        remaining_memory = max(0, usable_memory - current_tree_memory)
        
        # Calculate how many new nodes we can create
        # Each batch iteration may create up to batch_size new nodes
        max_new_nodes = remaining_memory // self.node_size_estimate
        
        # Clamp to configured limits
        optimal = max(
            self.batch_size_min, min(self.batch_size_max, max_new_nodes)
        )
        
        # Log if batch size changes significantly
        if abs(optimal - self._last_batch_size) >= self.batch_size_min // 2:
            self._adjustment_count += 1
            logger.info(
                "Dynamic batch size adjusted: %d -> %d (available=%.2fMB, "
                "budget=%.2fMB, tree_nodes=%d, adjustments=%d)",
                self._last_batch_size,
                optimal,
                available_bytes / (1024**2),
                safe_budget / (1024**2),
                current_node_count,
                self._adjustment_count,
            )
            self._last_batch_size = optimal
        
        return optimal
    
    def record_memory_sample(self, node_count: int) -> None:
        """Record a memory sample to refine node size estimates.
        
        Args:
            node_count: Current number of nodes in the tree
        """
        # Get current process memory
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        
        self._memory_samples.append((node_count, memory_bytes))
        
        # Keep only recent samples (last 100)
        if len(self._memory_samples) > 100:
            self._memory_samples = self._memory_samples[-100:]
        
        # Refine estimate if we have enough samples
        if len(self._memory_samples) >= 10:
            self._refine_node_size_estimate()
    
    def _refine_node_size_estimate(self) -> None:
        """Refine node size estimate based on memory samples."""
        if len(self._memory_samples) < 2:
            return
        
        # Calculate average memory increase per node
        samples = sorted(self._memory_samples, key=lambda x: x[0])
        
        # Use linear regression-like approach
        deltas = []
        for i in range(1, len(samples)):
            node_delta = samples[i][0] - samples[i - 1][0]
            mem_delta = samples[i][1] - samples[i - 1][1]
            
            if node_delta > 0:
                deltas.append(mem_delta / node_delta)
        
        if deltas:
            # Use median to avoid outliers
            deltas.sort()
            median_idx = len(deltas) // 2
            estimated = int(deltas[median_idx])
            
            # Only update if estimate is reasonable
            if 100 <= estimated <= 2000:
                old_estimate = self.node_size_estimate
                self.node_size_estimate = estimated
                
                if abs(self.node_size_estimate - old_estimate) > 50:
                    logger.debug(
                        "Refined MCTS node size estimate: %d -> %d bytes",
                        old_estimate,
                        self.node_size_estimate,
                    )
    
    def stats(self) -> dict:
        """Return current stats.
        
        Returns:
            Dictionary with current batch size, estimates, and adjustment count
        """
        return {
            "current_batch_size": self._last_batch_size,
            "node_size_estimate": self.node_size_estimate,
            "batch_size_min": self.batch_size_min,
            "batch_size_max": self.batch_size_max,
            "memory_safety_margin": self.memory_safety_margin,
            "adjustment_count": self._adjustment_count,
            "sample_count": len(self._memory_samples),
        }


class MCTSAI(HeuristicAI):
    """AI that uses Monte Carlo Tree Search.
    
    When `config.use_incremental_search` is True (the default), MCTSAI
    uses the make/unmake pattern on MutableGameState for faster search
    by avoiding object allocation overhead. When False, it falls back
    to the legacy immutable state cloning via apply_move().
    """
    
    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        memory_config: Optional[MemoryConfig] = None,
        dynamic_sizer: Optional[DynamicBatchSizer] = None,
        enable_dynamic_batching: bool = False,
    ):
        """Initialize MCTS AI.
        
        Args:
            player_number: The player number (1 or 2)
            config: AI configuration
            memory_config: Memory configuration for bounded structures
            dynamic_sizer: Optional dynamic batch sizer for memory-aware
                batching
            enable_dynamic_batching: Enable dynamic batching (default:
                False for backward compatibility). If True and dynamic_sizer
                is None, a default DynamicBatchSizer will be created.
        """
        super().__init__(player_number, config)
        
        # Configuration option for incremental search (make/unmake pattern)
        self.use_incremental_search = getattr(
            config, 'use_incremental_search', True
        )
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
        
        # Dynamic batch sizing (optional)
        self.enable_dynamic_batching = enable_dynamic_batching
        if enable_dynamic_batching:
            self.dynamic_sizer = dynamic_sizer or DynamicBatchSizer(
                memory_config=self.memory_config
            )
        else:
            self.dynamic_sizer = dynamic_sizer
        
        # Lightweight tree root for incremental search
        self.last_root_lite: Optional[MCTSNodeLite] = None

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
        
        Routes to incremental (make/unmake) or legacy (immutable) search
        based on the use_incremental_search configuration.
        """
        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None, None

        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0) for m in valid_moves
            }
            return selected, policy

        # Route to incremental or legacy search based on config
        if self.use_incremental_search:
            return self._search_incremental(game_state, valid_moves)
        else:
            return self._search_legacy(game_state, valid_moves)

    # =========================================================================
    # Legacy Search Methods (Immutable State)
    # =========================================================================

    def _search_legacy(
        self,
        game_state: GameState,
        valid_moves: List[Move],
    ) -> tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Legacy MCTS search using immutable state cloning via apply_move().
        
        This is the original implementation preserved for backward
        compatibility and A/B testing against the new incremental search.
        """
        # MCTS parameters - time limit based on difficulty
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            time_limit = 1.0 + (self.config.difficulty * 0.5)

        # Tree Reuse: Check if we have a subtree for the current state
        root: Optional[MCTSNode] = None
        if hasattr(self, "last_root") and self.last_root is not None:
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                for child in self.last_root.children:
                    if (
                        child.move.type == last_move.type
                        and child.move.to.x == last_move.to.x
                        and child.move.to.y == last_move.to.y
                    ):
                        root = child
                        root.parent = None
                        break

        if root is None:
            root = MCTSNode(game_state)
            root.untried_moves = list(valid_moves)

        end_time = time.time() + time_limit
        default_batch_size = 8
        node_count = 1

        # MCTS implementation with PUCT
        while time.time() < end_time:
            if self.enable_dynamic_batching and self.dynamic_sizer is not None:
                batch_size = self.dynamic_sizer.get_optimal_batch_size(
                    node_count
                )
            else:
                batch_size = default_batch_size
            
            leaves: List[Tuple[MCTSNode, GameState, List[Move]]] = []
            
            for _ in range(batch_size):
                node = root
                state = node.game_state
                played_moves: List[Move] = []

                # Selection
                while not node.untried_moves and node.children:
                    node = node.uct_select_child()
                    if node.move is not None:
                        state = self.rules_engine.apply_move(state, node.move)
                        played_moves.append(node.move)

                # Expansion
                if node.untried_moves:
                    m = cast(Move, self.get_random_element(node.untried_moves))
                    state = self.rules_engine.apply_move(state, m)

                    prior = None
                    m_key = str(m)
                    if m_key in node.policy_map:
                        prior = node.policy_map[m_key]

                    child = node.add_child(m, state, prior=prior)
                    node = child
                    node_count += 1
                    played_moves.append(m)

                leaves.append((node, state, played_moves))

                if time.time() >= end_time:
                    break

            if not leaves:
                break

            # Evaluation Phase
            self._evaluate_leaves_legacy(leaves, root)
        
        # Record memory sample if dynamic batching is enabled
        if self.enable_dynamic_batching and self.dynamic_sizer is not None:
            self.dynamic_sizer.record_memory_sample(node_count)
        
        self._log_stats()
        
        # Select best move based on visits
        return self._select_best_move_legacy(root, valid_moves)

    def _evaluate_leaves_legacy(
        self,
        leaves: List[Tuple[MCTSNode, GameState, List[Move]]],
        root: MCTSNode,
    ) -> None:
        """Evaluate leaf nodes using neural network or heuristic rollout."""
        if self.neural_net:
            states = [leaf[1] for leaf in leaves]
            
            cached_results: List[Tuple[int, float, Any]] = []
            uncached_indices: List[int] = []
            uncached_states: List[GameState] = []
            
            for i, state in enumerate(states):
                state_hash = state.zobrist_hash or 0
                cached = self.transposition_table.get(state_hash)
                if cached is not None:
                    cached_results.append((i, cached[0], cached[1]))
                else:
                    uncached_indices.append(i)
                    uncached_states.append(state)
            
            values: List[float] = [0.0] * len(states)
            policies: List[Any] = [None] * len(states)
            
            for idx, val, pol in cached_results:
                values[idx] = val
                policies[idx] = pol

            use_hex_nn = (
                self.hex_model is not None
                and self.hex_encoder is not None
                and states
                and states[0].board.type == BoardType.HEXAGONAL
            )
            
            if uncached_states:
                if use_hex_nn:
                    eval_values, eval_policies = self._evaluate_hex_batch(
                        uncached_states
                    )
                else:
                    eval_values, eval_policies = (
                        self.neural_net.evaluate_batch(uncached_states)
                    )
                
                for j, orig_idx in enumerate(uncached_indices):
                    values[orig_idx] = eval_values[j]
                    policies[orig_idx] = eval_policies[j]
                    
                    state_hash = uncached_states[j].zobrist_hash or 0
                    self.transposition_table.put(
                        state_hash,
                        (eval_values[j], eval_policies[j]),
                    )

            # Process results
            for i in range(len(leaves)):
                value = values[i]
                policy = policies[i]
                node, state, played_moves = leaves[i]

                if policy is None:
                    continue

                self._update_node_policy_legacy(
                    node, state, policy, bool(use_hex_nn)
                )

                # Backpropagation
                current_val = float(value) if value is not None else 0.0
                curr_node: Optional[MCTSNode] = node

                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    current_val = -current_val
                    curr_node = curr_node.parent
        else:
            # Fallback to Heuristic Rollout
            for node, state, played_moves in leaves:
                result = self._heuristic_rollout_legacy(state)

                if state.current_player == self.player_number:
                    val_for_leaf_player = result
                else:
                    val_for_leaf_player = -result

                current_val = val_for_leaf_player
                curr_node: Optional[MCTSNode] = node
                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    current_val = -current_val
                    curr_node = curr_node.parent

    def _evaluate_hex_batch(
        self, states: List[GameState]
    ) -> Tuple[List[float], List[Any]]:
        """Evaluate a batch of hex board states."""
        feature_batches = []
        globals_batches = []
        nn = self.neural_net  # type assertion for Pylance
        assert nn is not None
        for s in states:
            feats, globs = nn._extract_features(s)
            feature_batches.append(feats)
            globals_batches.append(globs)

        tensor_input = torch.FloatTensor(np.array(feature_batches))
        globals_input = torch.FloatTensor(np.array(globals_batches))

        hex_model = cast(HexNeuralNet, self.hex_model)
        with torch.no_grad():
            values_tensor, policy_logits = hex_model(
                tensor_input, globals_input, hex_mask=None
            )
            policy_probs = torch.softmax(policy_logits, dim=1)

        eval_values = values_tensor.cpu().numpy().flatten().tolist()
        eval_policies = policy_probs.cpu().numpy()
        
        return eval_values, list(eval_policies)

    def _update_node_policy_legacy(
        self,
        node: MCTSNode,
        state: GameState,
        policy: Any,
        use_hex_nn: bool,
    ) -> None:
        """Update node policy priors from neural network output."""
        valid_moves_state = self.rules_engine.get_valid_moves(
            state,
            state.current_player,
        )
        if not valid_moves_state:
            return
        
        node.untried_moves = list(valid_moves_state)
        node.policy_map = {}

        total_prob = 0.0
        for move in valid_moves_state:
            if use_hex_nn and self.hex_encoder is not None:
                idx = self.hex_encoder.encode_move(move, state.board)
            elif self.neural_net is not None:
                idx = self.neural_net.encode_move(move, state.board)
            else:
                idx = INVALID_MOVE_INDEX

            if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policy):
                prob = float(cast(Any, policy)[idx])
                node.policy_map[str(move)] = prob
                total_prob += prob

        if total_prob > 0:
            for move_key in node.policy_map:
                node.policy_map[move_key] /= total_prob
        else:
            uniform = 1.0 / len(valid_moves_state)
            for move in valid_moves_state:
                node.policy_map[str(move)] = uniform

    def _heuristic_rollout_legacy(self, state: GameState) -> float:
        """Perform heuristic-guided rollout simulation."""
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

            weights = self._compute_move_weights(moves, rollout_state)
            selected_move = self.rng.choices(moves, weights=weights, k=1)[0]
            rollout_state = self.rules_engine.apply_move(
                rollout_state,
                selected_move,
            )

        return self.evaluate_position(rollout_state)

    def _compute_move_weights(
        self, moves: List[Move], state: GameState
    ) -> List[float]:
        """Compute weights for move selection in rollout."""
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
                to_key = m.to.to_key()
                if to_key in state.board.stacks:
                    w = 5.0
                else:
                    w = 2.0
            elif m.type == "place_ring":
                w = 1.5
            weights.append(w)
        return weights

    def _select_best_move_legacy(
        self, root: MCTSNode, valid_moves: List[Move]
    ) -> Tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Select best move from legacy tree based on visit counts."""
        if root.children:
            total_visits = sum(c.visits for c in root.children)
            policy: Dict[str, float] = {}
            if total_visits > 0:
                for child in root.children:
                    policy[str(child.move)] = child.visits / total_visits
            else:
                uniform = 1.0 / len(root.children)
                for child in root.children:
                    policy[str(child.move)] = uniform
            
            best_child = max(root.children, key=lambda c: c.visits)
            selected = best_child.move
            
            self.last_root = best_child
            self.last_root.parent = None
        else:
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0)
                for m in valid_moves
            }

        self.move_count += 1
        return selected, policy

    # =========================================================================
    # Incremental Search Methods (Make/Unmake Pattern)
    # =========================================================================

    def _search_incremental(
        self,
        game_state: GameState,
        valid_moves: List[Move],
    ) -> tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Incremental MCTS search using make/unmake on MutableGameState.
        
        This provides significant speedup by avoiding object allocation
        overhead during tree search. State is modified in-place and restored
        using MoveUndo tokens.
        """
        # MCTS parameters - time limit based on difficulty
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            time_limit = 1.0 + (self.config.difficulty * 0.5)

        # Create mutable state once for the entire search
        mutable_state = MutableGameState.from_immutable(game_state)

        # Tree Reuse: Check if we have a subtree for the current state
        root: Optional[MCTSNodeLite] = None
        if self.last_root_lite is not None:
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                for child in self.last_root_lite.children:
                    if child.move is not None and (
                        child.move.type == last_move.type
                        and child.move.to.x == last_move.to.x
                        and child.move.to.y == last_move.to.y
                    ):
                        root = child
                        root.parent = None
                        break

        if root is None:
            root = MCTSNodeLite()
            root.untried_moves = list(valid_moves)

        end_time = time.time() + time_limit
        default_batch_size = 8
        node_count = 1

        # MCTS implementation with PUCT using make/unmake
        while time.time() < end_time:
            if self.enable_dynamic_batching and self.dynamic_sizer is not None:
                batch_size = self.dynamic_sizer.get_optimal_batch_size(
                    node_count
                )
            else:
                batch_size = default_batch_size
            
            # Collect leaves for batch evaluation
            leaves: List[Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]] = []
            
            for _ in range(batch_size):
                # Selection phase with make_move tracking
                node, path_undos, played_moves = self._select_with_mutable(
                    root, mutable_state
                )
                
                # Expansion phase
                if node.untried_moves:
                    m = cast(Move, self.get_random_element(node.untried_moves))
                    undo = mutable_state.make_move(m)
                    path_undos.append(undo)

                    prior = node.policy_map.get(str(m))
                    child = node.add_child(m, prior=prior)
                    node = child
                    node_count += 1
                    played_moves.append(m)

                leaves.append((node, path_undos, played_moves))

                # Restore state for next iteration in batch
                for undo in reversed(path_undos):
                    mutable_state.unmake_move(undo)

                if time.time() >= end_time:
                    break

            if not leaves:
                break

            # Evaluation Phase - replay paths and evaluate
            self._evaluate_leaves_incremental(
                leaves, mutable_state, root
            )
        
        # Record memory sample if dynamic batching is enabled
        if self.enable_dynamic_batching and self.dynamic_sizer is not None:
            self.dynamic_sizer.record_memory_sample(node_count)
        
        self._log_stats()
        
        # Select best move based on visits
        return self._select_best_move_incremental(root, valid_moves)

    def _select_with_mutable(
        self,
        node: MCTSNodeLite,
        mutable_state: MutableGameState,
    ) -> Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]:
        """Traverse tree using make_move, tracking path for unmake.
        
        Returns:
            Tuple of (selected node, undo tokens, played moves)
        """
        path_undos: List[MoveUndo] = []
        played_moves: List[Move] = []
        
        # Selection - traverse to leaf
        while node.is_fully_expanded() and not node.is_leaf():
            node = node.uct_select_child()
            if node.move is not None:
                undo = mutable_state.make_move(node.move)
                path_undos.append(undo)
                played_moves.append(node.move)
        
        return node, path_undos, played_moves

    def _evaluate_leaves_incremental(
        self,
        leaves: List[Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]],
        mutable_state: MutableGameState,
        root: MCTSNodeLite,
    ) -> None:
        """Evaluate leaf nodes using make/unmake pattern.
        
        For each leaf, we replay the path to reach that state,
        evaluate, then unmake to return to root.
        """
        if self.neural_net:
            # Batch evaluation - collect states
            states: List[GameState] = []
            for node, path_undos, played_moves in leaves:
                # Replay path to reach this leaf
                for undo in path_undos:
                    mutable_state.make_move(undo.move)
                
                # Convert to immutable for evaluation
                immutable = mutable_state.to_immutable()
                states.append(immutable)
                
                # Unmake to return to root
                for undo in reversed(path_undos):
                    mutable_state.unmake_move(undo)
            
            # Check transposition table
            cached_results: List[Tuple[int, float, Any]] = []
            uncached_indices: List[int] = []
            uncached_states: List[GameState] = []
            
            for i, state in enumerate(states):
                state_hash = state.zobrist_hash or 0
                cached = self.transposition_table.get(state_hash)
                if cached is not None:
                    cached_results.append((i, cached[0], cached[1]))
                else:
                    uncached_indices.append(i)
                    uncached_states.append(state)
            
            values: List[float] = [0.0] * len(states)
            policies: List[Any] = [None] * len(states)
            
            for idx, val, pol in cached_results:
                values[idx] = val
                policies[idx] = pol

            use_hex_nn = (
                self.hex_model is not None
                and self.hex_encoder is not None
                and states
                and states[0].board.type == BoardType.HEXAGONAL
            )
            
            if uncached_states:
                if use_hex_nn:
                    eval_values, eval_policies = self._evaluate_hex_batch(
                        uncached_states
                    )
                else:
                    eval_values, eval_policies = (
                        self.neural_net.evaluate_batch(uncached_states)
                    )
                
                for j, orig_idx in enumerate(uncached_indices):
                    values[orig_idx] = eval_values[j]
                    policies[orig_idx] = eval_policies[j]
                    
                    state_hash = uncached_states[j].zobrist_hash or 0
                    self.transposition_table.put(
                        state_hash,
                        (eval_values[j], eval_policies[j]),
                    )

            # Process results and backpropagate
            for i, (node, path_undos, played_moves) in enumerate(leaves):
                value = values[i]
                policy = policies[i]
                state = states[i]

                if policy is not None:
                    self._update_node_policy_lite(
                        node, state, policy, bool(use_hex_nn)
                    )

                # Backpropagation
                current_val = float(value) if value is not None else 0.0
                curr_node: Optional[MCTSNodeLite] = node

                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    current_val = -current_val
                    curr_node = curr_node.parent
        else:
            # Fallback to Heuristic Rollout with make/unmake
            for node, path_undos, played_moves in leaves:
                # Replay path to reach this leaf
                for undo in path_undos:
                    mutable_state.make_move(undo.move)
                
                # Perform rollout
                result = self._heuristic_rollout_mutable(mutable_state)
                
                # Unmake to return to root
                for undo in reversed(path_undos):
                    mutable_state.unmake_move(undo)

                if mutable_state.current_player == self.player_number:
                    val_for_leaf_player = result
                else:
                    val_for_leaf_player = -result

                current_val = val_for_leaf_player
                curr_node: Optional[MCTSNodeLite] = node
                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    current_val = -current_val
                    curr_node = curr_node.parent

    def _heuristic_rollout_mutable(
        self, mutable_state: MutableGameState
    ) -> float:
        """Perform heuristic-guided rollout simulation using make/unmake.
        
        All rollout moves are made and then unmade to preserve state.
        """
        rollout_depth = 3
        rollout_undos: List[MoveUndo] = []

        for _ in range(rollout_depth):
            if mutable_state.is_game_over():
                break

            # Get valid moves from immutable conversion
            immutable = mutable_state.to_immutable()
            moves = self.rules_engine.get_valid_moves(
                immutable,
                mutable_state.current_player,
            )
            if not moves:
                break

            weights = self._compute_move_weights_mutable(moves, mutable_state)
            selected_move = self.rng.choices(moves, weights=weights, k=1)[0]
            undo = mutable_state.make_move(selected_move)
            rollout_undos.append(undo)

        # Evaluate at rollout end
        result = self._evaluate_mutable(mutable_state)
        
        # Unmake all rollout moves
        for undo in reversed(rollout_undos):
            mutable_state.unmake_move(undo)

        return result

    def _compute_move_weights_mutable(
        self, moves: List[Move], state: MutableGameState
    ) -> List[float]:
        """Compute weights for move selection in mutable rollout."""
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
                to_key = m.to.to_key()
                if to_key in state.stacks:
                    w = 5.0
                else:
                    w = 2.0
            elif m.type == "place_ring":
                w = 1.5
            weights.append(w)
        return weights

    def _evaluate_mutable(self, state: MutableGameState) -> float:
        """Evaluate MutableGameState by converting to immutable."""
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.player_number:
                return 100000.0
            elif winner is not None:
                return -100000.0
            else:
                return 0.0
        
        immutable = state.to_immutable()
        return self.evaluate_position(immutable)

    def _update_node_policy_lite(
        self,
        node: MCTSNodeLite,
        state: GameState,
        policy: Any,
        use_hex_nn: bool,
    ) -> None:
        """Update lite node policy priors from neural network output."""
        valid_moves_state = self.rules_engine.get_valid_moves(
            state,
            state.current_player,
        )
        if not valid_moves_state:
            return
        
        node.untried_moves = list(valid_moves_state)
        node.policy_map = {}

        total_prob = 0.0
        for move in valid_moves_state:
            if use_hex_nn and self.hex_encoder is not None:
                idx = self.hex_encoder.encode_move(move, state.board)
            elif self.neural_net is not None:
                idx = self.neural_net.encode_move(move, state.board)
            else:
                idx = INVALID_MOVE_INDEX

            if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policy):
                prob = float(cast(Any, policy)[idx])
                node.policy_map[str(move)] = prob
                total_prob += prob

        if total_prob > 0:
            for move_key in node.policy_map:
                node.policy_map[move_key] /= total_prob
        else:
            uniform = 1.0 / len(valid_moves_state)
            for move in valid_moves_state:
                node.policy_map[str(move)] = uniform

    def _select_best_move_incremental(
        self, root: MCTSNodeLite, valid_moves: List[Move]
    ) -> Tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Select best move from incremental tree based on visit counts."""
        if root.children:
            total_visits = sum(c.visits for c in root.children)
            policy: Dict[str, float] = {}
            if total_visits > 0:
                for child in root.children:
                    if child.move is not None:
                        policy[str(child.move)] = child.visits / total_visits
            else:
                uniform = 1.0 / len(root.children)
                for child in root.children:
                    if child.move is not None:
                        policy[str(child.move)] = uniform
            
            best_child = max(root.children, key=lambda c: c.visits)
            selected = best_child.move
            
            self.last_root_lite = best_child
            self.last_root_lite.parent = None
        else:
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0)
                for m in valid_moves
            }

        self.move_count += 1
        return selected, policy

    # =========================================================================
    # Shared Utility Methods
    # =========================================================================

    def _log_stats(self) -> None:
        """Log transposition table and dynamic sizer stats."""
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
            
            if self.enable_dynamic_batching and self.dynamic_sizer is not None:
                ds_stats = self.dynamic_sizer.stats()
                logger.debug(
                    "MCTS dynamic batch sizer stats: batch_size=%d, "
                    "node_estimate=%d bytes, adjustments=%d",
                    ds_stats["current_batch_size"],
                    ds_stats["node_size_estimate"],
                    ds_stats["adjustment_count"],
                )
    
    def clear_tree(self) -> None:
        """Clear the MCTS tree to free memory.
        
        This should be called after move selection if memory is constrained.
        """
        if hasattr(self, "last_root"):
            self.last_root = None

"""
Descent AI implementation for RingRift
Based on "A Simple AlphaZero" (arXiv:2008.01188v4)
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import time
from enum import Enum

import numpy as np
import torch

from .base import BaseAI
from .bounded_transposition_table import BoundedTranspositionTable
from .neural_net import (
    NeuralNetAI,
    INVALID_MOVE_INDEX,
    ActionEncoderHex,
    HexNeuralNet,
)
from ..models import GameState, Move, AIConfig, BoardType
from ..utils.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    HEURISTIC = 0
    PROVEN_WIN = 1
    PROVEN_LOSS = 2
    DRAW = 3


class DescentAI(BaseAI):
    """
    AI that uses Descent Tree Search algorithm.
    Descent is a modification of Unbounded Best-First Minimax (UBFM).
    It iteratively extends the best sequence of actions to the terminal states.
    """
    
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
        
        # Transposition table to store values with bounded memory
        # Key: state_hash, Value: (value, children_values, status)
        # Entry size estimate: ~200 bytes per entry (hash + tuple of values)
        tt_limit = self.memory_config.get_transposition_table_limit_bytes()
        self.transposition_table = BoundedTranspositionTable.from_memory_limit(
            tt_limit,
            entry_size_estimate=200,
        )
        
        # Search log for Tree Learning
        # List of (features, value)
        # Only populated when collect_training_data is True to prevent
        # memory leaks in inference-only scenarios.
        self.collect_training_data: bool = False
        self.search_log: List[Tuple[Any, float]] = []

    def get_search_data(self) -> List[Tuple[Any, float]]:
        """Retrieve and clear the search log.
        
        Note: For training, set collect_training_data=True before calling
        select_move to enable search data collection.
        """
        data = self.search_log
        self.search_log = []
        return data
    
    def enable_training_data_collection(self, enabled: bool = True) -> None:
        """Enable or disable search data collection for training.
        
        When disabled (default), search_log is not populated, preventing
        memory accumulation in inference-only scenarios.
        
        Args:
            enabled: Whether to collect training data during search.
        """
        self.collect_training_data = enabled
        if not enabled:
            # Clear any existing data when disabling
            self.search_log.clear()
        
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using Descent search.
        """
        # No simulated thinking for Descent, we use the time for search.
        
        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None
            
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
            # get_random_element only returns None when the list is empty, but
            # we guard for type-checkers even though we already checked above.
            if selected is None:
                return None
            return selected
            
        # Descent parameters
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            # Default time limit based on difficulty
            # Difficulty 1: 0.1s, Difficulty 10: 2.0s
            time_limit = 0.1 + (self.config.difficulty * 0.2)
            
        end_time = time.time() + time_limit
        
        # Run Descent iterations until the deadline or until the root is
        # proven solved.
        iterations = 0
        while time.time() < end_time:
            self._descent_iteration(
                game_state,
                depth=0,
                deadline=end_time,
            )
            iterations += 1
            
            # Completion: Stop if root is solved
            state_key = self._get_state_key(game_state)
            entry = self.transposition_table.get(state_key)
            if entry is not None:
                if len(entry) == 3:
                    _, _, status = entry
                    if status in (
                        NodeStatus.PROVEN_WIN,
                        NodeStatus.PROVEN_LOSS,
                    ):
                        break
            
        # Select best move from root
        state_key = self._get_state_key(game_state)
        entry = self.transposition_table.get(state_key)
        if entry is not None:
            if len(entry) == 3:
                _, children_values, _ = entry
            else:
                _, children_values = entry
                
            if children_values:
                if game_state.current_player == self.player_number:
                    best_move_key = max(
                        children_values.items(),
                        key=lambda x: x[1][1],
                    )[0]
                else:
                    best_move_key = min(
                        children_values.items(),
                        key=lambda x: x[1][1],
                    )[0]
                return children_values[best_move_key][0]
        
        # Log transposition table stats at end of search
        if logger.isEnabledFor(logging.DEBUG):
            stats = self.transposition_table.stats()
            logger.debug(
                "Descent search completed: iterations=%d, tt_entries=%d, "
                "tt_hits=%d, tt_misses=%d, tt_evictions=%d, hit_rate=%.2f%%",
                iterations,
                stats["entries"],
                stats["hits"],
                stats["misses"],
                stats["evictions"],
                stats["hit_rate"] * 100,
            )
        
        # Fallback if something went wrong or no search happened. Use the
        # per-instance RNG for reproducible behaviour under a fixed seed.
        fallback = self.get_random_element(valid_moves)
        if fallback is None:
            return None
        return fallback

    def _descent_iteration(
        self,
        state: GameState,
        depth: int = 0,
        deadline: Optional[float] = None,
    ) -> float:
        """
        Perform one iteration of Descent search.

        Recursively selects the best child until a terminal position or
        timeout is reached, then backpropagates values to the root.
        """
        # Global time budget: if we have reached or exceeded the deadline,
        # return a heuristic value for the current node without further
        # descent or expansion.
        if deadline is not None and time.time() >= deadline:
            if state.game_status == "finished":
                return self._calculate_terminal_value(state, depth)
            return self.evaluate_position(state)

        # Check if terminal
        if state.game_status == "finished":
            return self._calculate_terminal_value(state, depth)
                
        state_key = self._get_state_key(state)
        
        # Check if state is in transposition table
        entry = self.transposition_table.get(state_key)
        if entry is not None:
            if len(entry) == 3:
                current_val, children_values, status = entry
            else:
                current_val, children_values = entry
                status = NodeStatus.HEURISTIC

            # If proven, stop searching this branch
            if status != NodeStatus.HEURISTIC:
                return current_val

            # Select best child to descend
            if not children_values:
                return current_val

            # Use value + policy tie-breaking
            # children_values stores (move, val, prob) or (move, val)
            def get_sort_key(item):
                move_key, data = item
                val = data[1]
                prob = data[2] if len(data) > 2 else 0.0
                # Primary: Value, Secondary: Policy prob
                return (val, prob)

            if state.current_player == self.player_number:
                best_move_key = max(
                    children_values.items(),
                    key=get_sort_key,
                )[0]
            else:
                # Opponent minimizes value. If values are equal we could
                # break ties by policy, but for now we simply minimise val.
                best_move_key = min(
                    children_values.items(),
                    key=lambda x: x[1][1],
                )[0]
             
            best_move = children_values[best_move_key][0]
                
            # Descend using the canonical rules engine
            next_state = self.rules_engine.apply_move(state, best_move)
            val = self._descent_iteration(
                next_state,
                depth + 1,
                deadline=deadline,
            )
            
            # Update child value
            # Preserve existing data (move, old_val, prob)
            old_data = children_values[best_move_key]
            if len(old_data) == 3:
                children_values[best_move_key] = (best_move, val, old_data[2])
            else:
                children_values[best_move_key] = (best_move, val)
            
            # Update current node value and status
            if state.current_player == self.player_number:
                new_best_val = max(v[1] for v in children_values.values())

                # Check for proven status.
                # If any child is PROVEN_WIN (1.0), we are PROVEN_WIN.
                # If ALL children are PROVEN_LOSS (-1.0), we are PROVEN_LOSS.
                # Note: 1.0/-1.0 are reserved for proven outcomes.
                if any(v[1] == 1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_WIN
                elif all(v[1] == -1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_LOSS
                else:
                    new_status = NodeStatus.HEURISTIC
            else:
                new_best_val = min(v[1] for v in children_values.values())

                # If any child is PROVEN_LOSS (-1.0), we are PROVEN_LOSS
                # (opponent wins). If ALL children are PROVEN_WIN (1.0),
                # we are PROVEN_WIN (opponent loses).
                if any(v[1] == -1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_LOSS
                elif all(v[1] == 1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_WIN
                else:
                    new_status = NodeStatus.HEURISTIC

            self.transposition_table.put(
                state_key,
                (new_best_val, children_values, new_status),
            )

            # Log update (only if collecting training data)
            if self.collect_training_data and self.neural_net:
                features, _ = self.neural_net._extract_features(state)
                self.search_log.append((features, new_best_val))
                
            return new_best_val

        else:
            # Expand node
            valid_moves = self.rules_engine.get_valid_moves(
                state,
                state.current_player,
            )
            
            if not valid_moves:
                return 0.0
                
            # Get policy if available
            move_probs: Dict[str, float] = {}
            if self.neural_net:
                try:
                    # Decide whether to use the hex-specific network based on
                    # the current board geometry.
                    use_hex_nn = (
                        self.hex_model is not None
                        and self.hex_encoder is not None
                        and state.board.type == BoardType.HEXAGONAL
                    )

                    if use_hex_nn:
                        # Single-state batch for the hex model, reusing the
                        # shared feature encoder from NeuralNetAI so that
                        # square and hex paths stay in sync.
                        feats, globs = self.neural_net._extract_features(state)
                        tensor_input = torch.FloatTensor(
                            np.array([feats])
                        )
                        globals_input = torch.FloatTensor(
                            np.array([globs])
                        )

                        with torch.no_grad():
                            _, policy_logits_tensor = self.hex_model(
                                tensor_input,
                                globals_input,
                                hex_mask=None,
                            )
                        policy_logits = policy_logits_tensor.cpu().numpy()[0]

                        valid_indices: List[int] = []
                        valid_moves_list: List[Move] = []
                        for m in valid_moves:
                            idx = self.hex_encoder.encode_move(m, state.board)
                            if idx != INVALID_MOVE_INDEX:
                                valid_indices.append(idx)
                                valid_moves_list.append(m)

                        if valid_indices:
                            logits = policy_logits[valid_indices]
                            # Stable softmax over just the legal moves.
                            logits = logits - np.max(logits)
                            exps = np.exp(logits)
                            probs = exps / np.sum(exps)

                            for m, p in zip(valid_moves_list, probs):
                                move_probs[str(m)] = float(p)
                        else:
                            # If no legal move can be represented in the hex
                            # policy head, fall back to a uniform prior.
                            uniform = 1.0 / len(valid_moves)
                            for m in valid_moves:
                                move_probs[str(m)] = uniform
                    else:
                        # Use the shared square/canonical network path. We
                        # evaluate just this single state and reuse the
                        # resulting policy logits below.
                        _, policy_batch = self.neural_net.evaluate_batch([state])
                        policy_logits = policy_batch[0]

                        valid_indices = []
                        valid_moves_list = []
                        for m in valid_moves:
                            # Encode using canonical coordinates derived from
                            # the current board geometry. Moves that fall
                            # outside the fixed 19×19 policy grid return
                            # INVALID_MOVE_INDEX and are skipped.
                            idx = self.neural_net.encode_move(m, state.board)
                            if idx != INVALID_MOVE_INDEX:
                                valid_indices.append(idx)
                                valid_moves_list.append(m)

                        if valid_indices:
                            logits = policy_logits[valid_indices]
                            # Stable softmax
                            logits = logits - np.max(logits)
                            exps = np.exp(logits)
                            probs = exps / np.sum(exps)

                            for m, p in zip(valid_moves_list, probs):
                                move_probs[str(m)] = float(p)
                        else:
                            # If all legal moves are outside the canonical
                            # policy grid (e.g. board larger than 19×19),
                            # fall back to a uniform prior over legal moves.
                            uniform = 1.0 / len(valid_moves)
                            for m in valid_moves:
                                move_probs[str(m)] = uniform
                except Exception:
                    # Fallback if NN fails entirely (e.g. missing weights)
                    pass

            # Evaluate children
            children_values = {}
            if state.current_player == self.player_number:
                best_val = float("-inf")
            else:
                best_val = float("inf")

            for move in valid_moves:
                next_state = self.rules_engine.apply_move(state, move)

                # If we are out of time, stop expanding and return the
                # current best_val so far.
                if deadline is not None and time.time() >= deadline:
                    break
                
                # Evaluate leaf
                if next_state.game_status == "finished":
                    if next_state.winner == self.player_number:
                        val = 1.0
                    elif next_state.winner is not None:
                        val = -1.0
                    else:
                        val = 0.0
                else:
                    val = self.evaluate_position(next_state)
                
                move_key = str(move)
                prob = move_probs.get(move_key, 0.0)
                children_values[move_key] = (move, val, prob)
                
                if state.current_player == self.player_number:
                    best_val = max(best_val, val)
                else:
                    best_val = min(best_val, val)
            
            # Determine status
            status = NodeStatus.HEURISTIC
            if best_val == 1.0 and state.current_player == self.player_number:
                status = NodeStatus.PROVEN_WIN
            elif (
                best_val == -1.0
                and state.current_player != self.player_number
            ):
                status = NodeStatus.PROVEN_LOSS

            self.transposition_table.put(
                state_key,
                (best_val, children_values, status),
            )

            # Log initial visit (only if collecting training data)
            if self.collect_training_data and self.neural_net:
                features, _ = self.neural_net._extract_features(state)
                self.search_log.append((features, best_val))
                
            return best_val

    def _get_state_key(self, state: GameState) -> int:
        """Generate a unique key for the game state using Zobrist hashing"""
        if state.zobrist_hash is not None:
            return state.zobrist_hash
        # Fallback if hash is missing (shouldn't happen with updated engine)
        from .zobrist import ZobristHash
        return ZobristHash().compute_initial_hash(state)
        
    def _calculate_terminal_value(self, state: GameState, depth: int) -> float:
        """Calculate terminal value with bonuses and discount"""
        base_val = 0.0
        if state.winner == self.player_number:
            base_val = 1.0
        elif state.winner is not None:
            base_val = -1.0
        else:
            # Draw
            return 0.0
            
        # Bonuses for tie-breaking metrics (Territory, Eliminated, Markers)
        # Bonuses for tie-breaking metrics (Territory, Eliminated, Markers)
        # Territory
        territory_count = 0
        for p_id in state.board.collapsed_spaces.values():
            if p_id == self.player_number:
                territory_count += 1

        # Eliminated
        eliminated_count = state.board.eliminated_rings.get(
            str(self.player_number),
            0,
        )

        # Markers
        marker_count = 0
        for m in state.board.markers.values():
            if m.player == self.player_number:
                marker_count += 1

        # Normalize bonuses to be small (max ~0.05 total). This ensures they
        # act as tie-breakers and don't override the win/loss signal.
        bonus = (
            (territory_count * 0.001)
            + (eliminated_count * 0.001)
            + (marker_count * 0.0001)
        )

        if base_val > 0:
            val = base_val + bonus
        else:
            # If losing, we still prefer having more territory/rings as
            # tie-breakers.
            val = base_val + bonus

        # Discount for depth (fewest moves to win).
        # Win fast (val > 0): val * gamma^depth decreases with depth.
        # Lose slow (val < 0): val * gamma^depth increases (towards 0)
        # with depth.
        gamma = 0.99
        discounted_val = val * (gamma ** depth)

        # Ensure we don't exceed bounds or flip sign.
        if base_val > 0:
            return max(0.001, min(1.0, discounted_val))
        elif base_val < 0:
            return max(-1.0, min(-0.001, discounted_val))
        return 0.0

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using neural net or heuristic"""
        val = 0.0
        if self.neural_net:
            val = self.neural_net.evaluate_position(game_state)
        else:
            # Heuristic fallback: simple material difference.
            my_elim = game_state.board.eliminated_rings.get(
                str(self.player_number),
                0,
            )

            opp_elim = 0
            for pid, count in game_state.board.eliminated_rings.items():
                if int(pid) != self.player_number:
                    opp_elim += count

            val = (my_elim - opp_elim) * 0.05

        # Clamp value to (-0.99, 0.99) to reserve 1.0/-1.0 for proven
        # terminal states.
        return max(-0.99, min(0.99, val))

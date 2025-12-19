"""
Improved MCTS Implementation for RingRift AI.

Provides advanced MCTS features including:
- PUCT with configurable exploration
- Progressive widening
- Virtual loss for parallelization
- Transposition tables
- Tree reuse between moves
"""

import math
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import random

from app.utils.checksum_utils import compute_string_checksum
from app.utils.optional_imports import (
    numpy as np,
    NUMPY_AVAILABLE,
    torch,
    TORCH_AVAILABLE,
)


logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 800
    cpuct: float = 1.5                    # PUCT exploration constant
    root_dirichlet_alpha: float = 0.3     # Dirichlet noise alpha
    root_noise_weight: float = 0.25       # Weight of Dirichlet noise
    virtual_loss: float = 1.0             # Virtual loss for parallel search
    pb_c_base: float = 19652              # PUCT base
    pb_c_init: float = 1.25               # PUCT init
    use_progressive_widening: bool = False
    pw_alpha: float = 0.5                 # Progressive widening alpha
    pw_beta: float = 0.5                  # Progressive widening beta
    use_transposition_table: bool = True
    tt_max_size: int = 100000             # Transposition table max size
    tree_reuse: bool = True               # Reuse tree between moves
    fpu_reduction: float = 0.25           # First play urgency reduction
    value_weight: float = 1.0             # Weight for value vs visit count


class GameState(ABC):
    """Abstract interface for game states."""

    @abstractmethod
    def get_legal_moves(self) -> List[int]:
        """Get list of legal moves."""
        pass

    @abstractmethod
    def apply_move(self, move: int) -> 'GameState':
        """Apply move and return new state."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        pass

    @abstractmethod
    def get_outcome(self, player: int) -> float:
        """Get outcome for player (-1, 0, or 1)."""
        pass

    @abstractmethod
    def current_player(self) -> int:
        """Get current player."""
        pass

    @abstractmethod
    def hash(self) -> str:
        """Get unique hash for state."""
        pass


class NeuralNetworkInterface(ABC):
    """Abstract interface for neural network."""

    @abstractmethod
    def evaluate(self, state: GameState) -> Tuple[List[float], float]:
        """
        Evaluate state with neural network.

        Returns:
            (policy, value) where policy is list of move probabilities
        """
        pass


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state_hash: str
    parent: Optional['MCTSNode'] = None
    move: Optional[int] = None
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_loss: float = 0.0
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def value(self) -> float:
        """Mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def effective_visits(self) -> float:
        """Visits including virtual loss."""
        return self.visit_count + self.virtual_loss


class TranspositionTable:
    """Hash table for storing evaluated positions."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.table: Dict[str, Tuple[List[float], float]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def get(self, state_hash: str) -> Optional[Tuple[List[float], float]]:
        """Get cached evaluation."""
        with self._lock:
            if state_hash in self.table:
                self.access_count[state_hash] += 1
                return self.table[state_hash]
            return None

    def put(self, state_hash: str, policy: List[float], value: float):
        """Store evaluation."""
        with self._lock:
            # Evict if full
            if len(self.table) >= self.max_size:
                # Remove least accessed entries
                sorted_keys = sorted(self.access_count.keys(),
                                     key=lambda k: self.access_count[k])
                for key in sorted_keys[:self.max_size // 10]:
                    del self.table[key]
                    del self.access_count[key]

            self.table[state_hash] = (policy, value)
            self.access_count[state_hash] = 1

    def clear(self):
        """Clear the table."""
        with self._lock:
            self.table.clear()
            self.access_count.clear()

    def __len__(self) -> int:
        return len(self.table)


class ImprovedMCTS:
    """
    Improved MCTS implementation with advanced features.
    """

    def __init__(
        self,
        network: NeuralNetworkInterface,
        config: MCTSConfig = None
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.root: Optional[MCTSNode] = None
        self.transposition_table = TranspositionTable(self.config.tt_max_size) \
            if self.config.use_transposition_table else None
        self._lock = threading.RLock()

    def _compute_puct(
        self,
        node: MCTSNode,
        child: MCTSNode,
        parent_visits: int,
        fpu_value: float
    ) -> float:
        """
        Compute PUCT score for a child node.

        Uses AlphaZero's variant: Q + c * P * sqrt(N) / (1 + n)
        """
        # Value estimate
        if child.visit_count == 0:
            q = fpu_value  # First play urgency
        else:
            q = child.value

        # Exploration bonus with dynamic c
        pb_c = math.log((parent_visits + self.config.pb_c_base + 1) /
                        self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= self.config.cpuct

        exploration = pb_c * child.prior * math.sqrt(parent_visits) / (1 + child.effective_visits)

        return q + exploration

    def _select_child(
        self,
        node: MCTSNode,
        state: GameState
    ) -> Tuple[int, MCTSNode]:
        """Select child with highest PUCT score."""
        parent_visits = node.effective_visits
        legal_moves = state.get_legal_moves()

        # FPU: Use parent's value minus reduction
        fpu_value = node.value - self.config.fpu_reduction

        best_score = float('-inf')
        best_move = None
        best_child = None

        for move in legal_moves:
            if move not in node.children:
                continue

            child = node.children[move]
            score = self._compute_puct(node, child, parent_visits, fpu_value)

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _expand(
        self,
        node: MCTSNode,
        state: GameState,
        policy: List[float]
    ):
        """Expand a node with policy prior."""
        legal_moves = state.get_legal_moves()

        # Apply progressive widening if enabled
        if self.config.use_progressive_widening:
            max_children = int(self.config.pw_alpha *
                               (node.visit_count ** self.config.pw_beta))
            max_children = max(1, min(max_children, len(legal_moves)))

            # Sort by policy and take top moves
            move_priors = [(m, policy[m] if m < len(policy) else 0.0)
                           for m in legal_moves]
            move_priors.sort(key=lambda x: x[1], reverse=True)
            legal_moves = [m for m, _ in move_priors[:max_children]]

        for move in legal_moves:
            if move not in node.children:
                prior = policy[move] if move < len(policy) else 0.0
                new_state = state.apply_move(move)
                child = MCTSNode(
                    state_hash=new_state.hash(),
                    parent=node,
                    move=move,
                    prior=prior
                )
                node.children[move] = child

        node.is_expanded = True

    def _add_noise(self, node: MCTSNode, legal_moves: List[int]):
        """Add Dirichlet noise to root for exploration."""
        alpha = self.config.root_dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal_moves))

        for i, move in enumerate(legal_moves):
            if move in node.children:
                child = node.children[move]
                child.prior = ((1 - self.config.root_noise_weight) * child.prior +
                               self.config.root_noise_weight * noise[i])

    def _apply_virtual_loss(self, node: MCTSNode):
        """Apply virtual loss for parallel search."""
        current = node
        while current is not None:
            current.virtual_loss += self.config.virtual_loss
            current = current.parent

    def _remove_virtual_loss(self, node: MCTSNode):
        """Remove virtual loss after backup."""
        current = node
        while current is not None:
            current.virtual_loss -= self.config.virtual_loss
            current = current.parent

    def _backup(self, node: MCTSNode, value: float, player: int):
        """Backup value through the tree."""
        current = node
        current_player = player

        while current is not None:
            current.visit_count += 1
            # Flip value for alternating players
            current.value_sum += value if current_player == player else -value
            current = current.parent
            current_player = 1 - current_player

    def _evaluate(self, state: GameState) -> Tuple[List[float], float]:
        """Evaluate state, using transposition table if available."""
        state_hash = state.hash()

        # Check transposition table
        if self.transposition_table:
            cached = self.transposition_table.get(state_hash)
            if cached is not None:
                return cached

        # Evaluate with network
        policy, value = self.network.evaluate(state)

        # Cache result
        if self.transposition_table:
            self.transposition_table.put(state_hash, policy, value)

        return policy, value

    def search(
        self,
        state: GameState,
        add_noise: bool = True
    ) -> int:
        """
        Run MCTS search and return best move.

        Args:
            state: Current game state
            add_noise: Whether to add Dirichlet noise to root

        Returns:
            Best move according to search
        """
        # Initialize or reuse root
        state_hash = state.hash()

        if self.config.tree_reuse and self.root is not None:
            # Try to find current state in existing tree
            if self.root.state_hash == state_hash:
                pass  # Already at correct root
            else:
                # Check if it's a child of previous root
                found = False
                for child in self.root.children.values():
                    if child.state_hash == state_hash:
                        self.root = child
                        self.root.parent = None
                        found = True
                        break

                if not found:
                    self.root = MCTSNode(state_hash=state_hash)
        else:
            self.root = MCTSNode(state_hash=state_hash)

        # Expand root if needed
        if not self.root.is_expanded:
            policy, _ = self._evaluate(state)
            self._expand(self.root, state, policy)

        # Add noise to root
        if add_noise and NUMPY_AVAILABLE:
            legal_moves = state.get_legal_moves()
            self._add_noise(self.root, legal_moves)

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(state)

        # Select best move
        return self._select_best_move(self.root)

    def _simulate(self, root_state: GameState):
        """Run one MCTS simulation."""
        node = self.root
        state = root_state
        path = [node]

        # Selection: traverse tree to leaf
        while node.is_expanded and not state.is_terminal():
            move, child = self._select_child(node, state)
            if child is None:
                break

            state = state.apply_move(move)
            node = child
            path.append(node)

        # Apply virtual loss
        self._apply_virtual_loss(node)

        try:
            # Evaluation and expansion
            if state.is_terminal():
                value = state.get_outcome(root_state.current_player())
            else:
                policy, value = self._evaluate(state)
                if not node.is_expanded:
                    self._expand(node, state, policy)

            # Backup
            self._backup(node, value, state.current_player())

        finally:
            # Remove virtual loss
            self._remove_virtual_loss(node)

    def _select_best_move(self, node: MCTSNode) -> int:
        """Select best move from root based on visit count."""
        best_move = None
        best_visits = -1

        for move, child in node.children.items():
            # Weighted score combining visits and value
            if self.config.value_weight < 1.0:
                score = (child.visit_count ** (1 - self.config.value_weight) *
                         (1 + child.value) ** self.config.value_weight)
            else:
                score = child.visit_count

            if score > best_visits:
                best_visits = score
                best_move = move

        return best_move

    def get_policy(self, temperature: float = 1.0) -> List[float]:
        """
        Get policy distribution from visit counts.

        Args:
            temperature: Temperature for softmax (0 = argmax)

        Returns:
            Policy distribution over moves
        """
        if self.root is None:
            return []

        visits = []
        moves = []
        for move, child in self.root.children.items():
            visits.append(child.visit_count)
            moves.append(move)

        if not visits:
            return []

        if temperature == 0:
            # Argmax
            max_idx = visits.index(max(visits))
            policy = [0.0] * len(visits)
            policy[max_idx] = 1.0
        else:
            # Softmax with temperature
            visits = np.array(visits, dtype=np.float64)
            visits = visits ** (1.0 / temperature)
            policy = visits / visits.sum()
            policy = policy.tolist()

        # Map to full action space
        full_policy = [0.0] * max(moves + [0]) + [0.0]
        for move, prob in zip(moves, policy):
            if move < len(full_policy):
                full_policy[move] = prob

        return full_policy

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search."""
        if self.root is None:
            return {}

        stats = {
            'root_visits': self.root.visit_count,
            'root_value': self.root.value,
            'num_children': len(self.root.children),
            'max_child_visits': max(c.visit_count for c in self.root.children.values()) if self.root.children else 0,
            'transposition_table_size': len(self.transposition_table) if self.transposition_table else 0
        }

        # Top moves
        top_moves = sorted(
            self.root.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True
        )[:5]

        stats['top_moves'] = [
            {
                'move': move,
                'visits': child.visit_count,
                'value': child.value,
                'prior': child.prior
            }
            for move, child in top_moves
        ]

        return stats


class ParallelMCTS:
    """
    Parallel MCTS using multiple threads.
    """

    def __init__(
        self,
        network: NeuralNetworkInterface,
        config: MCTSConfig = None,
        num_threads: int = 4
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.num_threads = num_threads
        self.root: Optional[MCTSNode] = None
        self.transposition_table = TranspositionTable(self.config.tt_max_size)
        self._lock = threading.RLock()

    def search(self, state: GameState, add_noise: bool = True) -> int:
        """Run parallel MCTS search."""
        import concurrent.futures

        # Initialize root
        state_hash = state.hash()
        self.root = MCTSNode(state_hash=state_hash)

        # Initial expansion
        policy, _ = self.network.evaluate(state)
        self._expand(self.root, state, policy)

        if add_noise and NUMPY_AVAILABLE:
            legal_moves = state.get_legal_moves()
            self._add_noise(self.root, legal_moves)

        # Run parallel simulations
        sims_per_thread = self.config.num_simulations // self.num_threads

        def run_simulations(n_sims):
            mcts = ImprovedMCTS(self.network, self.config)
            mcts.root = self.root
            mcts.transposition_table = self.transposition_table

            for _ in range(n_sims):
                with self._lock:
                    mcts._simulate(state)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(run_simulations, sims_per_thread)
                for _ in range(self.num_threads)
            ]
            concurrent.futures.wait(futures)

        # Select best move
        return self._select_best_move()

    def _expand(self, node: MCTSNode, state: GameState, policy: List[float]):
        """Expand node."""
        legal_moves = state.get_legal_moves()
        for move in legal_moves:
            prior = policy[move] if move < len(policy) else 0.0
            new_state = state.apply_move(move)
            child = MCTSNode(
                state_hash=new_state.hash(),
                parent=node,
                move=move,
                prior=prior
            )
            node.children[move] = child
        node.is_expanded = True

    def _add_noise(self, node: MCTSNode, legal_moves: List[int]):
        """Add Dirichlet noise to root."""
        alpha = self.config.root_dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal_moves))

        for i, move in enumerate(legal_moves):
            if move in node.children:
                child = node.children[move]
                child.prior = ((1 - self.config.root_noise_weight) * child.prior +
                               self.config.root_noise_weight * noise[i])

    def _select_best_move(self) -> int:
        """Select best move based on visit counts."""
        best_move = None
        best_visits = -1

        for move, child in self.root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move

        return best_move


class MCTSWithPonder:
    """
    MCTS with pondering (background search).
    """

    def __init__(
        self,
        network: NeuralNetworkInterface,
        config: MCTSConfig = None
    ):
        self.mcts = ImprovedMCTS(network, config)
        self._pondering = False
        self._ponder_thread: Optional[threading.Thread] = None
        self._stop_ponder = threading.Event()
        self._current_state: Optional[GameState] = None

    def search(self, state: GameState, add_noise: bool = True) -> int:
        """Run search, using pondering results if available."""
        self.stop_ponder()

        # If we were pondering on this position, use the results
        if self._current_state is not None and state.hash() == self._current_state.hash():
            # Continue from pondering
            pass
        else:
            # Reset
            self.mcts.root = None

        return self.mcts.search(state, add_noise)

    def start_ponder(self, state: GameState):
        """Start pondering on predicted opponent response."""
        self.stop_ponder()

        self._current_state = state
        self._stop_ponder.clear()
        self._pondering = True

        def ponder_loop():
            while not self._stop_ponder.is_set():
                self.mcts._simulate(state)

        self._ponder_thread = threading.Thread(target=ponder_loop, daemon=True)
        self._ponder_thread.start()

    def stop_ponder(self):
        """Stop pondering."""
        if self._pondering:
            self._stop_ponder.set()
            if self._ponder_thread:
                self._ponder_thread.join(timeout=1.0)
            self._pondering = False


def main():
    """Demonstrate improved MCTS."""
    if not NUMPY_AVAILABLE:
        print("NumPy not available for demonstration")
        return

    # Create a simple dummy game state and network
    class DummyState(GameState):
        def __init__(self, board=None, player=0, move_history=None):
            self.board = board or [0] * 64
            self._player = player
            self.move_history = move_history or []

        def get_legal_moves(self) -> List[int]:
            return [i for i in range(64) if self.board[i] == 0]

        def apply_move(self, move: int) -> 'DummyState':
            new_board = self.board.copy()
            new_board[move] = self._player + 1
            return DummyState(
                new_board,
                1 - self._player,
                self.move_history + [move]
            )

        def is_terminal(self) -> bool:
            return len(self.get_legal_moves()) == 0 or len(self.move_history) > 50

        def get_outcome(self, player: int) -> float:
            return 0.0  # Draw

        def current_player(self) -> int:
            return self._player

        def hash(self) -> str:
            return compute_string_checksum(str(self.board), algorithm="md5")

    class DummyNetwork(NeuralNetworkInterface):
        def evaluate(self, state: GameState) -> Tuple[List[float], float]:
            # Random policy and value
            legal_moves = state.get_legal_moves()
            policy = [0.0] * 64
            for m in legal_moves:
                policy[m] = 1.0 / len(legal_moves)
            value = random.uniform(-0.1, 0.1)
            return policy, value

    print("=== Improved MCTS Demo ===\n")

    # Create MCTS
    network = DummyNetwork()
    config = MCTSConfig(
        num_simulations=400,
        cpuct=1.5,
        use_transposition_table=True,
        tree_reuse=True
    )
    mcts = ImprovedMCTS(network, config)

    # Run search
    state = DummyState()
    print("Running MCTS search...")

    start = time.perf_counter()
    best_move = mcts.search(state)
    elapsed = time.perf_counter() - start

    print(f"Best move: {best_move}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Simulations/sec: {config.num_simulations / elapsed:.1f}")

    # Get statistics
    stats = mcts.get_search_statistics()
    print(f"\nSearch Statistics:")
    print(f"  Root visits: {stats['root_visits']}")
    print(f"  Root value: {stats['root_value']:.4f}")
    print(f"  Transposition table size: {stats['transposition_table_size']}")

    print(f"\nTop moves:")
    for mv in stats['top_moves'][:3]:
        print(f"  Move {mv['move']}: visits={mv['visits']}, value={mv['value']:.3f}, prior={mv['prior']:.3f}")

    # Get policy
    policy = mcts.get_policy(temperature=1.0)
    print(f"\nPolicy (top 5):")
    sorted_policy = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:5]
    for move, prob in sorted_policy:
        print(f"  Move {move}: {prob:.3f}")

    # Test tree reuse
    print("\n=== Tree Reuse Test ===")
    new_state = state.apply_move(best_move)
    new_state = new_state.apply_move(random.choice(new_state.get_legal_moves()))

    start = time.perf_counter()
    best_move2 = mcts.search(new_state)
    elapsed2 = time.perf_counter() - start

    print(f"Second search time: {elapsed2:.3f}s")
    print(f"Tree reuse saved: {elapsed - elapsed2:.3f}s")


if __name__ == "__main__":
    main()

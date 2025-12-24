"""GPU-accelerated Gumbel MCTS tree using tensors.

This module implements a Structure-of-Arrays (SoA) representation of the MCTS tree
for GPU-accelerated search. Instead of Python objects with dictionaries, we use
flat tensors that can be efficiently processed on GPU.

Key architectural patterns (from research):
1. **Object Pool Pattern**: All tree nodes pre-allocated as flat tensor arrays
2. **Structure of Arrays (SoA)**: Separate tensors for each attribute (visit counts, values, etc.)
3. **Precomputed Simulation Orchestration**: Sequential Halving budget pre-computed
4. **Lock-free Design**: No atomic operations, uses reduction patterns

Performance optimizations:
- Replace mask indexing with direct int indexing (1250% faster)
- Pre-compute indices rather than masks
- Use torch.no_grad() throughout (no gradients needed for search)

References:
- DeepMind MCTX: https://github.com/google-deepmind/mctx
- MCTS-NC: https://github.com/pklesk/mcts_numba_cuda
- AlphaZero.jl: 13x GPU speedup achieved

December 2025: Initial implementation for GPU Gumbel MCTS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..models import GameState, Move
    from .gpu_batch_state import BatchGameState

logger = logging.getLogger(__name__)


@dataclass
class TensorGumbelTree:
    """GPU-resident Gumbel MCTS tree as Structure of Arrays.

    This class stores all tree nodes as pre-allocated tensors, enabling
    efficient GPU operations for MCTS search.

    The tree is organized as:
    - batch_size: Number of parallel trees (usually 1 for single game search)
    - max_nodes: Maximum number of nodes per tree (pre-allocated)
    - num_actions: Maximum number of actions per node

    All tensors are stored on GPU and have shape (batch_size, max_nodes, ...).

    Attributes:
        device: GPU device for tensors
        batch_size: Number of parallel trees
        max_nodes: Maximum nodes per tree
        num_actions: Action space size
        board_height: Board height for move encoding
        board_width: Board width for move encoding

        # Tree structure tensors
        visit_counts: (batch_size, max_nodes, num_actions) - visits per child
        total_values: (batch_size, max_nodes, num_actions) - accumulated values
        prior_logits: (batch_size, max_nodes, num_actions) - policy logits
        parent_idx: (batch_size, max_nodes) - parent node index (-1 for root)
        children_idx: (batch_size, max_nodes, num_actions) - child node indices
        node_depth: (batch_size, max_nodes) - depth in tree
        is_expanded: (batch_size, max_nodes) - whether node is expanded

        # Gumbel-specific
        gumbel_noise: (batch_size, num_actions) - root Gumbel noise
        remaining_mask: (batch_size, num_actions) - actions still in consideration

        # Sequential Halving state
        action_values: (batch_size, num_actions) - accumulated values per action
        action_visits: (batch_size, num_actions) - visit counts per action
        current_phase: int - current Sequential Halving phase

        # Node allocation
        next_node_idx: (batch_size,) - next free node index per tree
    """

    device: torch.device
    batch_size: int
    max_nodes: int
    num_actions: int
    board_height: int
    board_width: int

    # Tree structure tensors (initialized in __post_init__)
    visit_counts: torch.Tensor = field(init=False)
    total_values: torch.Tensor = field(init=False)
    prior_logits: torch.Tensor = field(init=False)
    parent_idx: torch.Tensor = field(init=False)
    children_idx: torch.Tensor = field(init=False)
    node_depth: torch.Tensor = field(init=False)
    is_expanded: torch.Tensor = field(init=False)

    # Gumbel-specific
    gumbel_noise: torch.Tensor = field(init=False)
    remaining_mask: torch.Tensor = field(init=False)

    # Sequential Halving state
    action_values: torch.Tensor = field(init=False)
    action_visits: torch.Tensor = field(init=False)
    current_phase: int = field(init=False)

    # Node allocation
    next_node_idx: torch.Tensor = field(init=False)

    def __post_init__(self):
        """Initialize all tensors on GPU."""
        with torch.no_grad():
            # Tree structure tensors
            self.visit_counts = torch.zeros(
                self.batch_size, self.max_nodes, self.num_actions,
                dtype=torch.int32, device=self.device
            )
            self.total_values = torch.zeros(
                self.batch_size, self.max_nodes, self.num_actions,
                dtype=torch.float32, device=self.device
            )
            self.prior_logits = torch.zeros(
                self.batch_size, self.max_nodes, self.num_actions,
                dtype=torch.float32, device=self.device
            )
            self.parent_idx = torch.full(
                (self.batch_size, self.max_nodes),
                fill_value=-1, dtype=torch.int32, device=self.device
            )
            self.children_idx = torch.full(
                (self.batch_size, self.max_nodes, self.num_actions),
                fill_value=-1, dtype=torch.int32, device=self.device
            )
            self.node_depth = torch.zeros(
                self.batch_size, self.max_nodes,
                dtype=torch.int32, device=self.device
            )
            self.is_expanded = torch.zeros(
                self.batch_size, self.max_nodes,
                dtype=torch.bool, device=self.device
            )

            # Gumbel-specific (for root node only)
            self.gumbel_noise = torch.zeros(
                self.batch_size, self.num_actions,
                dtype=torch.float32, device=self.device
            )
            self.remaining_mask = torch.ones(
                self.batch_size, self.num_actions,
                dtype=torch.bool, device=self.device
            )

            # Sequential Halving state
            self.action_values = torch.zeros(
                self.batch_size, self.num_actions,
                dtype=torch.float32, device=self.device
            )
            self.action_visits = torch.zeros(
                self.batch_size, self.num_actions,
                dtype=torch.int32, device=self.device
            )
            self.current_phase = 0

            # Node allocation (start at 1, 0 is root)
            self.next_node_idx = torch.ones(
                self.batch_size, dtype=torch.int32, device=self.device
            )

    @classmethod
    def create(
        cls,
        batch_size: int = 1,
        max_nodes: int = 1024,
        num_actions: int = 256,  # max legal moves in RingRift
        board_height: int = 8,
        board_width: int = 8,
        device: torch.device | str | None = None,
    ) -> "TensorGumbelTree":
        """Create a new tensor tree.

        Args:
            batch_size: Number of parallel trees
            max_nodes: Maximum nodes per tree
            num_actions: Maximum actions per node
            board_height: Board height for move encoding
            board_width: Board width for move encoding
            device: GPU device (defaults to cuda if available)

        Returns:
            TensorGumbelTree instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        return cls(
            device=device,
            batch_size=batch_size,
            max_nodes=max_nodes,
            num_actions=num_actions,
            board_height=board_height,
            board_width=board_width,
        )

    def reset(self) -> None:
        """Reset tree for new search."""
        with torch.no_grad():
            self.visit_counts.zero_()
            self.total_values.zero_()
            self.prior_logits.zero_()
            self.parent_idx.fill_(-1)
            self.children_idx.fill_(-1)
            self.node_depth.zero_()
            self.is_expanded.zero_()
            self.gumbel_noise.zero_()
            self.remaining_mask.fill_(True)
            self.action_values.zero_()
            self.action_visits.zero_()
            self.current_phase = 0
            self.next_node_idx.fill_(1)

    def initialize_root(
        self,
        prior_logits: torch.Tensor,
        num_sampled_actions: int = 16,
        gumbel_scale: float = 1.0,
    ) -> torch.Tensor:
        """Initialize root node with Gumbel-Top-K sampling.

        Args:
            prior_logits: (batch_size, num_actions) policy logits from NN
            num_sampled_actions: Number of actions to sample (k)
            gumbel_scale: Scale for Gumbel noise

        Returns:
            (batch_size, num_sampled_actions) indices of sampled actions
        """
        with torch.no_grad():
            batch_size = prior_logits.shape[0]
            num_actions = prior_logits.shape[1]

            # Store prior logits for root (node 0)
            self.prior_logits[:batch_size, 0, :num_actions] = prior_logits

            # Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
            uniform = torch.rand_like(prior_logits).clamp(min=1e-10, max=1.0 - 1e-10)
            self.gumbel_noise[:batch_size, :num_actions] = (
                -torch.log(-torch.log(uniform)) * gumbel_scale
            )

            # Compute perturbed logits for action ranking
            perturbed_logits = prior_logits + self.gumbel_noise[:batch_size, :num_actions]

            # Sample top-K actions
            k = min(num_sampled_actions, num_actions)
            _, top_k_indices = torch.topk(perturbed_logits, k=k, dim=1)

            # Initialize remaining mask to only include sampled actions
            self.remaining_mask.zero_()
            for b in range(batch_size):
                self.remaining_mask[b, top_k_indices[b]] = True

            # Mark root as expanded
            self.is_expanded[:batch_size, 0] = True

            return top_k_indices

    def compute_sequential_halving_budget(
        self,
        total_budget: int,
        num_actions: int,
    ) -> list[tuple[int, int]]:
        """Compute budget allocation for Sequential Halving.

        Args:
            total_budget: Total simulation budget
            num_actions: Number of actions to consider

        Returns:
            List of (num_actions, sims_per_action) for each phase
        """
        if num_actions <= 1:
            return [(1, total_budget)]

        num_phases = int(math.ceil(math.log2(num_actions)))
        phases = []

        remaining_actions = num_actions
        budget_per_phase = total_budget // num_phases

        for phase in range(num_phases):
            sims_per_action = max(1, budget_per_phase // remaining_actions)
            phases.append((remaining_actions, sims_per_action))
            remaining_actions = max(1, remaining_actions // 2)

        return phases

    def get_remaining_action_indices(self, tree_idx: int = 0) -> torch.Tensor:
        """Get indices of actions still in consideration.

        Uses direct indexing instead of mask indexing for performance.

        Args:
            tree_idx: Which tree in the batch

        Returns:
            (num_remaining,) int tensor of action indices
        """
        with torch.no_grad():
            # Direct indexing is 1250% faster than mask indexing
            return torch.nonzero(
                self.remaining_mask[tree_idx], as_tuple=True
            )[0]

    def update_action_values(
        self,
        action_indices: torch.Tensor,
        values: torch.Tensor,
        tree_idx: int = 0,
    ) -> None:
        """Update action values from simulation results.

        Args:
            action_indices: (num_actions,) action indices
            values: (num_actions, sims_per_action) or (num_actions,) values
            tree_idx: Which tree in the batch
        """
        with torch.no_grad():
            if values.dim() == 1:
                # Single value per action
                self.action_values[tree_idx, action_indices] += values
                self.action_visits[tree_idx, action_indices] += 1
            else:
                # Multiple values per action (from sims)
                self.action_values[tree_idx, action_indices] += values.sum(dim=1)
                self.action_visits[tree_idx, action_indices] += values.shape[1]

    def prune_actions(self, tree_idx: int = 0) -> None:
        """Prune bottom half of actions based on values.

        Called at end of each Sequential Halving phase.

        Args:
            tree_idx: Which tree in the batch
        """
        with torch.no_grad():
            # Get remaining action indices
            remaining_indices = self.get_remaining_action_indices(tree_idx)
            num_remaining = len(remaining_indices)

            if num_remaining <= 1:
                return

            # Compute mean values for remaining actions
            values = self.action_values[tree_idx, remaining_indices]
            visits = self.action_visits[tree_idx, remaining_indices].float()
            mean_values = values / visits.clamp(min=1)

            # Find median
            sorted_values, _ = torch.sort(mean_values)
            median_idx = num_remaining // 2
            median_value = sorted_values[median_idx]

            # Keep top half (above or equal to median)
            keep_mask = mean_values >= median_value
            prune_indices = remaining_indices[~keep_mask]

            # Update remaining mask
            self.remaining_mask[tree_idx, prune_indices] = False

    def get_best_action(self, tree_idx: int = 0) -> int:
        """Get the best action after Sequential Halving.

        Args:
            tree_idx: Which tree in the batch

        Returns:
            Index of best action
        """
        with torch.no_grad():
            # Get remaining actions
            remaining_indices = self.get_remaining_action_indices(tree_idx)

            if len(remaining_indices) == 0:
                logger.warning("No remaining actions, returning 0")
                return 0

            # Compute mean values
            values = self.action_values[tree_idx, remaining_indices]
            visits = self.action_visits[tree_idx, remaining_indices].float()
            mean_values = values / visits.clamp(min=1)

            # Add Gumbel + prior for tie-breaking (completed Q-value)
            gumbel = self.gumbel_noise[tree_idx, remaining_indices]
            prior = self.prior_logits[tree_idx, 0, remaining_indices]
            scores = mean_values + (gumbel + prior) / 10.0

            # Return action with highest score
            best_local_idx = torch.argmax(scores).item()
            return remaining_indices[best_local_idx].item()

    def get_policy_distribution(self, tree_idx: int = 0) -> torch.Tensor:
        """Get soft policy distribution from visit counts.

        Used for training targets.

        Args:
            tree_idx: Which tree in the batch

        Returns:
            (num_actions,) probability distribution
        """
        with torch.no_grad():
            visits = self.action_visits[tree_idx].float()
            total = visits.sum()
            if total > 0:
                return visits / total
            else:
                # Uniform over remaining actions
                remaining = self.remaining_mask[tree_idx].float()
                return remaining / remaining.sum()


@dataclass
class GPUGumbelMCTSConfig:
    """Configuration for GPU Gumbel MCTS search."""

    # Gumbel-Top-K parameters
    num_sampled_actions: int = 16
    gumbel_scale: float = 1.0

    # Sequential Halving
    simulation_budget: int = 800
    c_visit: float = 50.0
    c_scale: float = 1.0

    # Tree size limits
    max_nodes: int = 1024
    max_actions: int = 256

    # Rollout parameters
    max_rollout_depth: int = 10

    # Evaluation mode:
    # - "heuristic": Fast GPU heuristic only (~100ms)
    # - "nn": Neural network only (~1-2s)
    # - "hybrid": Heuristic for early phases, NN for final phase (~300ms)
    eval_mode: str = "heuristic"

    # Legacy compatibility (deprecated, use eval_mode instead)
    use_nn_rollout: bool = False  # If True, equivalent to eval_mode="nn"

    # Device
    device: str = "cuda"

    def __post_init__(self):
        """Handle legacy use_nn_rollout flag."""
        if self.use_nn_rollout and self.eval_mode == "heuristic":
            self.eval_mode = "nn"


def compute_gumbel_completed_q(
    values: torch.Tensor,
    visits: torch.Tensor,
    gumbel: torch.Tensor,
    prior: torch.Tensor,
    c_visit: float = 50.0,
) -> torch.Tensor:
    """Compute Gumbel-completed Q-values for action selection.

    From "Policy improvement by planning with Gumbel" (Danihelka et al., 2022):
    The completed Q-value mixes empirical Q with prior-based completion.

    Args:
        values: (batch, num_actions) accumulated values
        visits: (batch, num_actions) visit counts
        gumbel: (batch, num_actions) Gumbel noise
        prior: (batch, num_actions) prior policy logits
        c_visit: Mixing coefficient

    Returns:
        (batch, num_actions) completed Q-values
    """
    with torch.no_grad():
        # Empirical Q-value
        mean_q = values / visits.float().clamp(min=1)

        # Mixing coefficient based on visits
        max_visits = visits.float().max(dim=1, keepdim=True).values
        mix = c_visit / (c_visit + max_visits)

        # Prior-based value estimate (scaled logits)
        prior_value = prior / 10.0

        # Mixed value
        completed_q = (1 - mix) * mean_q + mix * prior_value

        # Add Gumbel for stochastic ranking
        return completed_q + gumbel


class GPUGumbelMCTS:
    """GPU-accelerated Gumbel MCTS search.

    This class orchestrates the full Gumbel MCTS search using GPU tensors
    and the existing gpu_parallel_games infrastructure for rollouts.

    The key innovation is treating tree branches as a batch of game states
    that can be processed in parallel on GPU.

    Example:
        searcher = GPUGumbelMCTS(config)
        best_move, policy = searcher.search(game_state, neural_net)
    """

    def __init__(self, config: GPUGumbelMCTSConfig):
        """Initialize GPU Gumbel MCTS searcher.

        Args:
            config: Configuration for the search
        """
        self.config = config
        self.device = torch.device(config.device)
        self.tree: TensorGumbelTree | None = None

    def search(
        self,
        game_state: "GameState",
        neural_net: Any,
        valid_moves: list["Move"] | None = None,
    ) -> tuple["Move", dict[str, float]]:
        """Run GPU Gumbel MCTS search from the given position.

        Args:
            game_state: Current game state
            neural_net: Neural network for policy/value evaluation
            valid_moves: Optional list of valid moves (computed if not provided)

        Returns:
            (best_move, policy_dict) tuple where policy_dict maps move keys to probabilities
        """
        from ..models import BoardType
        from ..rules.default_engine import DefaultRulesEngine
        from .gpu_batch_state import BatchGameState

        # Get valid moves if not provided
        if valid_moves is None:
            engine = DefaultRulesEngine()
            valid_moves = engine.get_valid_moves(game_state, game_state.current_player)

        if len(valid_moves) == 0:
            raise ValueError("No valid moves available")

        if len(valid_moves) == 1:
            # Only one move, no search needed
            move = valid_moves[0]
            move_key = self._move_to_key(move)
            return move, {move_key: 1.0}

        # Get board dimensions
        board_size = game_state.board.size
        num_actions = len(valid_moves)

        # Initialize tree - use full action space to handle all valid moves
        # The tree's num_actions must accommodate all valid moves
        tree_num_actions = max(num_actions, self.config.max_actions)

        self.tree = TensorGumbelTree.create(
            batch_size=1,
            max_nodes=self.config.max_nodes,
            num_actions=tree_num_actions,
            board_height=board_size,
            board_width=board_size,
            device=self.device,
        )
        self.tree.reset()

        # Get policy logits from neural network
        with torch.no_grad():
            # Evaluate root position
            policy_logits, root_value = self._evaluate_position(
                game_state, neural_net, valid_moves
            )

            # Initialize root with Gumbel-Top-K sampling
            num_sampled = min(self.config.num_sampled_actions, num_actions)
            top_k_indices = self.tree.initialize_root(
                policy_logits.unsqueeze(0),
                num_sampled_actions=num_sampled,
                gumbel_scale=self.config.gumbel_scale,
            )

            # Map indices to moves
            top_k_moves = [valid_moves[idx] for idx in top_k_indices[0].tolist()]

            # Run Sequential Halving
            best_action_idx = self._sequential_halving_gpu(
                game_state,
                top_k_moves,
                top_k_indices[0],
                neural_net,
            )

            # Get best move
            best_move = top_k_moves[best_action_idx]

            # Defensive validation: ensure move is from original valid_moves
            # This catches indexing bugs that could return wrong moves
            if best_action_idx < 0 or best_action_idx >= len(top_k_moves):
                logger.error(
                    f"GPU tree returned invalid action index {best_action_idx} "
                    f"(valid range: 0-{len(top_k_moves)-1}), falling back to first move"
                )
                best_move = valid_moves[0]

            # Validate move type matches game phase (RR-GPU-TREE-001 debugging)
            from ..models import GamePhase, MoveType
            phase = game_state.current_phase
            move_type = best_move.type

            # Phase/move compatibility check (matches _assert_phase_move_invariant)
            phase_move_valid = True
            if phase == GamePhase.LINE_PROCESSING:
                allowed = {MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_OPTION,
                          MoveType.CHOOSE_LINE_REWARD, MoveType.NO_LINE_ACTION,
                          MoveType.ELIMINATE_RINGS_FROM_STACK}
                phase_move_valid = move_type in allowed
            elif phase == GamePhase.TERRITORY_PROCESSING:
                allowed = {MoveType.CHOOSE_TERRITORY_OPTION, MoveType.PROCESS_TERRITORY_REGION,
                          MoveType.ELIMINATE_RINGS_FROM_STACK, MoveType.SKIP_TERRITORY_PROCESSING,
                          MoveType.NO_TERRITORY_ACTION}
                phase_move_valid = move_type in allowed
            elif phase == GamePhase.FORCED_ELIMINATION:
                phase_move_valid = move_type == MoveType.FORCED_ELIMINATION

            if not phase_move_valid:
                logger.error(
                    f"GPU tree phase/move mismatch: move_type={move_type.value} "
                    f"in phase={phase.value}. top_k had types: "
                    f"{[m.type.value for m in top_k_moves[:5]]}. "
                    f"valid_moves had types: {[m.type.value for m in valid_moves[:5]]}. "
                    f"Falling back to first valid_move."
                )
                # Fall back to first valid move (which should be phase-valid)
                best_move = valid_moves[0]

            # Build policy distribution
            # The policy tensor is indexed by tree action indices (from top_k_indices),
            # not by sequential valid_moves indices
            policy = self.tree.get_policy_distribution(tree_idx=0)
            policy_dict = {}
            for i, move in enumerate(valid_moves):
                move_key = self._move_to_key(move)
                # i is the index into valid_moves, which is also the tree action index
                # used when initializing the tree (top_k_indices are sampled from 0..len(valid_moves)-1)
                policy_dict[move_key] = policy[i].item() if i < len(policy) else 0.0

            return best_move, policy_dict

    def _sequential_halving_gpu(
        self,
        root_state: "GameState",
        candidate_moves: list["Move"],
        candidate_indices: torch.Tensor,
        neural_net: Any,
    ) -> int:
        """Run GPU-accelerated Sequential Halving.

        Args:
            root_state: Root game state
            candidate_moves: List of candidate moves to evaluate
            candidate_indices: Indices in the full action space (tree action indices)
            neural_net: Neural network for evaluation

        Returns:
            Index of best move in candidate_moves (local index)
        """
        from .gpu_batch_state import BatchGameState

        num_actions = len(candidate_moves)
        budget = self.config.simulation_budget
        phases = self.tree.compute_sequential_halving_budget(budget, num_actions)

        # Create mapping from tree action index -> local candidate index
        # candidate_indices[local_idx] = tree_action_idx
        tree_idx_to_local = {
            int(tree_idx): local_idx
            for local_idx, tree_idx in enumerate(candidate_indices.tolist())
        }

        for phase_idx, (num_remaining, sims_per_action) in enumerate(phases):
            # Get remaining action indices (tree indices)
            remaining_tree_indices = self.tree.get_remaining_action_indices(tree_idx=0)
            num_remaining = len(remaining_tree_indices)

            if num_remaining <= 1:
                break

            # Create batch of states for all simulations
            # Each remaining action gets sims_per_action copies
            sim_states = []
            action_to_sim_indices = {}  # tree_idx -> [sim_indices]

            for batch_idx, tree_action_idx in enumerate(remaining_tree_indices.tolist()):
                # Map tree index to local index in candidate_moves
                local_idx = tree_idx_to_local.get(tree_action_idx)
                if local_idx is None:
                    logger.warning(f"Tree action index {tree_action_idx} not in candidate mapping")
                    continue

                move = candidate_moves[local_idx]
                action_to_sim_indices[tree_action_idx] = list(range(
                    batch_idx * sims_per_action,
                    (batch_idx + 1) * sims_per_action
                ))

                # Apply move to root and create copies
                child_state = self._apply_move_cpu(root_state, move)
                for _ in range(sims_per_action):
                    sim_states.append(child_state)

            # Convert to GPU batch
            if len(sim_states) > 0:
                batch = BatchGameState.from_game_states(sim_states, device=self.device)

                # Use heuristic for all phases (fast)
                values = self._gpu_rollout(batch, neural_net, use_nn=False)

                # Aggregate values per action
                for action_idx, sim_indices in action_to_sim_indices.items():
                    action_values = values[sim_indices]
                    self.tree.action_values[0, action_idx] += action_values.sum()
                    self.tree.action_visits[0, action_idx] += len(sim_indices)

            # Prune bottom half of actions
            if phase_idx < len(phases) - 1:
                self.tree.prune_actions(tree_idx=0)

        # In hybrid mode, do a final NN evaluation of the top 2 candidates
        # to make the final decision with higher accuracy
        if self.config.eval_mode == "hybrid" and neural_net is not None:
            remaining_tree_indices = self.tree.get_remaining_action_indices(tree_idx=0)
            if len(remaining_tree_indices) >= 2:
                # Evaluate top 2 candidates with NN
                final_states = []
                final_indices = []
                for tree_idx in remaining_tree_indices[:2].tolist():
                    local_idx = tree_idx_to_local.get(tree_idx)
                    if local_idx is not None:
                        move = candidate_moves[local_idx]
                        child_state = self._apply_move_cpu(root_state, move)
                        final_states.append(child_state)
                        final_indices.append(tree_idx)

                if final_states:
                    final_batch = BatchGameState.from_game_states(
                        final_states, device=self.device
                    )
                    nn_values = self._gpu_rollout(final_batch, neural_net, use_nn=True)

                    # Add NN values with high weight to influence final decision
                    nn_weight = 10.0  # Strong weight for NN evaluation
                    for i, tree_idx in enumerate(final_indices):
                        self.tree.action_values[0, tree_idx] += nn_values[i] * nn_weight
                        self.tree.action_visits[0, tree_idx] += 1

        # Get best action (tree index) and convert to local index
        best_tree_idx = self.tree.get_best_action(tree_idx=0)
        best_local_idx = tree_idx_to_local.get(best_tree_idx, 0)
        return best_local_idx

    def _gpu_rollout(
        self,
        batch_state: "BatchGameState",
        neural_net: Any,
        use_nn: bool = False,
    ) -> torch.Tensor:
        """Run GPU rollouts and evaluate positions.

        Supports three evaluation modes via config.eval_mode:
        1. "heuristic": Fast GPU heuristic only (~100ms)
        2. "nn": Neural network only (~1-2s)
        3. "hybrid": Controlled by use_nn parameter

        Args:
            batch_state: Batch of game states to evaluate
            neural_net: Neural network (used if eval_mode="nn" or use_nn=True)
            use_nn: Override for hybrid mode - True uses NN, False uses heuristic

        Returns:
            (batch_size,) tensor of position values in [-1, 1]
        """
        with torch.no_grad():
            # Determine whether to use NN
            should_use_nn = False
            if self.config.eval_mode == "nn":
                should_use_nn = True
            elif self.config.eval_mode == "hybrid":
                should_use_nn = use_nn
            # "heuristic" mode: always use heuristic

            if should_use_nn and neural_net is not None:
                return self._nn_rollout(batch_state, neural_net)

            return self._heuristic_rollout(batch_state)

    def _heuristic_rollout(
        self,
        batch_state: "BatchGameState",
    ) -> torch.Tensor:
        """Evaluate positions using GPU heuristic.

        This is very fast (~1ms for 100 positions) but less accurate
        than neural network evaluation.

        Args:
            batch_state: Batch of game states to evaluate

        Returns:
            (batch_size,) tensor of position values in [-1, 1]
        """
        from .gpu_heuristic import evaluate_positions_batch
        from .heuristic_weights import BASE_V1_BALANCED_WEIGHTS

        scores = evaluate_positions_batch(batch_state, BASE_V1_BALANCED_WEIGHTS)

        # Get current player scores and normalize to [-1, 1]
        batch_size = batch_state.batch_size
        current_players = batch_state.current_player.long()  # (batch_size,) as long for indexing

        # Gather current player's score using advanced indexing
        batch_indices = torch.arange(batch_size, device=self.device)
        player_scores = scores[batch_indices, current_players]

        # Normalize to [-1, 1]
        values = torch.tanh(player_scores / 1000.0)

        return values

    def _nn_rollout(
        self,
        batch_state: "BatchGameState",
        neural_net: Any,
    ) -> torch.Tensor:
        """Evaluate positions using neural network.

        Uses GPU-native encoding to avoid Python object conversion overhead.
        Falls back to per-state evaluation if direct encoding fails.

        Args:
            batch_state: Batch of game states to evaluate
            neural_net: Neural network with evaluate_batch() method

        Returns:
            (batch_size,) tensor of position values in [-1, 1]
        """
        batch_size = batch_state.batch_size
        values = torch.zeros(batch_size, device=self.device)

        # Try GPU-native batch encoding (5-10x faster)
        try:
            return self._nn_rollout_gpu_native(batch_state, neural_net)
        except Exception as e:
            logger.debug(f"GPU-native NN rollout failed, using fallback: {e}")

        # Fallback: Convert BatchGameState to individual GameStates
        game_states = []
        for i in range(batch_size):
            try:
                state = batch_state.to_game_state(i)
                game_states.append(state)
            except Exception as e:
                logger.warning(f"Failed to convert batch state {i}: {e}")
                game_states.append(None)

        valid_states = []
        valid_indices = []
        for i, state in enumerate(game_states):
            if state is not None:
                valid_states.append(state)
                valid_indices.append(i)

        if not valid_states:
            return values

        try:
            nn_values, _ = neural_net.evaluate_batch(valid_states)
            for idx, val in zip(valid_indices, nn_values):
                values[idx] = float(val)
        except Exception as e:
            logger.warning(f"NN batch evaluation failed: {e}, using heuristic fallback")
            return self._heuristic_rollout(batch_state)

        return values

    def _nn_rollout_gpu_native(
        self,
        batch_state: "BatchGameState",
        neural_net: Any,
    ) -> torch.Tensor:
        """GPU-native NN rollout without Python object conversion.

        This is the optimized path that encodes BatchGameState directly
        to neural network input format, avoiding 5-10x overhead from
        creating Python GameState objects.

        Args:
            batch_state: Batch of game states to evaluate
            neural_net: Neural network model

        Returns:
            (batch_size,) tensor of position values in [-1, 1]
        """
        # Encode directly on GPU
        features, globals_vec = batch_state.encode_for_nn()

        # Features need to be expanded for history (model expects 14 * 4 = 56 channels)
        # For now, repeat current frame 4 times (no actual history in rollout)
        history_length = getattr(neural_net, 'history_length', 3)
        if history_length > 0:
            # Stack current features multiple times to fill history channels
            features = features.repeat(1, history_length + 1, 1, 1)

        # Get the model and run forward pass directly
        model = neural_net.model if hasattr(neural_net, 'model') else neural_net

        with torch.no_grad():
            # Move to model device if needed
            model_device = next(model.parameters()).device
            if features.device != model_device:
                features = features.to(model_device)
                globals_vec = globals_vec.to(model_device)

            # Run model forward pass
            value_out, _ = model(features, globals_vec)

            # Get value for current player (index 0 in multi-player output)
            # The model outputs [batch, num_players], we want current player's value
            batch_indices = torch.arange(batch_state.batch_size, device=model_device)
            player_indices = (batch_state.current_player - 1).clamp(0, value_out.shape[1] - 1)
            player_indices = player_indices.to(model_device)

            values = value_out[batch_indices, player_indices]

        return values.to(self.device)

    def _evaluate_position(
        self,
        game_state: "GameState",
        neural_net: Any,
        valid_moves: list["Move"],
    ) -> tuple[torch.Tensor, float]:
        """Evaluate position using neural network.

        Args:
            game_state: Game state to evaluate
            neural_net: Neural network
            valid_moves: List of valid moves

        Returns:
            (policy_logits, value) tuple
        """
        # Get policy logits for valid moves
        num_moves = len(valid_moves)
        policy_logits = torch.zeros(num_moves, device=self.device)

        # Use neural network to get policy
        try:
            if hasattr(neural_net, 'evaluate_batch'):
                values, policies = neural_net.evaluate_batch([game_state])
                value = values[0] if values is not None else 0.0

                # Map policy to valid moves
                if policies is not None:
                    # Ensure policies are float32 to avoid dtype mismatch
                    policy_tensor = policies[0]
                    if hasattr(policy_tensor, 'float'):
                        policy_tensor = policy_tensor.float()
                    for i, move in enumerate(valid_moves):
                        move_idx = neural_net.encode_move(move, game_state.board)
                        if 0 <= move_idx < len(policy_tensor):
                            policy_logits[i] = float(policy_tensor[move_idx])
            else:
                value = 0.0
        except Exception as e:
            logger.warning(f"NN evaluation failed: {e}, using uniform policy")
            value = 0.0
            policy_logits = torch.zeros(num_moves, device=self.device)

        return policy_logits, value

    def _apply_move_cpu(self, state: "GameState", move: "Move") -> "GameState":
        """Apply move to game state using CPU rules engine.

        Args:
            state: Current game state
            move: Move to apply

        Returns:
            New game state after move
        """
        from ..rules.mutable_state import MutableGameState

        mstate = MutableGameState.from_immutable(state)
        mstate.make_move(move)
        return mstate.to_immutable()

    def _move_to_key(self, move: "Move") -> str:
        """Convert move to string key for policy dict.

        Args:
            move: Move to convert

        Returns:
            String key for the move
        """
        # Use type + positions + placement_count for unique key
        # (simulated moves all share 'id=simulated')
        if hasattr(move, 'type') and hasattr(move, 'to'):
            from_str = f"{move.from_pos.x},{move.from_pos.y}" if move.from_pos else "none"
            to_str = f"{move.to.x},{move.to.y}" if move.to else "none"
            # Include placement_count for ring placement moves
            count_str = f"_{move.placement_count}" if hasattr(move, 'placement_count') and move.placement_count else ""
            return f"{move.type.value}_{from_str}_{to_str}{count_str}"
        elif hasattr(move, 'id') and move.id != 'simulated':
            return move.id
        else:
            return str(move)


@dataclass
class MultiTreeMCTSConfig:
    """Configuration for multi-tree parallel MCTS.

    This extends GPUGumbelMCTSConfig with batch-specific settings.
    """
    # Gumbel-Top-K parameters
    num_sampled_actions: int = 16
    gumbel_scale: float = 1.0

    # Sequential Halving
    simulation_budget: int = 64  # Lower budget per tree, more trees
    c_visit: float = 50.0
    c_scale: float = 1.0

    # Tree size limits
    max_nodes: int = 256  # Smaller per tree
    max_actions: int = 256

    # Rollout parameters
    max_rollout_depth: int = 10
    eval_mode: str = "heuristic"

    # Device
    device: str = "cuda"


class MultiTreeMCTS:
    """Multi-tree parallel GPU MCTS for batch game processing.

    This class runs MCTS search for multiple games simultaneously,
    achieving 5-10x additional speedup over single-tree GPU MCTS.

    The key insight is that we can grow N independent trees in parallel,
    with all simulations and evaluations batched together on GPU.

    Example:
        searcher = MultiTreeMCTS(config)
        moves, policies = searcher.search_batch(game_states, neural_net)

    Performance characteristics:
    - Single tree (GPUGumbelMCTS): ~10ms per search
    - Multi-tree (64 games): ~50ms total = 0.8ms per game effective
    """

    def __init__(self, config: MultiTreeMCTSConfig):
        """Initialize multi-tree MCTS searcher.

        Args:
            config: Configuration for the search
        """
        self.config = config
        self.device = torch.device(config.device)
        self.tree: TensorGumbelTree | None = None

    def search_batch(
        self,
        game_states: list["GameState"],
        neural_net: Any,
    ) -> tuple[list["Move"], list[dict[str, float]]]:
        """Run parallel MCTS search for multiple games.

        Args:
            game_states: List of game states to search
            neural_net: Neural network for policy/value (optional)

        Returns:
            (moves, policies) where:
            - moves: List of best moves for each game
            - policies: List of policy dicts for each game
        """
        from ..rules.default_engine import DefaultRulesEngine
        from .gpu_batch_state import BatchGameState

        if not game_states:
            return [], []

        batch_size = len(game_states)
        engine = DefaultRulesEngine()

        # Get valid moves for each game
        all_valid_moves: list[list["Move"]] = []
        for gs in game_states:
            moves = engine.get_valid_moves(gs, gs.current_player)
            all_valid_moves.append(moves)

        # Find max number of actions across all games
        max_num_actions = max(len(moves) for moves in all_valid_moves)
        tree_num_actions = max(max_num_actions, self.config.max_actions)

        # Get board size (assume all same)
        board_size = game_states[0].board.size

        # Create tree for all games
        self.tree = TensorGumbelTree.create(
            batch_size=batch_size,
            max_nodes=self.config.max_nodes,
            num_actions=tree_num_actions,
            board_height=board_size,
            board_width=board_size,
            device=self.device,
        )
        self.tree.reset()

        with torch.no_grad():
            # Batch evaluate all root positions
            policy_logits_batch = self._evaluate_positions_batch(
                game_states, all_valid_moves, neural_net, tree_num_actions
            )

            # Initialize all roots with Gumbel-Top-K sampling
            num_sampled = min(self.config.num_sampled_actions, max_num_actions)
            top_k_indices = self.tree.initialize_root(
                policy_logits_batch,
                num_sampled_actions=num_sampled,
                gumbel_scale=self.config.gumbel_scale,
            )

            # Map indices to moves for each game
            all_top_k_moves: list[list["Move"]] = []
            for b in range(batch_size):
                valid_moves = all_valid_moves[b]
                indices = top_k_indices[b].tolist()
                top_k_moves = [
                    valid_moves[idx] if idx < len(valid_moves) else valid_moves[0]
                    for idx in indices
                ]
                all_top_k_moves.append(top_k_moves)

            # Run parallel Sequential Halving across all trees
            best_action_indices = self._parallel_sequential_halving(
                game_states,
                all_top_k_moves,
                top_k_indices,
                neural_net,
            )

            # Collect results
            result_moves = []
            result_policies = []

            for b in range(batch_size):
                best_idx = best_action_indices[b]
                best_move = all_top_k_moves[b][best_idx]
                result_moves.append(best_move)

                # Build policy dict
                policy = self.tree.get_policy_distribution(tree_idx=b)
                valid_moves = all_valid_moves[b]
                policy_dict = {}
                for i, move in enumerate(valid_moves):
                    move_key = self._move_to_key(move)
                    policy_dict[move_key] = policy[i].item() if i < len(policy) else 0.0
                result_policies.append(policy_dict)

            return result_moves, result_policies

    def _evaluate_positions_batch(
        self,
        game_states: list["GameState"],
        all_valid_moves: list[list["Move"]],
        neural_net: Any,
        tree_num_actions: int,
    ) -> torch.Tensor:
        """Batch evaluate all root positions.

        Args:
            game_states: List of game states
            all_valid_moves: Valid moves for each game
            neural_net: Neural network
            tree_num_actions: Size of action dimension

        Returns:
            (batch_size, tree_num_actions) policy logits
        """
        batch_size = len(game_states)
        policy_logits = torch.zeros(batch_size, tree_num_actions, device=self.device)

        if neural_net is not None and hasattr(neural_net, 'evaluate_batch'):
            try:
                values, policies = neural_net.evaluate_batch(game_states)

                if policies is not None:
                    # Ensure policies are float32 to avoid dtype mismatch
                    if hasattr(policies, 'float'):
                        policies = policies.float()
                    for b, (gs, valid_moves) in enumerate(zip(game_states, all_valid_moves)):
                        for i, move in enumerate(valid_moves):
                            move_idx = neural_net.encode_move(move, gs.board)
                            if 0 <= move_idx < len(policies[b]):
                                policy_logits[b, i] = float(policies[b][move_idx])
            except Exception as e:
                logger.warning(f"Batch NN evaluation failed: {e}")

        return policy_logits

    def _parallel_sequential_halving(
        self,
        root_states: list["GameState"],
        all_candidate_moves: list[list["Move"]],
        top_k_indices: torch.Tensor,
        neural_net: Any,
    ) -> list[int]:
        """Run Sequential Halving in parallel across all trees.

        This is the core of Phase 3 optimization - all trees are processed
        together in each phase, with all simulations batched on GPU.

        Args:
            root_states: Root game states for each tree
            all_candidate_moves: Candidate moves for each game
            top_k_indices: (batch_size, k) tensor of sampled action indices
            neural_net: Neural network for evaluation

        Returns:
            List of best action indices (one per game)
        """
        from .gpu_batch_state import BatchGameState

        batch_size = len(root_states)
        k = top_k_indices.shape[1]  # Number of sampled actions
        tree_num_actions = self.tree.num_actions  # Full action space for tree indexing
        budget = self.config.simulation_budget
        phases = self.tree.compute_sequential_halving_budget(budget, k)

        # Create mapping from tree action index -> local candidate index for each game
        all_tree_idx_to_local: list[dict[int, int]] = []
        for b in range(batch_size):
            mapping = {
                int(tree_idx): local_idx
                for local_idx, tree_idx in enumerate(top_k_indices[b].tolist())
            }
            all_tree_idx_to_local.append(mapping)

        for phase_idx, (num_remaining, sims_per_action) in enumerate(phases):
            # Collect unique child states first (5-10% speedup from avoiding inner loop)
            unique_states: list["GameState"] = []
            unique_tree_idx: list[int] = []
            unique_action_idx: list[int] = []

            for b in range(batch_size):
                remaining_tree_indices = self.tree.get_remaining_action_indices(tree_idx=b)
                tree_idx_to_local = all_tree_idx_to_local[b]

                for tree_action_idx in remaining_tree_indices.tolist():
                    local_idx = tree_idx_to_local.get(tree_action_idx)
                    if local_idx is None:
                        continue

                    move = all_candidate_moves[b][local_idx]

                    # Apply move to get child state (done once per unique action)
                    child_state = self._apply_move_cpu(root_states[b], move)
                    unique_states.append(child_state)
                    unique_tree_idx.append(b)
                    unique_action_idx.append(tree_action_idx)

            if not unique_states:
                break

            # Expand states for sims_per_action copies using list replication
            num_unique = len(unique_states)
            all_sim_states = unique_states * sims_per_action

            # Build indices as tensors directly using repeat (avoids tuple list)
            tree_indices = torch.tensor(
                unique_tree_idx, dtype=torch.long, device=self.device
            ).repeat(sims_per_action)
            action_indices = torch.tensor(
                unique_action_idx, dtype=torch.long, device=self.device
            ).repeat(sims_per_action)

            # Convert to GPU batch and run rollouts
            batch = BatchGameState.from_game_states(all_sim_states, device=self.device)
            values = self._gpu_rollout(batch, neural_net)

            # Vectorized value aggregation using scatter_add (10-20% speedup)
            num_sim = len(all_sim_states)
            if num_sim > 0:
                # Flatten to 1D index using full tree action space
                flat_indices = tree_indices * tree_num_actions + action_indices

                # Scatter-add values to flat buffer then reshape
                flat_values = torch.zeros(
                    batch_size * tree_num_actions, dtype=values.dtype, device=self.device
                )
                flat_values.scatter_add_(0, flat_indices, values)

                # Scatter-add counts
                flat_counts = torch.zeros(
                    batch_size * tree_num_actions, dtype=torch.long, device=self.device
                )
                flat_counts.scatter_add_(
                    0, flat_indices, torch.ones(num_sim, dtype=torch.long, device=self.device)
                )

                # Update tree tensors in-place
                self.tree.action_values += flat_values.view(batch_size, tree_num_actions)
                self.tree.action_visits += flat_counts.view(batch_size, tree_num_actions)

            # Prune bottom half of actions in each tree
            if phase_idx < len(phases) - 1:
                for b in range(batch_size):
                    self.tree.prune_actions(tree_idx=b)

        # Get best action for each tree
        best_indices = []
        for b in range(batch_size):
            best_tree_idx = self.tree.get_best_action(tree_idx=b)
            best_local_idx = all_tree_idx_to_local[b].get(best_tree_idx, 0)
            best_indices.append(best_local_idx)

        return best_indices

    def _gpu_rollout(
        self,
        batch_state: "BatchGameState",
        neural_net: Any,
    ) -> torch.Tensor:
        """Run GPU rollouts and evaluate positions.

        Args:
            batch_state: Batch of game states to evaluate
            neural_net: Neural network (optional)

        Returns:
            (batch_size,) tensor of position values in [-1, 1]
        """
        with torch.no_grad():
            if self.config.eval_mode == "nn" and neural_net is not None:
                return self._nn_rollout(batch_state, neural_net)
            return self._heuristic_rollout(batch_state)

    def _heuristic_rollout(
        self,
        batch_state: "BatchGameState",
    ) -> torch.Tensor:
        """Evaluate positions using GPU heuristic.

        Args:
            batch_state: Batch of game states to evaluate

        Returns:
            (batch_size,) tensor of position values in [-1, 1]
        """
        from .gpu_heuristic import evaluate_positions_batch
        from .heuristic_weights import BASE_V1_BALANCED_WEIGHTS

        scores = evaluate_positions_batch(batch_state, BASE_V1_BALANCED_WEIGHTS)

        batch_size = batch_state.batch_size
        current_players = batch_state.current_player.long()

        batch_indices = torch.arange(batch_size, device=self.device)
        player_scores = scores[batch_indices, current_players]

        values = torch.tanh(player_scores / 1000.0)
        return values

    def _nn_rollout(
        self,
        batch_state: "BatchGameState",
        neural_net: Any,
    ) -> torch.Tensor:
        """Evaluate positions using neural network.

        Args:
            batch_state: Batch of game states
            neural_net: Neural network

        Returns:
            (batch_size,) tensor of position values
        """
        batch_size = batch_state.batch_size

        # Convert to game states for NN
        game_states = []
        for i in range(batch_size):
            try:
                state = batch_state.to_game_state(i)
                game_states.append(state)
            except Exception:
                game_states.append(None)

        valid_states = [s for s in game_states if s is not None]
        valid_indices = [i for i, s in enumerate(game_states) if s is not None]

        values = torch.zeros(batch_size, device=self.device)

        if valid_states and hasattr(neural_net, 'evaluate_batch'):
            try:
                nn_values, _ = neural_net.evaluate_batch(valid_states)
                for idx, val in zip(valid_indices, nn_values):
                    values[idx] = float(val)
            except Exception as e:
                logger.warning(f"NN batch eval failed: {e}")
                return self._heuristic_rollout(batch_state)

        return values

    def _apply_move_cpu(self, state: "GameState", move: "Move") -> "GameState":
        """Apply move using CPU rules engine.

        Args:
            state: Current game state
            move: Move to apply

        Returns:
            New game state
        """
        from ..rules.mutable_state import MutableGameState

        mstate = MutableGameState.from_immutable(state)
        mstate.make_move(move)
        return mstate.to_immutable()

    def _move_to_key(self, move: "Move") -> str:
        """Convert move to string key for policy dict."""
        if hasattr(move, 'type') and hasattr(move, 'to'):
            from_str = f"{move.from_pos.x},{move.from_pos.y}" if move.from_pos else "none"
            to_str = f"{move.to.x},{move.to.y}" if move.to else "none"
            count_str = f"_{move.placement_count}" if hasattr(move, 'placement_count') and move.placement_count else ""
            return f"{move.type.value}_{from_str}_{to_str}{count_str}"
        elif hasattr(move, 'id') and move.id != 'simulated':
            return move.id
        else:
            return str(move)

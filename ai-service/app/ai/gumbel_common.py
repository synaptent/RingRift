"""Shared data structures for Gumbel MCTS implementations.

This module provides unified data structures used across all Gumbel MCTS variants:
- gumbel_mcts_ai.py (standard single-game)
- gmo_gumbel_hybrid.py (GMO value network integration)
- multi_game_gumbel.py (multi-game parallel)
- batched_gumbel_mcts.py (batched NN evaluation)
- tensor_gumbel_tree.py (GPU tensor-based tree)

By consolidating here, we:
1. Eliminate code duplication (was 3 copies of GumbelAction)
2. Ensure consistent behavior across variants
3. Simplify maintenance and testing

Usage:
    from app.ai.gumbel_common import GumbelAction, GumbelNode, LeafEvalRequest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GameState, Move


@dataclass
class GumbelAction:
    """Represents an action with its Gumbel-perturbed value and statistics.

    Used in Gumbel AlphaZero-style MCTS for efficient action selection via
    Gumbel-Top-K sampling and Sequential Halving.

    Attributes:
        move: The game move this action represents.
        policy_logit: Raw log-probability from the policy network.
        gumbel_noise: Gumbel(0,1) noise sample for stochastic ranking.
        perturbed_value: policy_logit + gumbel_noise (for initial ranking).
        visit_count: Number of simulations through this action.
        total_value: Cumulative value from all simulations.

    Example:
        >>> action = GumbelAction(
        ...     move=some_move,
        ...     policy_logit=1.5,
        ...     gumbel_noise=0.3,
        ...     perturbed_value=1.8,
        ... )
        >>> # After 10 simulations with total value 7.5
        >>> action.visit_count = 10
        >>> action.total_value = 7.5
        >>> action.mean_value
        0.75
    """

    move: "Move"
    policy_logit: float  # Raw log-probability from NN
    gumbel_noise: float  # Gumbel(0,1) noise sample
    perturbed_value: float  # logit + gumbel (for initial ranking)
    visit_count: int = 0
    total_value: float = 0.0  # Sum of values from simulations

    @property
    def mean_value(self) -> float:
        """Mean value from simulations (Q-value).

        Returns:
            Average simulation value, or 0.0 if no simulations yet.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def completed_q(
        self,
        max_visits: int,
        c_visit: float = 50.0,
        use_simple_additive: bool = False,
    ) -> float:
        """Compute the completed Q-value for action selection.

        This accounts for visit count asymmetry by mixing the empirical
        Q-value with a prior-based completion. From the Gumbel AlphaZero paper.

        Args:
            max_visits: Maximum visit count among all actions being compared.
            c_visit: Mixing coefficient for visit completion. Higher values
                give more weight to the prior (policy_logit).
            use_simple_additive: If True, use simplified additive formula
                (mean_value + perturbed_value) for high-throughput scenarios.
                Default False uses the standard mixing formula.

        Returns:
            Completed Q-value estimate suitable for final action selection.
        """
        if use_simple_additive:
            # Simplified additive formula for high-throughput multi-game scenarios
            # Trades accuracy for speed by avoiding mixing computation
            if self.visit_count == 0:
                return self.perturbed_value
            return self.mean_value + self.perturbed_value

        if self.visit_count == 0:
            # Use prior value (normalized policy logit as proxy)
            return self.policy_logit / 10.0  # Scale to reasonable range

        # Mix empirical Q with prior based on visit ratio
        mix = c_visit / (c_visit + max_visits)
        return (1 - mix) * self.mean_value + mix * (self.policy_logit / 10.0)

    @classmethod
    def from_gumbel_score(
        cls,
        move: "Move",
        gumbel_score: float,
    ) -> "GumbelAction":
        """Create a simplified GumbelAction from a pre-computed gumbel score.

        For use cases where policy_logit and gumbel_noise are combined
        upstream (e.g., multi_game_gumbel.py).

        Args:
            move: The game move.
            gumbel_score: Pre-computed policy_logit + gumbel_noise.

        Returns:
            GumbelAction with gumbel_score stored in perturbed_value.
        """
        return cls(
            move=move,
            policy_logit=gumbel_score,  # Approximate
            gumbel_noise=0.0,
            perturbed_value=gumbel_score,
        )


@dataclass
class GumbelNode:
    """Lightweight node for Gumbel MCTS tree traversal.

    Used for tree-based search variants that maintain explicit node structure
    (as opposed to tensor-based implementations).

    Attributes:
        move: The move that led to this node (None for root).
        parent: Parent node in the tree.
        children: Child nodes keyed by move string representation.
        visit_count: Total visits through this node.
        total_value: Cumulative value from all visits.
        prior: Prior probability from policy network.
        to_move_is_root: Whether the player to move is the root player.
            Used for value sign flipping in two-player games.

    Example:
        >>> root = GumbelNode()
        >>> child = GumbelNode(move=some_move, parent=root, prior=0.3)
        >>> root.children["e2e4"] = child
    """

    move: "Move | None" = None
    parent: "GumbelNode | None" = None
    children: dict[str, "GumbelNode"] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    to_move_is_root: bool = True

    @property
    def mean_value(self) -> float:
        """Mean value (Q-value) for this node.

        Returns:
            Average value from all visits, or 0.0 if unvisited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (unexpanded).

        Returns:
            True if node has no children.
        """
        return len(self.children) == 0

    def ucb_score(self, c_puct: float = 1.5, parent_visits: int | None = None) -> float:
        """Compute UCB score for node selection.

        Uses the PUCT formula: Q + c_puct * prior * sqrt(parent_visits) / (1 + visits)

        Args:
            c_puct: Exploration constant.
            parent_visits: Parent's visit count (uses self.parent if None).

        Returns:
            UCB score for node selection.
        """
        import math

        if parent_visits is None:
            parent_visits = self.parent.visit_count if self.parent else 1

        q_value = self.mean_value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q_value + exploration


@dataclass
class LeafEvalRequest:
    """Pending leaf evaluation request for batch processing.

    Used to collect leaf nodes for batched neural network evaluation,
    enabling 5-50x speedup through GPU batching.

    Attributes:
        game_state: The game state at the leaf to evaluate.
        is_opponent_perspective: Whether evaluation needs sign flipping.
        action_idx: Index of the action this leaf corresponds to.
        simulation_idx: Which simulation within the action's allocation.
    """

    game_state: "GameState"
    is_opponent_perspective: bool
    action_idx: int
    simulation_idx: int


@dataclass
class BatchLeafEvalRequest:
    """Pending leaf evaluation for batch processing across multiple games.

    Extended version of LeafEvalRequest that includes game identification
    for batched multi-game scenarios.

    Attributes:
        game_state: The game state at the leaf to evaluate.
        is_opponent_perspective: Whether evaluation needs sign flipping.
        game_idx: Which game this request belongs to.
        action_idx: Index of the action within the game.
        simulation_idx: Which simulation within the action's allocation.
    """

    game_state: "GameState"
    is_opponent_perspective: bool
    game_idx: int
    action_idx: int
    simulation_idx: int


@dataclass
class GameSearchState:
    """State of MCTS search for a single game in multi-game scenarios.

    Tracks the progress of Sequential Halving for one game when running
    multiple games in parallel.

    Attributes:
        game_idx: Index of this game in the batch.
        game_state: Current game state being searched.
        valid_moves: List of valid moves from this state.
        actions: All candidate actions after Gumbel-Top-K sampling.
        remaining_actions: Actions still in contention after halving.
        best_move: Selected move (set when search completes).
        is_complete: Whether search has finished for this game.
    """

    game_idx: int
    game_state: "GameState"
    valid_moves: list["Move"] = field(default_factory=list)
    actions: list[GumbelAction] = field(default_factory=list)
    remaining_actions: list[GumbelAction] = field(default_factory=list)
    best_move: "Move | None" = None
    is_complete: bool = False


# Sequential Halving utilities
def compute_sequential_halving_schedule(
    num_actions: int,
    budget: int,
) -> list[tuple[int, int]]:
    """Compute the Sequential Halving phase schedule.

    Divides the simulation budget across log2(num_actions) phases,
    halving the number of candidate actions each phase.

    Args:
        num_actions: Initial number of candidate actions (k from Gumbel-Top-K).
        budget: Total simulation budget.

    Returns:
        List of (num_actions_in_phase, sims_per_action) tuples.

    Example:
        >>> compute_sequential_halving_schedule(16, 800)
        [(16, 12), (8, 25), (4, 50), (2, 100), (1, 200)]
    """
    import math

    if num_actions <= 0 or budget <= 0:
        return []

    num_phases = max(1, int(math.ceil(math.log2(num_actions))))
    budget_per_phase = budget // num_phases

    schedule = []
    remaining_actions = num_actions

    for phase in range(num_phases):
        if remaining_actions <= 0:
            break

        sims_per_action = max(1, budget_per_phase // remaining_actions)
        schedule.append((remaining_actions, sims_per_action))

        # Halve for next phase
        remaining_actions = max(1, remaining_actions // 2)

    return schedule


def select_top_k_gumbel(
    policy_logits: list[float],
    k: int,
    temperature: float = 1.0,
) -> list[int]:
    """Select top-k action indices using Gumbel-Top-K trick.

    Adds Gumbel(0,1) noise to logits and takes the top-k, which is equivalent
    to sampling k items without replacement from the softmax distribution.

    Args:
        policy_logits: Raw log-probabilities for each action.
        k: Number of actions to select.
        temperature: Temperature for softmax (higher = more exploration).

    Returns:
        Indices of the selected actions.
    """
    import numpy as np

    if len(policy_logits) <= k:
        return list(range(len(policy_logits)))

    # Apply temperature
    scaled_logits = [l / temperature for l in policy_logits]

    # Add Gumbel noise
    gumbel_noise = -np.log(-np.log(np.random.uniform(size=len(scaled_logits)) + 1e-10) + 1e-10)
    perturbed = [l + g for l, g in zip(scaled_logits, gumbel_noise)]

    # Return top-k indices
    indices = np.argsort(perturbed)[-k:][::-1]
    return indices.tolist()


# =============================================================================
# Default Constants (unified across all implementations)
# =============================================================================

# Number of actions to sample via Gumbel-Top-K
GUMBEL_DEFAULT_K = 16

# Visit completion coefficient for completed_q mixing
GUMBEL_DEFAULT_C_VISIT = 50.0

# Exploration constant for UCB selection
GUMBEL_DEFAULT_C_PUCT = 1.5


# =============================================================================
# Simulation Budget Tiers
# =============================================================================
# Different budgets serve different use cases. Use the appropriate tier:

# THROUGHPUT: Multi-tree parallel selfplay (64+ games at once)
# - Low per-game budget, high aggregate throughput
# - Used by: tensor_gumbel_tree.py MultiTreeMCTS
GUMBEL_BUDGET_THROUGHPUT = 64

# STANDARD: Single-game search with good quality/latency tradeoff
# - Balanced for human-facing games (D6-D9)
# - Used by: gumbel_mcts_ai.py default, unified_orchestrator.py
GUMBEL_BUDGET_STANDARD = 150
GUMBEL_DEFAULT_BUDGET = GUMBEL_BUDGET_STANDARD  # Backward compatibility alias

# QUALITY: High-quality search for training data generation
# - Maximum strength, higher latency acceptable
# - Used by: batched_gumbel_mcts.py, factory.py expert mode
GUMBEL_BUDGET_QUALITY = 800

# ULTIMATE: Extended search for maximum strength (D11)
# - Competition/benchmark mode
GUMBEL_BUDGET_ULTIMATE = 1600


def get_budget_for_difficulty(difficulty: int) -> int:
    """Get recommended Gumbel budget for a difficulty level.

    Args:
        difficulty: Difficulty level 1-11

    Returns:
        Recommended simulation budget
    """
    if difficulty <= 6:
        return GUMBEL_BUDGET_STANDARD
    elif difficulty <= 9:
        return GUMBEL_BUDGET_QUALITY
    else:
        return GUMBEL_BUDGET_ULTIMATE

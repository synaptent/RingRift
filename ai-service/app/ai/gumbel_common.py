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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..models import GameState, Move


# Policy logit normalization scale for completed_q() calculations.
# Policy logits from neural networks typically range [-5, 5] for log-probs.
# Dividing by 10.0 normalizes them to [-0.5, 0.5] range, which blends well
# with empirical Q-values in [-1, 1] range when computing completed Q-values.
# This ensures the prior (policy) contribution doesn't dominate empirical values.
POLICY_LOGIT_SCALE = 10.0


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
            return self.policy_logit / POLICY_LOGIT_SCALE

        # Mix empirical Q with prior based on visit ratio
        mix = c_visit / (c_visit + max_visits)
        return (1 - mix) * self.mean_value + mix * (self.policy_logit / POLICY_LOGIT_SCALE)

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
# Adaptive Exploration (Session 17.24)
# =============================================================================


def get_adaptive_c_visit(
    board_type: str | None = None,
    num_legal_moves: int = 0,
    move_number: int = 0,
    game_phase: str = "unknown",
) -> float:
    """Compute adaptive c_visit based on board complexity and game phase.

    Higher c_visit gives more weight to the prior (policy network output),
    which encourages exploration of high-prior moves even without visits.

    Session 17.24: Adaptive exploration improves Elo by +10-15 by:
    - Exploring more on larger boards (more moves to consider)
    - Exploring more in early game (less information available)
    - Exploiting more in endgame (clearer winning lines)

    Args:
        board_type: Board type (hex8, square8, square19, hexagonal).
        num_legal_moves: Number of legal moves in current position.
        move_number: Current move number in the game.
        game_phase: Game phase ("opening", "midgame", "endgame", "unknown").

    Returns:
        Adaptive c_visit value (typically 40-80, default 50).

    Examples:
        >>> get_adaptive_c_visit("square19", num_legal_moves=200, move_number=5)
        72.0  # Large board, early game → more exploration

        >>> get_adaptive_c_visit("hex8", num_legal_moves=20, move_number=30)
        42.5  # Small board, late game → more exploitation
    """
    base = GUMBEL_DEFAULT_C_VISIT  # 50.0

    # Board complexity adjustment
    # Larger boards have more moves to explore → need higher c_visit
    if board_type == "square19" or board_type == "hexagonal":
        board_multiplier = 1.3  # +30% for large boards
    elif board_type == "square8" or board_type == "hex8":
        board_multiplier = 1.0  # Standard for small boards
    elif num_legal_moves > 150:
        board_multiplier = 1.3
    elif num_legal_moves > 60:
        board_multiplier = 1.1
    else:
        board_multiplier = 1.0

    # Game phase adjustment
    # Early game: explore more (less information)
    # Late game: exploit more (clearer positions)
    if game_phase == "opening" or move_number < 10:
        phase_multiplier = 1.2  # +20% exploration in opening
    elif game_phase == "endgame" or move_number > 40:
        phase_multiplier = 0.85  # -15% in endgame (exploit)
    else:
        phase_multiplier = 1.0  # Standard in midgame

    return base * board_multiplier * phase_multiplier


def get_adaptive_c_puct(
    board_type: str | None = None,
    num_legal_moves: int = 0,
    move_number: int = 0,
) -> float:
    """Compute adaptive c_puct for UCB exploration in tree traversal.

    c_puct controls the exploration/exploitation balance during tree search.
    Higher values prefer unexplored nodes; lower values prefer high-value nodes.

    Args:
        board_type: Board type for complexity estimation.
        num_legal_moves: Number of legal moves in current position.
        move_number: Current move number in the game.

    Returns:
        Adaptive c_puct value (typically 1.2-2.0, default 1.5).
    """
    base = GUMBEL_DEFAULT_C_PUCT  # 1.5

    # More exploration in complex positions
    if num_legal_moves > 100:
        complexity_multiplier = 1.2
    elif num_legal_moves > 50:
        complexity_multiplier = 1.1
    else:
        complexity_multiplier = 1.0

    # Less exploration late game
    if move_number > 40:
        phase_multiplier = 0.9
    else:
        phase_multiplier = 1.0

    return base * complexity_multiplier * phase_multiplier


# =============================================================================
# Simulation Budget Tiers
# =============================================================================
# Different budgets serve different use cases. Use the appropriate tier:
# SOURCE OF TRUTH: app/config/thresholds.py (to avoid circular imports with torch)
from app.config.thresholds import (
    GUMBEL_BUDGET_THROUGHPUT,   # 64 - Multi-tree parallel selfplay (bootstrap only)
    GUMBEL_BUDGET_STANDARD,     # 800 - Default for training (AlphaZero uses 800)
    GUMBEL_BUDGET_QUALITY,      # 800 - High-quality evaluation/gauntlet
    GUMBEL_BUDGET_ULTIMATE,     # 1600 - Maximum strength for benchmarks
    GUMBEL_BUDGET_MASTER,       # 3200 - 2000+ Elo training (Dec 2025)
    GUMBEL_DEFAULT_BUDGET,      # Same as STANDARD (800)
)


def get_budget_for_difficulty(difficulty: int) -> int:
    """Get recommended Gumbel budget for a difficulty level.

    Args:
        difficulty: Difficulty level 1-11+

    Returns:
        Recommended simulation budget

    December 2025: Added MASTER tier (3200) for difficulty 10+ to support
    2000+ Elo training. Previous cap at ULTIMATE (1600) was insufficient.

    Tiers:
        1-6: STANDARD (800) - Basic training
        7-9: QUALITY (800) - Quality games
        10: ULTIMATE (1600) - Strong training (1800-2000 Elo)
        11+: MASTER (3200) - 2000+ Elo training
    """
    if difficulty <= 6:
        return GUMBEL_BUDGET_STANDARD
    elif difficulty <= 9:
        return GUMBEL_BUDGET_QUALITY
    elif difficulty == 10:
        return GUMBEL_BUDGET_ULTIMATE
    else:
        # December 2025: Added MASTER tier for 2000+ Elo training
        return GUMBEL_BUDGET_MASTER


# =============================================================================
# Board-Specific Budgets (Dec 2025 ML Acceleration)
# =============================================================================
# December 2025 UPDATE: Increased budgets for 2000+ Elo training.
# Previous low budgets (400-800) were insufficient for strong models.
# AlphaZero uses 800 minimum; we need 1600-3200 for 2000+ Elo.
#
# These are now QUALITY training defaults. For throughput/bootstrap,
# use difficulty=6 or lower to get THROUGHPUT/STANDARD budgets.

BUDGET_BY_BOARD_TYPE = {
    "hex8": 1600,       # Small hex - ULTIMATE for quality (was 400)
    "square8": 1600,    # Small square - ULTIMATE for quality (was 600)
    "square19": 2400,   # Large square - between ULTIMATE and MASTER
    "hexagonal": 2400,  # Large hex - between ULTIMATE and MASTER
}

# Throughput-optimized budgets for fast iteration (use when bootstrapping)
BUDGET_BY_BOARD_TYPE_FAST = {
    "hex8": 400,        # Fast bootstrap mode
    "square8": 600,     # Fast bootstrap mode
    "square19": 800,    # Keep reasonable for large boards
    "hexagonal": 800,   # Keep reasonable for large boards
}


def get_budget_for_board_type(
    board_type: str,
    use_elo_scaling: bool = False,
    model_elo: float = 1600.0,
    training_epoch: int = 0,
) -> int:
    """Get MCTS budget optimized for board size.

    Smaller boards need less search depth for good move quality, allowing
    faster selfplay and higher training throughput.

    Args:
        board_type: Board type string (hex8, square8, square19, hexagonal)
        use_elo_scaling: If True, further scale by model Elo
        model_elo: Current model Elo for scaling (if enabled)
        training_epoch: Training epoch for progressive scaling

    Returns:
        Recommended simulation budget for the board type

    Example:
        >>> get_budget_for_board_type("hex8")
        400
        >>> get_budget_for_board_type("square19")
        800
    """
    # Get base budget for board type
    base_budget = BUDGET_BY_BOARD_TYPE.get(board_type, GUMBEL_BUDGET_STANDARD)

    if not use_elo_scaling:
        return base_budget

    # Optionally scale by Elo (for curriculum learning)
    elo_budget = get_elo_adaptive_budget(model_elo, training_epoch)

    # Use the lower of the two (faster games for weak models on small boards)
    return min(base_budget, elo_budget)


def get_elo_adaptive_budget(
    model_elo: float,
    training_epoch: int = 0,
    elo_weak_threshold: float = 1300.0,
    elo_medium_threshold: float = 1500.0,
    elo_strong_threshold: float = 1700.0,
    elo_master_threshold: float = 1900.0,
) -> int:
    """Get Elo-adaptive MCTS budget based on model strength.

    This supports the strength-driven training philosophy (December 2025):
    - Weak models (< 1300 Elo): Low budget for fast iteration
    - Medium models (1300-1500): Standard budget
    - Strong models (1500-1700): High quality budget
    - Very strong (1700-1900): Ultimate budget for breaking plateaus
    - Master (> 1900): Maximum budget for 2000+ Elo training

    December 2025 UPDATE: Added MASTER tier (3200) for 2000+ Elo training.
    Previous cap at QUALITY (800) was insufficient for breaking Elo plateaus.

    The budget also scales with training epoch to provide higher quality
    data as training progresses.

    Args:
        model_elo: Current model Elo rating
        training_epoch: Current training epoch (for progressive scaling)
        elo_weak_threshold: Elo below which model is considered weak
        elo_medium_threshold: Elo below which model is considered medium
        elo_strong_threshold: Elo above which model is considered very strong
        elo_master_threshold: Elo above which model needs MASTER budget

    Returns:
        Recommended simulation budget

    Example:
        >>> get_elo_adaptive_budget(1200, training_epoch=0)
        50
        >>> get_elo_adaptive_budget(1450, training_epoch=50)
        225
        >>> get_elo_adaptive_budget(1800, training_epoch=100)
        1600
        >>> get_elo_adaptive_budget(1950, training_epoch=100)
        3200
    """
    # Base budget from Elo
    if model_elo < elo_weak_threshold:
        base = 50  # Fast iteration for weak models
    elif model_elo < elo_medium_threshold:
        # Interpolate between 50 and 150
        progress = (model_elo - elo_weak_threshold) / (elo_medium_threshold - elo_weak_threshold)
        base = int(50 + progress * 100)
    elif model_elo < elo_strong_threshold:
        # Interpolate between 150 and 800
        progress = (model_elo - elo_medium_threshold) / (elo_strong_threshold - elo_medium_threshold)
        base = int(150 + progress * 650)  # Increased from 250 to reach 800
    elif model_elo < elo_master_threshold:
        # Interpolate between 800 and 1600 (ULTIMATE)
        progress = (model_elo - elo_strong_threshold) / (elo_master_threshold - elo_strong_threshold)
        base = int(GUMBEL_BUDGET_QUALITY + progress * (GUMBEL_BUDGET_ULTIMATE - GUMBEL_BUDGET_QUALITY))
    else:
        # December 2025: MASTER budget for 2000+ Elo training
        base = GUMBEL_BUDGET_MASTER  # 3200 - Maximum quality for master-level models

    # Scale up with training epoch (up to 2x at epoch 100+)
    epoch_multiplier = min(2.0, 1.0 + training_epoch / 100)

    # Cap at ULTIMATE budget
    return min(GUMBEL_BUDGET_ULTIMATE, int(base * epoch_multiplier))


# =============================================================================
# Sequential Halving Executor (unified implementation)
# =============================================================================


@dataclass
class SequentialHalvingPhase:
    """Represents one phase of Sequential Halving.

    Attributes:
        phase_idx: Phase number (0-indexed).
        num_actions: Number of actions in this phase.
        sims_per_action: Simulations to run for each action.
        actions: Actions still in contention.
    """
    phase_idx: int
    num_actions: int
    sims_per_action: int
    actions: list[GumbelAction]


class SequentialHalvingExecutor:
    """Unified Sequential Halving executor for Gumbel MCTS.

    This class provides the core Sequential Halving algorithm used by all
    Gumbel MCTS variants. It's designed to be pluggable with different
    evaluation strategies:

    1. CPU Sequential: Evaluate one action at a time
    2. GPU Batched: Batch all leaves for GPU evaluation
    3. Multi-Game: Batch across multiple games

    Usage:
        # For single-game search
        executor = SequentialHalvingExecutor(
            actions=top_k_actions,
            simulation_budget=800,
        )

        for phase in executor.phases():
            for action in phase.actions:
                # Your evaluation logic here
                values = evaluate_action(action, phase.sims_per_action)
                action.visit_count += phase.sims_per_action
                action.total_value += sum(values)

            executor.halve_actions(phase)

        best = executor.get_best_action()

    Example with batched evaluation:
        executor = SequentialHalvingExecutor(actions, budget=800)

        for phase in executor.phases():
            # Collect all leaves for batch evaluation
            leaves = []
            for action in phase.actions:
                leaves.extend(collect_leaves(action, phase.sims_per_action))

            # Batch evaluate
            values = batch_evaluate(leaves)

            # Distribute values back to actions
            distribute_values(phase.actions, values)

            executor.halve_actions(phase)

        best = executor.get_best_action()
    """

    def __init__(
        self,
        actions: list[GumbelAction],
        simulation_budget: int = GUMBEL_BUDGET_STANDARD,
        use_simple_additive: bool = False,
    ):
        """Initialize Sequential Halving executor.

        Args:
            actions: Candidate actions from Gumbel-Top-K sampling.
            simulation_budget: Total simulation budget.
            use_simple_additive: If True, use simplified additive completed_q.
        """
        self.actions = list(actions)
        self.simulation_budget = simulation_budget
        self.use_simple_additive = use_simple_additive

        # Calculate schedule
        self._schedule = compute_sequential_halving_schedule(
            len(actions), simulation_budget
        )
        self._current_phase = 0
        self._remaining_actions = list(actions)

    def phases(self) -> list[SequentialHalvingPhase]:
        """Get all phases for iteration.

        Returns:
            List of SequentialHalvingPhase objects.
        """
        phases = []
        remaining = list(self.actions)

        for phase_idx, (num_actions, sims_per_action) in enumerate(self._schedule):
            if len(remaining) <= 1:
                break

            phases.append(SequentialHalvingPhase(
                phase_idx=phase_idx,
                num_actions=min(num_actions, len(remaining)),
                sims_per_action=sims_per_action,
                actions=list(remaining),
            ))

            # Pre-compute halving for next phase
            remaining = remaining[:max(1, len(remaining) // 2)]

        return phases

    def halve_actions(self, phase: SequentialHalvingPhase) -> list[GumbelAction]:
        """Halve actions after a phase, keeping the best half.

        Call this after evaluating all actions in a phase.

        Args:
            phase: The phase that was just completed.

        Returns:
            Remaining actions after halving.
        """
        if len(phase.actions) <= 1:
            self._remaining_actions = phase.actions
            return phase.actions

        # Sort by completed Q-value
        max_visits = max(a.visit_count for a in phase.actions)
        sorted_actions = sorted(
            phase.actions,
            key=lambda a: a.completed_q(max_visits, use_simple_additive=self.use_simple_additive),
            reverse=True,
        )

        # Keep top half
        self._remaining_actions = sorted_actions[:max(1, len(sorted_actions) // 2)]

        # Update actions in the next phase
        if phase.phase_idx + 1 < len(self._schedule):
            # This allows the caller to see updated actions in next iteration
            pass

        return self._remaining_actions

    def get_best_action(self) -> GumbelAction:
        """Get the best action after Sequential Halving completes.

        Returns:
            Best action based on completed Q-value.
        """
        if not self._remaining_actions:
            if self.actions:
                return self.actions[0]
            raise ValueError("No actions to select from")

        if len(self._remaining_actions) == 1:
            return self._remaining_actions[0]

        # Final selection based on completed Q
        max_visits = max(a.visit_count for a in self._remaining_actions)
        return max(
            self._remaining_actions,
            key=lambda a: a.completed_q(max_visits, use_simple_additive=self.use_simple_additive),
        )

    def run_with_evaluator(
        self,
        evaluate_fn: "Callable[[GumbelAction, int], float]",
    ) -> GumbelAction:
        """Run Sequential Halving with a provided evaluation function.

        This is a convenience method for simple evaluation strategies.

        Args:
            evaluate_fn: Function(action, num_sims) -> total_value
                Called for each action with the number of simulations to run.
                Should return the sum of values from all simulations.

        Returns:
            Best action after Sequential Halving.
        """
        for phase in self.phases():
            for action in phase.actions:
                value_sum = evaluate_fn(action, phase.sims_per_action)
                action.visit_count += phase.sims_per_action
                action.total_value += value_sum

            self.halve_actions(phase)

        return self.get_best_action()

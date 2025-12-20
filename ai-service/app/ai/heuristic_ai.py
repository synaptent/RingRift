"""
Heuristic AI implementation for RingRift.

This agent evaluates a flat set of legal moves using a weighted heuristic
score and selects the best candidate (with optional stochastic tie‑breaking).

Key configuration notes:

- ``use_incremental_search`` (default: True) is accepted for API consistency
  with tree‑search AIs (MinimaxAI, MCTSAI, DescentAI) but has limited impact
  here because HeuristicAI only evaluates one ply deep.
- ``training_move_sample_limit`` (when > 0) enables deterministic random
  subsampling of the legal move set to bound evaluation cost on large boards;
  sampling uses the AI's RNG so behaviour remains reproducible under a fixed
  seed.

Configurable Weight Constants (v1.1 refactor)
=============================================
As of the v1.1 refactor, previously hard‑coded penalties and bonuses have
been converted to configurable weight constants for full weight‑space
exploration during training. This includes:

- ``_evaluate_stack_control``: WEIGHT_NO_STACKS_PENALTY, WEIGHT_SINGLE_STACK_PENALTY,
  WEIGHT_STACK_DIVERSITY_BONUS
- ``_evaluate_line_potential``: WEIGHT_TWO_IN_ROW, WEIGHT_THREE_IN_ROW, WEIGHT_FOUR_IN_ROW
- ``_evaluate_line_connectivity``: WEIGHT_CONNECTED_NEIGHBOR, WEIGHT_GAP_POTENTIAL
- ``_evaluate_stack_mobility``: WEIGHT_BLOCKED_STACK_PENALTY
- ``_victory_proximity_base_for_player``: WEIGHT_VICTORY_THRESHOLD_BONUS,
  WEIGHT_RINGS_PROXIMITY_FACTOR, WEIGHT_TERRITORY_PROXIMITY_FACTOR

A zero‑weight profile now produces zero evaluations (random play), enabling
CMA‑ES, GA, and other optimisers to fully explore the fitness landscape.

Known Weight Redundancies
=========================
**WEIGHT_TWO_IN_ROW vs WEIGHT_CONNECTED_NEIGHBOR**: These weights are
mathematically redundant. Both count the same thing: adjacent marker pairs
in line directions. The calculations in ``_evaluate_line_potential`` and
``_evaluate_line_connectivity`` iterate over all markers, check each of
6/8 directions, and add a bonus when there's an adjacent friendly marker
at distance 1.

The effective contribution from adjacent pairs is::

    count × (WEIGHT_TWO_IN_ROW × WEIGHT_LINE_POTENTIAL +
             WEIGHT_CONNECTED_NEIGHBOR × WEIGHT_LINE_CONNECTIVITY)

This collapses to a single effective weight. The 4 parameters don't add
expressiveness - CMA-ES searches a higher-dimensional space with redundant
solutions.

**Why keep both?** The methods themselves are still needed because:
- ``_evaluate_line_potential`` has 3-in-row and 4-in-row (unique features)
- ``_evaluate_line_connectivity`` has WEIGHT_GAP_POTENTIAL (unique feature)

The performance impact of the redundant counting is negligible (~µs per
evaluation). The CMA-ES impact is minor (2 extra dimensions). Left in place
for now; consider consolidating in a future refactor.

**Note on WEIGHT_CONNECTED_NEIGHBOR justification**: The original rationale
("connected markers are harder to isolate and capture") is incorrect.
RingRift has no Go-style capture mechanics - markers are NOT subject to
surrounding/liberties capture. Markers can only be flipped when a stack
moves over them. Their spatial connectivity provides no defensive benefit.

Performance Optimization Flags (Environment Variables)
=======================================================
Safe optimizations (no change to play strength):

- ``RINGRIFT_USE_FAST_TERRITORY=true`` (default: true)
  NumPy-based territory detection. ~2x faster on large boards.

- ``RINGRIFT_USE_MOVE_CACHE=true`` (default: true)
  Caches valid moves by board state hash. Reduces redundant move generation.

- ``RINGRIFT_USE_PARALLEL_EVAL=true`` (default: auto)
  Parallel evaluation using ProcessPoolExecutor. ~4x faster on large boards.
  Auto-enabled for hex and 19x19 boards when move count >= PARALLEL_MIN_MOVES.
  Set ``RINGRIFT_PARALLEL_MIN_MOVES`` (default: 50) to control threshold.

Unsafe optimizations (may affect play strength):

- ``RINGRIFT_USE_MAKE_UNMAKE=true`` (default: false)
  Uses lightweight state with simplified evaluation. ~4x faster but weaker.
  Recommended for training/benchmarking only.

- ``RINGRIFT_USE_BATCH_EVAL=true`` (default: false)
  Delta-based batch evaluation. Additional speedup but significantly
  simplified evaluation. Training/benchmarking only.

- ``RINGRIFT_EARLY_TERM_THRESHOLD=50`` (default: 0/disabled)
  Stop evaluating when a move is this much better than average.
  May miss better moves. Only useful with fast evaluation paths.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..models import (
    AIConfig,
    BoardType,
    GameState,
    Move,
    MoveType,
    Player as PlayerState,
    Position,
    RingStack,
)
from ..rules.core import count_buried_rings, is_eligible_for_recovery
from ..rules.geometry import BoardGeometry
from ..rules.recovery import has_any_recovery_move
from .base import BaseAI
from .batch_eval import (
    BoardArrays,
    batch_evaluate_positions,
    get_or_update_board_arrays,
    prepare_moves_for_batch,
)
from .fast_geometry import FastGeometry
from .heuristic_weights import HEURISTIC_WEIGHT_PROFILES
from .lightweight_eval import evaluate_position_light, extract_weights_from_ai
from .lightweight_state import LightweightState, MoveUndo
from .move_cache import USE_MOVE_CACHE, cache_moves, get_cached_moves
from .numba_eval import (
    evaluate_line_potential_numba,
    prepare_marker_arrays,
)
from .swap_evaluation import SwapEvaluator, SwapWeights

# Environment flag to enable make/unmake optimization
# WARNING: When enabled, uses simplified evaluation (fewer heuristic features).
# This provides ~4x speedup but may result in weaker play.
# Recommended for training/benchmarking, not for competitive play.
USE_MAKE_UNMAKE = os.getenv('RINGRIFT_USE_MAKE_UNMAKE', 'false').lower() == 'true'

# Environment flag to enable batch evaluation (uses NumPy vectorization)
# WARNING: Uses delta-based approximation that cannot compute all heuristic features.
# This provides additional speedup but results in significantly simplified evaluation.
# Recommended only for training/benchmarking where exact play strength isn't critical.
USE_BATCH_EVAL = os.getenv('RINGRIFT_USE_BATCH_EVAL', 'false').lower() == 'true'

# Threshold for when to use batch evaluation (number of moves)
BATCH_EVAL_THRESHOLD = int(os.getenv('RINGRIFT_BATCH_EVAL_THRESHOLD', '100'))

# Early termination: stop evaluating when we find a move this much better than others
# Set to 0 to disable early termination
# WARNING: May cause AI to miss better moves. Only useful with fast evaluation paths.
EARLY_TERM_THRESHOLD = float(os.getenv('RINGRIFT_EARLY_TERM_THRESHOLD', '0'))

# Minimum moves to evaluate before considering early termination
EARLY_TERM_MIN_MOVES = int(os.getenv('RINGRIFT_EARLY_TERM_MIN_MOVES', '20'))

# Enable parallel evaluation using ProcessPoolExecutor
# Uses full heuristics but distributes apply_move + evaluate across CPU cores.
# Safe optimization that doesn't change play strength.
USE_PARALLEL_EVAL = os.getenv('RINGRIFT_USE_PARALLEL_EVAL', 'false').lower() == 'true'

# Number of workers for parallel evaluation (0 = auto-detect based on CPU count)
PARALLEL_WORKERS = int(os.getenv('RINGRIFT_PARALLEL_WORKERS', '0'))

# Minimum moves to trigger parallel evaluation (overhead not worth it for small counts)
PARALLEL_MIN_MOVES = int(os.getenv('RINGRIFT_PARALLEL_MIN_MOVES', '50'))

# Global persistent process pool for parallel evaluation
_parallel_executor: ProcessPoolExecutor | None = None


def _get_parallel_executor() -> ProcessPoolExecutor:
    """Get or create the global parallel executor."""
    global _parallel_executor
    if _parallel_executor is None:
        num_workers = PARALLEL_WORKERS if PARALLEL_WORKERS > 0 else None
        _parallel_executor = ProcessPoolExecutor(max_workers=num_workers)
    return _parallel_executor


def _shutdown_parallel_executor() -> None:
    """Shutdown the global parallel executor."""
    global _parallel_executor
    if _parallel_executor is not None:
        _parallel_executor.shutdown(wait=False)
        _parallel_executor = None


def _evaluate_moves_chunk_worker(args: tuple) -> list[tuple[int, float]]:
    """
    Worker function for parallel chunk evaluation.

    Evaluates a chunk of moves in one worker to amortize overhead.
    Returns list of (move_index, score) tuples.
    """
    chunk_moves, game_state, player_number, config_dict = args

    # Recreate AI config
    from ..models import AIConfig
    config = AIConfig(
        difficulty=config_dict.get('difficulty', 5),
        rng_seed=config_dict.get('rng_seed'),
        think_time=0,
    )

    # Create a fresh AI instance for this worker (reused for all moves in chunk)
    ai = HeuristicAI(player_number, config)

    results = []
    for move_idx, move in chunk_moves:
        # Apply move and evaluate
        next_state = ai.rules_engine.apply_move(game_state, move)

        # Handle SWAP_SIDES perspective change
        if move.type == MoveType.SWAP_SIDES:
            original_player = ai.player_number
            ai.player_number = 1 if original_player == 2 else 2
            score = ai.evaluate_position(next_state)
            score += ai.evaluate_swap_opening_bonus(game_state)
            ai.player_number = original_player
        else:
            score = ai.evaluate_position(next_state)

        results.append((move_idx, score))

    return results


class HeuristicAI(BaseAI):
    """Heuristic AI that scores and selects strategic moves.

    HeuristicAI performs single‑ply evaluation of all valid moves and
    selects the best candidate based on a weighted combination of heuristic
    features. Unlike tree‑search AIs (Minimax, MCTS, Descent), it does not
    perform deep search and therefore gains only limited benefit from the
    make/unmake pattern.

    The ``use_incremental_search`` config option is accepted for API
    consistency with tree‑search AIs but has limited practical impact on
    performance. The ``heuristic_eval_mode`` and ``heuristic_profile_id``
    fields on :class:`AIConfig` control evaluation depth and weight
    profiles respectively.
    """

    # Evaluation weights for different factors
    WEIGHT_STACK_CONTROL = 10.0
    WEIGHT_STACK_HEIGHT = 5.0
    WEIGHT_CAP_HEIGHT = 6.0  # v1.5: Summed cap height (capture power dominance)
    WEIGHT_TERRITORY = 8.0
    WEIGHT_RINGS_IN_HAND = 3.0
    WEIGHT_CENTER_CONTROL = 4.0
    WEIGHT_ADJACENCY = 2.0
    WEIGHT_OPPONENT_THREAT = 6.0
    WEIGHT_MOBILITY = 4.0
    WEIGHT_ELIMINATED_RINGS = 12.0
    WEIGHT_LINE_POTENTIAL = 7.0
    WEIGHT_VICTORY_PROXIMITY = 20.0
    WEIGHT_MARKER_COUNT = 1.5
    WEIGHT_VULNERABILITY = 8.0
    WEIGHT_OVERTAKE_POTENTIAL = 8.0
    WEIGHT_TERRITORY_CLOSURE = 10.0
    WEIGHT_LINE_CONNECTIVITY = 6.0
    WEIGHT_TERRITORY_SAFETY = 5.0
    WEIGHT_STACK_MOBILITY = 4.0
    WEIGHT_OPPONENT_VICTORY_THREAT = 6.0
    WEIGHT_FORCED_ELIMINATION_RISK = 4.0
    WEIGHT_LPS_ACTION_ADVANTAGE = 2.0
    WEIGHT_MULTI_LEADER_THREAT = 2.0

    # Penalty/bonus weights for stack diversification (previously hardcoded)
    # These enable full weight-space exploration during training
    WEIGHT_NO_STACKS_PENALTY = 50.0      # Penalty for having zero stacks
    WEIGHT_SINGLE_STACK_PENALTY = 10.0   # Penalty for only one stack (vulnerable)
    WEIGHT_STACK_DIVERSITY_BONUS = 2.0   # Bonus per additional stack beyond 1

    # Mobility penalty weights (previously hardcoded)
    WEIGHT_SAFE_MOVE_BONUS = 1.0         # Bonus per safe move available
    WEIGHT_NO_SAFE_MOVES_PENALTY = 2.0   # Penalty when no safe moves exist

    # Victory proximity threshold weights (previously hardcoded)
    WEIGHT_VICTORY_THRESHOLD_BONUS = 1000.0  # Bonus when at/near victory
    WEIGHT_RINGS_PROXIMITY_FACTOR = 50.0     # Factor for rings-based proximity
    WEIGHT_TERRITORY_PROXIMITY_FACTOR = 50.0  # Factor for territory-based proximity

    # Line potential weights (previously hardcoded)
    WEIGHT_TWO_IN_ROW = 1.0        # Bonus for 2 markers in a row
    WEIGHT_THREE_IN_ROW = 2.0      # Additional bonus for 3 in a row
    WEIGHT_FOUR_IN_ROW = 5.0       # Additional bonus for 4 in a row (almost a line)

    # Line connectivity weights (previously hardcoded)
    WEIGHT_CONNECTED_NEIGHBOR = 1.0   # Bonus for connected marker neighbor
    WEIGHT_GAP_POTENTIAL = 0.5        # Bonus for gap with connection potential

    # Stack mobility weights (previously hardcoded)
    WEIGHT_BLOCKED_STACK_PENALTY = 5.0  # Penalty for completely blocked stacks

    # v1.2: Swap (pie rule) opening evaluation weights
    # These reward P2 for swapping into advantageous P1 opening positions
    WEIGHT_SWAP_OPENING_CENTER = 15.0     # Bonus per P1 stack in center
    WEIGHT_SWAP_OPENING_ADJACENCY = 3.0   # Bonus for P1 stacks near center
    WEIGHT_SWAP_OPENING_HEIGHT = 2.0      # Bonus per stack height on P1 stacks

    # v1.3: Enhanced swap evaluation - Opening Position Classifier weights
    # These provide finer-grained control over swap decisions based on
    # position type (beyond just center/adjacency)
    WEIGHT_SWAP_CORNER_PENALTY = 8.0      # Penalty for corner positions (weak openings)
    WEIGHT_SWAP_EDGE_BONUS = 2.0          # Bonus for edge positions (moderate)
    WEIGHT_SWAP_DIAGONAL_BONUS = 6.0      # Bonus for key diagonal positions
    WEIGHT_SWAP_OPENING_STRENGTH = 20.0   # Multiplier for normalized opening strength (0-1)

    # v1.4: Training diversity - Swap decision randomness
    # Controls stochastic exploration during training to create diverse swap decisions
    WEIGHT_SWAP_EXPLORATION_TEMPERATURE = 0.0  # Temperature for swap decision noise (0 = deterministic)

    # v1.6: Recovery action evaluation weights (RR-CANON-R110–R115)
    # Recovery allows temporarily eliminated players to slide markers to form lines
    WEIGHT_RECOVERY_POTENTIAL = 6.0         # Value of having recovery available (threat potential)
    WEIGHT_RECOVERY_ELIGIBILITY = 8.0       # Bonus/penalty for recovery eligibility status
    WEIGHT_BURIED_RING_VALUE = 3.0          # Value of buried rings as recovery resource
    WEIGHT_RECOVERY_THREAT = 5.0            # Threat from opponent's recovery potential

    def __init__(self, player_number: int, config: AIConfig) -> None:
        """
        Initialise HeuristicAI with an optional heuristic weight profile.

        When ``config.heuristic_profile_id`` is set (typically to the same
        value as the canonical difficulty ``profile_id`` from the ladder,
        e.g. ``"v1-heuristic-5"``), the corresponding entry in
        HEURISTIC_WEIGHT_PROFILES is used to override the class-level weight
        constants for this instance. If no profile is found, the built-in
        defaults defined above are used unchanged to preserve current
        behaviour.

        The ``use_incremental_search`` config option is read for API
        consistency with tree-search AIs but has minimal impact on
        HeuristicAI since it only performs single-depth evaluation.

        The optional ``heuristic_eval_mode`` config field controls whether
        this instance runs the full structural heuristic suite (``"full"``)
        or a lighter subset (``"light"``, which skips Tier-2 structural/
        global features entirely). Any value other than the literal string
        ``"light"`` (including ``None``) is normalised to ``"full"`` to
        preserve existing behaviour for callers that do not opt in
        explicitly.
        """
        super().__init__(player_number, config)

        # Fast geometry module for pre-computed adjacency and board keys.
        # Provides significant speedup for evaluation by avoiding Position
        # object creation in hot loops.
        self._fast_geo: FastGeometry = FastGeometry.get_instance()

        # Read use_incremental_search for API consistency with other AIs.
        # Limited benefit for single-depth evaluation but maintains
        # consistent configuration interface across all AI implementations.
        self.use_incremental_search: bool = getattr(
            config, 'use_incremental_search', True
        )

        # Normalise heuristic evaluation mode. Only the literal string
        # "light" opts into the lightweight evaluator; everything else
        # (None, "full", unknown) is treated as "full" for backward
        # compatibility.
        mode = getattr(config, "heuristic_eval_mode", None)
        self.eval_mode: str = "light" if mode == "light" else "full"

        # Cached BoardArrays for lazy reuse across evaluations.
        # This avoids recreating NumPy arrays on each move evaluation.
        self._cached_board_arrays: BoardArrays | None = None

        self._apply_weight_profile()

        # Initialize swap evaluator with current weights
        self._swap_evaluator: SwapEvaluator | None = None

    def _apply_weight_profile(self) -> None:
        """Override evaluation weights for this instance from a profile.

        The profile id is taken from ``config.heuristic_profile_id`` when
        provided; otherwise we infer a ladder-aligned id of the form
        ``v1-heuristic-<difficulty>``. The concrete weight vectors live in
        :mod:`app.ai.heuristic_weights` and are looked up via the shared
        ``HEURISTIC_WEIGHT_PROFILES`` registry.

        This is deliberately lightweight: it simply sets attributes like
        ``WEIGHT_STACK_CONTROL`` on the instance, which shadow the class-level
        constants without changing them globally.

        After applying the profile, optional weight noise is added if
        ``config.weight_noise`` is set (0.0-1.0 range, e.g. 0.1 = 10% noise).
        This creates diverse evaluation functions for training data generation.
        """
        # Prefer an explicit profile id from AIConfig when provided.
        profile_id = getattr(self.config, "heuristic_profile_id", None)

        # If none is provided, attempt to infer from the canonical ladder
        # naming convention for heuristic difficulties (v1-heuristic-2/3/4/5).
        if not profile_id and 1 <= self.config.difficulty <= 10:
            inferred = f"v1-heuristic-{self.config.difficulty}"
            profile_id = inferred

        if profile_id:
            weights = HEURISTIC_WEIGHT_PROFILES.get(profile_id)
            if weights:
                for name, value in weights.items():
                    setattr(self, name, value)

        # Apply weight noise for training diversity
        self._apply_weight_noise()

    def _apply_weight_noise(self) -> None:
        """Apply random noise to weights for training diversity.

        When ``config.weight_noise`` is set (0.0-1.0), each WEIGHT_* attribute
        is multiplied by a random factor in the range [1-noise, 1+noise].
        This uses the per-instance RNG for reproducibility when rng_seed is set.

        Example: weight_noise=0.1 means weights vary by ±10%.
        """
        noise_level = getattr(self.config, "weight_noise", 0.0)
        if not noise_level or noise_level <= 0:
            return

        # Clamp noise to reasonable range
        noise_level = min(1.0, max(0.0, float(noise_level)))

        # Find all WEIGHT_* attributes on this instance
        weight_attrs = [
            attr for attr in dir(self)
            if attr.startswith("WEIGHT_") and isinstance(getattr(self, attr, None), (int, float))
        ]

        for attr in weight_attrs:
            original_value = getattr(self, attr)
            # Apply multiplicative noise: value * (1 + uniform(-noise, +noise))
            noise_factor = 1.0 + self.rng.uniform(-noise_level, noise_level)
            noisy_value = original_value * noise_factor
            setattr(self, attr, noisy_value)

    @property
    def swap_evaluator(self) -> SwapEvaluator:
        """Lazily create swap evaluator with current weights."""
        if self._swap_evaluator is None:
            self._swap_evaluator = SwapEvaluator(
                weights=SwapWeights.from_heuristic_ai(self),
                fast_geo=self._fast_geo,
            )
        return self._swap_evaluator

    def _victory_proximity_base_for_player(
        self,
        game_state: GameState,
        player: PlayerState,
    ) -> float:
        """Compute base victory proximity score for a player.

        Uses configurable weight constants for full training exploration.
        """
        # LPS proximity: treat a player nearing the required consecutive
        # exclusive rounds as an imminent victory threat. This must respect
        # per-game overrides (rulesOptions.lpsRoundsRequired).
        lps_player = getattr(game_state, "lps_consecutive_exclusive_player", None)
        lps_rounds = getattr(game_state, "lps_consecutive_exclusive_rounds", 0)
        if lps_player == getattr(player, "player_number", None) and isinstance(
            lps_rounds, int
        ):
            required_rounds = getattr(
                game_state,
                "lps_rounds_required",
                getattr(game_state, "lpsRoundsRequired", 3),
            )
            if not isinstance(required_rounds, int) or required_rounds <= 0:
                required_rounds = 3

            if lps_rounds >= required_rounds and required_rounds >= 1:
                return self.WEIGHT_VICTORY_THRESHOLD_BONUS
            if lps_rounds > 0:
                if required_rounds <= 1:
                    return self.WEIGHT_VICTORY_THRESHOLD_BONUS
                denom = float(required_rounds - 1)
                frac = min(1.0, max(0.0, float(lps_rounds) / denom))
                return self.WEIGHT_VICTORY_THRESHOLD_BONUS * (0.90 + 0.09 * frac)

        rings_needed = game_state.victory_threshold - player.eliminated_rings
        territory_needed = (
            game_state.territory_victory_threshold - player.territory_spaces
        )

        if rings_needed <= 0 or territory_needed <= 0:
            return self.WEIGHT_VICTORY_THRESHOLD_BONUS

        score = 0.0
        score += (1.0 / max(1, rings_needed)) * self.WEIGHT_RINGS_PROXIMITY_FACTOR
        score += (
            (1.0 / max(1, territory_needed)) * self.WEIGHT_TERRITORY_PROXIMITY_FACTOR
        )
        return score

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move using heuristic evaluation.

        Args:
            game_state: Current game state.

        Returns:
            The best heuristic :class:`Move` or ``None`` if there are no
            valid moves for this player.
        """
        # Try to get valid moves from cache first
        valid_moves = None
        if USE_MOVE_CACHE:
            valid_moves = get_cached_moves(game_state, self.player_number)

        # If not cached, get from rules engine and cache the result
        if valid_moves is None:
            valid_moves = self.get_valid_moves(game_state)
            if USE_MOVE_CACHE and valid_moves:
                cache_moves(game_state, self.player_number, valid_moves)

        if not valid_moves:
            return None

        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
        else:
            # Apply training move sample limit if configured
            moves_to_evaluate = self._sample_moves_for_training(valid_moves)

            # Evaluate each move and collect all moves with the best score
            # (random tie-breaking among equally good moves)
            best_moves: list[Move] = []
            best_score = float('-inf')

            # Choose evaluation strategy based on number of moves and flags
            num_moves = len(moves_to_evaluate)
            use_batch = (
                USE_BATCH_EVAL and
                USE_MAKE_UNMAKE and
                num_moves >= BATCH_EVAL_THRESHOLD
            )

            if use_batch:
                # Batch path: NumPy vectorized evaluation (fastest for many moves)
                move_scores = self._evaluate_moves_batch(game_state, moves_to_evaluate)

                for move, score in move_scores:
                    # Add stochastic exploration for SWAP_SIDES
                    if move.type == MoveType.SWAP_SIDES and self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE > 0:
                        noise = self.rng.gauss(0, self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE)
                        score += noise

                    if score > best_score:
                        best_score = score
                        best_moves = [move]
                    elif score == best_score:
                        best_moves.append(move)

            elif USE_MAKE_UNMAKE:
                # Make/unmake path: lightweight state with sequential evaluation
                move_scores = self._evaluate_moves_fast(game_state, moves_to_evaluate)

                for move, score in move_scores:
                    # Add stochastic exploration for SWAP_SIDES
                    if move.type == MoveType.SWAP_SIDES and self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE > 0:
                        noise = self.rng.gauss(0, self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE)
                        score += noise

                    if score > best_score:
                        best_score = score
                        best_moves = [move]
                    elif score == best_score:
                        best_moves.append(move)
            elif self._should_use_parallel(game_state, len(moves_to_evaluate)):
                # Parallel path: distribute apply_move + evaluate across CPU cores
                # Uses full heuristics - no change in play strength
                move_scores = self._evaluate_moves_parallel(
                    game_state, moves_to_evaluate
                )

                for move, score in move_scores:
                    # Add stochastic exploration for SWAP_SIDES
                    if move.type == MoveType.SWAP_SIDES and self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE > 0:
                        noise = self.rng.gauss(0, self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE)
                        score += noise

                    if score > best_score:
                        best_score = score
                        best_moves = [move]
                    elif score == best_score:
                        best_moves.append(move)
            else:
                # Original path: full state copy per move (sequential)
                for move in moves_to_evaluate:
                    # Simulate the move to get the resulting state
                    next_state = self.rules_engine.apply_move(game_state, move)

                    # For SWAP_SIDES, evaluate from the NEW player's perspective
                    # After swap, the AI that was P2 becomes P1 and vice versa
                    if move.type == MoveType.SWAP_SIDES:
                        # Temporarily change perspective to evaluate as new player
                        original_player = self.player_number
                        # After swap, P2 becomes P1 (swap toggles 1<->2)
                        self.player_number = 1 if original_player == 2 else 2
                        score = self.evaluate_position(next_state)
                        # Add swap bonus for strong opening positions
                        score += self.evaluate_swap_opening_bonus(game_state)

                        # Add stochastic exploration for training diversity
                        if self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE > 0:
                            noise = self.rng.gauss(
                                0,
                                self.WEIGHT_SWAP_EXPLORATION_TEMPERATURE
                            )
                            score += noise

                        # Restore original perspective
                        self.player_number = original_player
                    else:
                        score = self.evaluate_position(next_state)

                    if score > best_score:
                        # New best score - clear previous candidates
                        best_score = score
                        best_moves = [move]
                    elif score == best_score:
                        # Tie - add to candidates for random selection
                        best_moves.append(move)

            selected = (
                self.get_random_element(best_moves)
                if best_moves
                else self.get_random_element(valid_moves)
            )

        self.move_count += 1
        return selected

    def _sample_moves_for_training(self, moves: list[Move]) -> list[Move]:
        """
        Sample moves for evaluation if training_move_sample_limit is set.

        This is a training/evaluation performance optimization that randomly
        samples a subset of moves when there are too many to evaluate
        efficiently. The sampling uses self.rng (inherited from BaseAI),
        ensuring deterministic behavior when a seed is provided.

        Args:
            moves: Full list of valid moves

        Returns:
            Either the original list (if no limit or under limit) or a
            random sample up to the configured limit.
        """
        limit = getattr(self.config, "training_move_sample_limit", None)

        # No sampling if limit is not configured or moves are under limit
        if limit is None or limit <= 0 or len(moves) <= limit:
            return moves

        # Use self.rng (seeded in BaseAI) for deterministic sampling.
        # This ensures reproducibility when config.rng_seed is provided.
        return self.rng.sample(moves, limit)

    def _should_use_parallel(
        self,
        game_state: GameState,
        num_moves: int,
    ) -> bool:
        """
        Determine if parallel evaluation should be used.

        Parallel is enabled by default for large boards (hex, 19x19)
        where the speedup outweighs overhead. Can be overridden via
        RINGRIFT_USE_PARALLEL_EVAL environment variable.

        Args:
            game_state: Current game state
            num_moves: Number of moves to evaluate

        Returns:
            True if parallel evaluation should be used
        """
        # Check minimum move threshold
        if num_moves < PARALLEL_MIN_MOVES:
            return False

        # If explicitly set, honor the environment variable
        if USE_PARALLEL_EVAL:
            return True

        # Auto-enable for large boards where parallel provides benefit
        board_type = game_state.board.type
        if board_type == BoardType.HEXAGONAL:
            return True
        return board_type == BoardType.SQUARE19

    def _evaluate_moves_parallel(
        self,
        game_state: GameState,
        moves: list[Move],
    ) -> list[tuple[Move, float]]:
        """
        Evaluate moves in parallel using ProcessPoolExecutor.

        Uses full heuristics - no change in play strength compared to
        sequential evaluation. Distributes apply_move + evaluate_position
        across multiple CPU cores using chunked work distribution.

        Args:
            game_state: Current game state
            moves: List of moves to evaluate

        Returns:
            List of (move, score) tuples
        """
        # Prepare worker arguments
        config_dict = {
            'difficulty': self.config.difficulty,
            'rng_seed': getattr(self.config, 'rng_seed', None),
        }

        # Determine number of workers and chunk size
        num_workers = PARALLEL_WORKERS if PARALLEL_WORKERS > 0 else (
            multiprocessing.cpu_count() or 4
        )
        chunk_size = max(1, len(moves) // num_workers)

        # Create chunks of moves
        indexed_moves = list(enumerate(moves))
        chunks = []
        for i in range(0, len(indexed_moves), chunk_size):
            chunk = indexed_moves[i:i + chunk_size]
            chunks.append((chunk, game_state, self.player_number, config_dict))

        # Use persistent executor to avoid process creation overhead
        executor = _get_parallel_executor()

        # Evaluate chunks in parallel
        results: dict[int, float] = {}
        futures = [
            executor.submit(_evaluate_moves_chunk_worker, chunk_args)
            for chunk_args in chunks
        ]
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                for move_idx, score in chunk_results:
                    results[move_idx] = score
            except Exception:
                # On error, moves in this chunk get very low scores
                pass

        # Reconstruct (move, score) list in original order
        move_scores = []
        for i, move in enumerate(moves):
            score = results.get(i, float('-inf'))
            move_scores.append((move, score))

        return move_scores

    def _evaluate_moves_fast(
        self,
        game_state: GameState,
        moves: list[Move],
        early_term: bool = True,
    ) -> list[tuple]:
        """
        Evaluate moves using make/unmake pattern with lightweight state.

        This avoids creating new GameState objects for each candidate move,
        instead using in-place mutation with undo capability.

        Args:
            game_state: Current game state
            moves: List of moves to evaluate
            early_term: If True, may terminate early if dominant move found

        Returns:
            List of (move, score) tuples
        """
        # Convert to lightweight state once
        light_state = LightweightState.from_game_state(game_state)

        # Extract weights for fast evaluation
        weights = extract_weights_from_ai(self)

        results = []
        best_score = float('-inf')
        second_best_score = float('-inf')

        for i, move in enumerate(moves):
            # Apply move to lightweight state
            undo = self._make_move_light(light_state, move)

            # Evaluate position
            if move.type == MoveType.SWAP_SIDES:
                # After swap, evaluate from new player's perspective
                new_player = 1 if self.player_number == 2 else 2
                score = evaluate_position_light(
                    light_state, new_player, weights, self.eval_mode
                )
                # Add swap bonus for strong opening positions
                score += self._evaluate_swap_opening_bonus_light(
                    light_state, player_number=new_player
                )
            else:
                score = evaluate_position_light(
                    light_state, self.player_number, weights, self.eval_mode
                )

            results.append((move, score))

            # Track best and second-best scores for early termination
            if score > best_score:
                second_best_score = best_score
                best_score = score
            elif score > second_best_score:
                second_best_score = score

            # Early termination check: if we've evaluated enough moves
            # and found a dominant move, stop early
            if (early_term and
                    EARLY_TERM_THRESHOLD > 0 and
                    i >= EARLY_TERM_MIN_MOVES and
                    best_score - second_best_score >= EARLY_TERM_THRESHOLD):
                # Undo before breaking
                light_state.unmake_move(undo)
                break

            # Undo move to restore state
            light_state.unmake_move(undo)

        return results

    def _make_move_light(
        self,
        state: LightweightState,
        move: Move,
    ) -> MoveUndo:
        """Apply a move to lightweight state and return undo info."""
        if move.type == MoveType.PLACE_RING:
            if move.to is None:
                return MoveUndo()
            to_key = move.to.to_key()
            return state.make_place_ring(to_key, move.player)

        elif move.type == MoveType.MOVE_STACK:
            if move.to is None:
                return MoveUndo()
            to_key = move.to.to_key()
            from_key = move.from_pos.to_key() if move.from_pos else ""
            return state.make_move_stack(from_key, to_key, move.player)

        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.CHAIN_CAPTURE,
        ):
            if move.to is None:
                return MoveUndo()
            to_key = move.to.to_key()
            from_key = move.from_pos.to_key() if move.from_pos else ""
            return state.make_capture(from_key, to_key, move.player)

        elif move.type == MoveType.SWAP_SIDES:
            # Swap is a meta-move that swaps player identities
            # For evaluation purposes, we just need to track the swap
            return MoveUndo()

        else:
            # For other move types (line processing, territory, etc.)
            # we return empty undo as these are decision moves
            return MoveUndo()

    def _evaluate_swap_opening_bonus_light(
        self,
        state: LightweightState,
        player_number: int | None = None,
    ) -> float:
        """Evaluate swap opening bonus using lightweight state.

        Delegates to SwapEvaluator.evaluate_swap_light for consistency.
        """
        effective_player = (
            int(player_number)
            if isinstance(player_number, int) and player_number > 0
            else self.player_number
        )
        return self.swap_evaluator.evaluate_swap_light(state.stacks, effective_player)

    def _evaluate_moves_batch(
        self,
        game_state: GameState,
        moves: list[Move],
    ) -> list[tuple]:
        """
        Evaluate moves using NumPy batch operations.

        This is the fastest path for large numbers of moves (100+),
        using vectorized operations instead of iterating through moves.

        Args:
            game_state: Current game state
            moves: List of moves to evaluate

        Returns:
            List of (move, score) tuples
        """
        if not moves:
            return []

        # Convert to lightweight state first, then to NumPy arrays
        # Use lazy BoardArrays reuse for faster repeated evaluations
        light_state = LightweightState.from_game_state(game_state)
        board_arrays = get_or_update_board_arrays(
            light_state, self._cached_board_arrays
        )
        # Cache for next call
        self._cached_board_arrays = board_arrays

        # Extract weights for batch evaluation
        weights = extract_weights_from_ai(self)

        # Prepare moves for batch processing
        move_tuples = prepare_moves_for_batch(moves, board_arrays.position_to_idx)

        # Batch evaluate all moves
        scores = batch_evaluate_positions(
            board_arrays,
            move_tuples,
            self.player_number,
            weights,
        )

        # Handle SWAP_SIDES moves separately (need special evaluation)
        results = []
        for i, move in enumerate(moves):
            score = float(scores[i])

            # For SWAP_SIDES, batch evaluation scores are in the original
            # player's perspective; swap changes seat identity, so we must
            # re-evaluate from the swapped perspective.
            if move.type == MoveType.SWAP_SIDES:
                new_player = 1 if self.player_number == 2 else 2
                score = evaluate_position_light(
                    light_state, new_player, weights, self.eval_mode
                )
                score += self._evaluate_swap_opening_bonus_light(
                    light_state, player_number=new_player
                )

            results.append((move, score))

        return results

    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the current position using heuristics

        Args:
            game_state: Current game state

        Returns:
            Evaluation score (positive = good for this AI)
        """
        # Check for game over first
        if game_state.game_status == "completed":
            if game_state.winner == self.player_number:
                return 100000.0
            elif game_state.winner is not None:
                return -100000.0
            else:
                return 0.0

        # Delegate to the shared component computation so that the scalar
        # evaluation and the per-feature breakdown remain strictly aligned.
        components = self._compute_component_scores(game_state)
        return sum(components.values())

    def _compute_component_scores(
        self,
        game_state: GameState,
    ) -> dict[str, float]:
        """
        Compute per-feature component scores for the current position.

        This helper centralises mode-aware gating for Tier 0/1/2 features so
        that :meth:`evaluate_position` and :meth:`get_evaluation_breakdown`
        remain consistent. In ``"light"`` mode, Tier-2 structural features are
        not evaluated at all and are reported as ``0.0``.
        """
        # Clear visibility cache for this evaluation (avoids stale data)
        self._visible_stacks_cache: dict[str, list] = {}
        scores: dict[str, float] = {}

        # Tier 0 (core) features – always computed.
        scores["stack_control"] = self._evaluate_stack_control(game_state)
        scores["territory"] = self._evaluate_territory(game_state)
        scores["rings_in_hand"] = self._evaluate_rings_in_hand(game_state)
        scores["center_control"] = self._evaluate_center_control(game_state)

        # Tier 1 (local adjacency/mobility) features – always computed.
        scores["opponent_threats"] = self._evaluate_opponent_threats(
            game_state
        )
        scores["mobility"] = self._evaluate_mobility(game_state)

        # Remaining Tier 0 core features.
        scores["eliminated_rings"] = self._evaluate_eliminated_rings(
            game_state
        )

        # Tier 2 (structural/global) – gated by eval_mode.
        if self.eval_mode == "full":
            scores["line_potential"] = self._evaluate_line_potential(
                game_state
            )
        else:
            scores["line_potential"] = 0.0

        scores["victory_proximity"] = self._evaluate_victory_proximity(
            game_state
        )
        scores["opponent_victory_threat"] = (
            self._evaluate_opponent_victory_threat(game_state)
        )
        scores["marker_count"] = self._evaluate_marker_count(game_state)

        if self.eval_mode == "full":
            scores["vulnerability"] = self._evaluate_vulnerability(game_state)
            scores["overtake_potential"] = self._evaluate_overtake_potential(
                game_state
            )
            scores["territory_closure"] = self._evaluate_territory_closure(
                game_state
            )
            scores["line_connectivity"] = self._evaluate_line_connectivity(
                game_state
            )
            scores["territory_safety"] = self._evaluate_territory_safety(
                game_state
            )
        else:
            scores["vulnerability"] = 0.0
            scores["overtake_potential"] = 0.0
            scores["territory_closure"] = 0.0
            scores["line_connectivity"] = 0.0
            scores["territory_safety"] = 0.0

        # Tier 1 stack-level mobility – always computed.
        scores["stack_mobility"] = self._evaluate_stack_mobility(game_state)

        # High-signal structural features – gated by eval_mode.
        if self.eval_mode == "full":
            scores["forced_elimination_risk"] = (
                self._evaluate_forced_elimination_risk(game_state)
            )
            scores["lps_action_advantage"] = (
                self._evaluate_lps_action_advantage(game_state)
            )
        else:
            scores["forced_elimination_risk"] = 0.0
            scores["lps_action_advantage"] = 0.0

        # Multi-leader threat is considered core and always computed.
        scores["multi_leader_threat"] = self._evaluate_multi_leader_threat(
            game_state
        )

        # Recovery potential evaluation (v1.6) – gated by eval_mode.
        # This captures the strategic value of recovery as a threat/resource.
        if self.eval_mode == "full":
            scores["recovery_potential"] = self._evaluate_recovery_potential(
                game_state
            )
        else:
            scores["recovery_potential"] = 0.0

        return scores

    def get_evaluation_breakdown(
        self,
        game_state: GameState
    ) -> dict[str, float]:
        """
        Get detailed breakdown of position evaluation

        Args:
            game_state: Current game state

        Returns:
            Dictionary with evaluation components
        """
        components = self._compute_component_scores(game_state)
        total = sum(components.values())
        breakdown: dict[str, float] = {"total": total}
        breakdown.update(components)
        return breakdown

    def _evaluate_stack_control(self, game_state: GameState) -> float:
        """Evaluate stack control (made symmetric).

        All penalties and bonuses use configurable weights to enable
        full weight-space exploration during training.

        v1.5: Added cap height tracking - sum of cap_height across controlled
        stacks measures capture power dominance separately from total height.

        v1.6: Made diversification penalties symmetric by computing relative
        advantage (my_penalty - opponent_penalty) instead of absolute penalty.
        """
        score = 0.0
        my_stacks = 0
        opponent_stacks = 0
        my_height = 0
        opponent_height = 0
        my_cap_height = 0
        opponent_cap_height = 0

        for stack in game_state.board.stacks.values():
            if stack.controlling_player == self.player_number:
                my_stacks += 1
                # Diminishing returns for height > 5 to discourage mega-stacks
                h = stack.stack_height
                effective_height = h if h <= 5 else 5 + (h - 5) * 0.1
                my_height += effective_height
                # Cap height measures capture power (per rules §10.1)
                my_cap_height += stack.cap_height
            else:
                opponent_stacks += 1
                h = stack.stack_height
                effective_height = h if h <= 5 else 5 + (h - 5) * 0.1
                opponent_height += effective_height
                opponent_cap_height += stack.cap_height

        # Stack diversification - now symmetric using relative penalties
        def diversification_score(stacks: int) -> float:
            """Compute diversification score for a stack count."""
            if stacks == 0:
                return -self.WEIGHT_NO_STACKS_PENALTY
            elif stacks == 1:
                return -self.WEIGHT_SINGLE_STACK_PENALTY
            else:
                return stacks * self.WEIGHT_STACK_DIVERSITY_BONUS

        my_diversity = diversification_score(my_stacks)
        opp_diversity = diversification_score(opponent_stacks)
        score += (my_diversity - opp_diversity)

        score += (my_stacks - opponent_stacks) * self.WEIGHT_STACK_CONTROL
        score += (my_height - opponent_height) * self.WEIGHT_STACK_HEIGHT
        score += (my_cap_height - opponent_cap_height) * self.WEIGHT_CAP_HEIGHT

        return score

    def _evaluate_territory(self, game_state: GameState) -> float:
        """Evaluate territory control"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        my_territory = my_player.territory_spaces

        # Compare with opponents
        opponent_territory = 0
        for player in game_state.players:
            if player.player_number != self.player_number:
                opponent_territory = max(
                    opponent_territory,
                    player.territory_spaces
                )

        return (my_territory - opponent_territory) * self.WEIGHT_TERRITORY

    def _evaluate_rings_in_hand(self, game_state: GameState) -> float:
        """Evaluate rings remaining in hand (relative to opponents).

        Made symmetric: computes (my_rings - max_opponent_rings) so that
        the evaluation sums to approximately zero across all players.
        """
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        my_rings = my_player.rings_in_hand

        # Find max opponent rings for symmetric evaluation
        max_opponent_rings = 0
        for p in game_state.players:
            if p.player_number != self.player_number:
                max_opponent_rings = max(max_opponent_rings, p.rings_in_hand)

        # Symmetric: advantage over best opponent
        return (my_rings - max_opponent_rings) * self.WEIGHT_RINGS_IN_HAND

    def _evaluate_center_control(self, game_state: GameState) -> float:
        """Evaluate control of center positions"""
        score = 0.0
        center_positions = self._get_center_positions(game_state)

        for pos_key in center_positions:
            if pos_key in game_state.board.stacks:
                stack = game_state.board.stacks[pos_key]
                if stack.controlling_player == self.player_number:
                    score += self.WEIGHT_CENTER_CONTROL
                else:
                    score -= self.WEIGHT_CENTER_CONTROL * 0.5

        return score

    def _evaluate_opponent_threats(self, game_state: GameState) -> float:
        """Evaluate opponent threats (stacks near our stacks)"""
        score = 0.0
        board = game_state.board
        board_type = board.type
        stacks = board.stacks
        my_stacks = [s for s in stacks.values()
                     if s.controlling_player == self.player_number]

        for my_stack in my_stacks:
            # Use fast key-based adjacency lookup
            pos_key = my_stack.position.to_key()
            adjacent_keys = self._get_adjacent_keys(pos_key, board_type)
            for adj_key in adjacent_keys:
                if adj_key in stacks:
                    adj_stack = stacks[adj_key]
                    if adj_stack.controlling_player != self.player_number:
                        # Opponent stack adjacent to ours is a threat.
                        # Capture power is based on cap height per compact
                        # rules §10.1, so we compare using cap height here
                        # rather than total stack height.
                        threat_level = (
                            adj_stack.cap_height - my_stack.cap_height
                        )
                        score -= (
                            threat_level * self.WEIGHT_OPPONENT_THREAT * 0.5
                        )

        return score

    def _evaluate_mobility(self, game_state: GameState) -> float:
        """Evaluate mobility (number of valid moves)"""
        # Optimization: Use pseudo-mobility instead of full move generation
        # Full move generation is too expensive for evaluation function.

        board = game_state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces

        # My pseudo-mobility
        my_stacks = [
            s for s in stacks.values()
            if s.controlling_player == self.player_number
        ]
        my_mobility = 0
        for stack in my_stacks:
            # Use fast key-based adjacency lookup
            pos_key = stack.position.to_key()
            adjacent_keys = self._get_adjacent_keys(pos_key, board_type)
            for adj_key in adjacent_keys:
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    # Capture power is based on cap height per compact rules §10.1
                    if (target.controlling_player != self.player_number
                            and stack.cap_height >= target.cap_height):
                        my_mobility += 1
                else:
                    my_mobility += 1

        # Opponent pseudo-mobility
        opp_stacks = [
            s for s in stacks.values()
            if s.controlling_player != self.player_number
        ]
        opp_mobility = 0
        for stack in opp_stacks:
            # Use fast key-based adjacency lookup
            pos_key = stack.position.to_key()
            adjacent_keys = self._get_adjacent_keys(pos_key, board_type)
            for adj_key in adjacent_keys:
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    # Capture power is based on cap height per compact rules §10.1
                    if (target.controlling_player == self.player_number
                            and stack.cap_height >= target.cap_height):
                        opp_mobility += 1
                else:
                    opp_mobility += 1
        score = (my_mobility - opp_mobility) * self.WEIGHT_MOBILITY
        return score

    def _iterate_board_keys(self, board) -> list[str]:
        """
        Iterate over all logical board coordinate keys for the given board.

        Uses pre-computed board keys from FastGeometry for O(1) lookup
        instead of regenerating Position objects on every call.
        """
        return self._fast_geo.get_all_board_keys(board.type)

    def _approx_real_actions_for_player(
        self,
        game_state: GameState,
        player_number: int,
    ) -> int:
        """
        Approximate the number of "real" actions (moves + placements) available
        to the given player.

        - Counts one move per stack that has at least one legal-looking move
          (empty neighbor or capturable enemy stack).
        - Adds one additional action if the player has rings in hand and there
          exists at least one empty, non-collapsed space where a ring could be
          placed.
        """
        board = game_state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces
        approx_moves = 0

        for stack in stacks.values():
            if stack.controlling_player != player_number:
                continue

            # Use fast key-based adjacency lookup
            pos_key = stack.position.to_key()
            adjacent_keys = self._get_adjacent_keys(pos_key, board_type)
            stack_has_any_move = False
            for key in adjacent_keys:
                if key in collapsed:
                    continue

                if key in stacks:
                    target = stacks[key]
                    if (
                        target.controlling_player != player_number
                        and stack.cap_height >= target.cap_height
                    ):
                        stack_has_any_move = True
                        break
                else:
                    stack_has_any_move = True
                    break

            if stack_has_any_move:
                approx_moves += 1

        approx_placement = 0
        player = next(
            (
                p
                for p in game_state.players
                if p.player_number == player_number
            ),
            None,
        )
        if player and player.rings_in_hand > 0:
            has_empty = any(
                key not in board.stacks and key not in board.collapsed_spaces
                for key in self._iterate_board_keys(board)
            )
            if has_empty:
                approx_placement = 1

        return approx_moves + approx_placement

    def _evaluate_influence(self, game_state: GameState) -> float:
        """Evaluate board influence"""
        board = game_state.board
        board_type = board.type

        # Influence map: +1 for my stack, -1 for opponent stack
        # Decay by distance

        influence_map = {}

        for stack in board.stacks.values():
            if stack.controlling_player == self.player_number:
                value = 1.0
            else:
                value = -1.0
            # Base influence at stack position
            pos_key = stack.position.to_key()
            influence_map[pos_key] = (
                influence_map.get(pos_key, 0) + value * 2.0
            )

            # Project to neighbors (distance 1) using fast key lookup
            neighbor_keys = self._get_adjacent_keys(pos_key, board_type)
            for n_key in neighbor_keys:
                influence_map[n_key] = (
                    influence_map.get(n_key, 0) + value * 1.0
                )

        # Sum up positive influence (my control) vs negative (opponent control)
        total_influence = sum(influence_map.values())

        return total_influence * 2.0  # Weight for influence

    def _evaluate_eliminated_rings(self, game_state: GameState) -> float:
        """Evaluate eliminated rings"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        return my_player.eliminated_rings * self.WEIGHT_ELIMINATED_RINGS

    def _evaluate_line_potential(self, game_state: GameState) -> float:
        """Evaluate potential to form lines (2, 3, 4 in a row).

        Weights used:
        - WEIGHT_TWO_IN_ROW: Adjacent marker pairs in line directions
        - WEIGHT_THREE_IN_ROW: Three consecutive markers
        - WEIGHT_FOUR_IN_ROW: Four consecutive markers (almost a winning line)

        Note: WEIGHT_TWO_IN_ROW is redundant with WEIGHT_CONNECTED_NEIGHBOR in
        _evaluate_line_connectivity - both count the same adjacent marker pairs.
        See module docstring "Known Weight Redundancies" for details.

        Uses Numba JIT-compiled evaluation when available for ~10x speedup.
        Falls back to FastGeometry pre-computed offset tables for O(1) lookups.
        """
        board = game_state.board
        board_type = board.type
        markers = board.markers

        # Numba JIT path disabled by default - benchmarks show data conversion
        # overhead negates JIT gains for typical marker counts (10-50).
        # Enable via environment variable for experimentation on larger boards.
        # if NUMBA_AVAILABLE and os.getenv('RINGRIFT_USE_NUMBA_EVAL') and len(markers) > 0:
        #     return self._evaluate_line_potential_numba(game_state)

        # Fallback: Pure Python with FastGeometry optimizations
        score = 0.0

        # Get number of directions for this board type
        num_directions = 8 if board_type != BoardType.HEXAGONAL else 6

        # Iterate through all markers of the player
        my_markers = [
            m for m in markers.values()
            if m.player == self.player_number
        ]

        for marker in my_markers:
            start_key = marker.position.to_key()

            for dir_idx in range(num_directions):
                # Check for 2 or 3 markers in a row
                # We only check forward to avoid double counting (mostly)

                # Check length 2 (use ultra-fast pre-computed lookup)
                key2 = self._fast_geo.offset_key_fast(start_key, dir_idx, 1, board_type)
                if key2 is None:
                    continue

                if (key2 in markers and
                        markers[key2].player == self.player_number):
                    score += self.WEIGHT_TWO_IN_ROW  # 2 in a row

                    # Check length 3
                    key3 = self._fast_geo.offset_key_fast(start_key, dir_idx, 2, board_type)
                    if key3 is None:
                        continue

                    if (key3 in markers and
                            markers[key3].player == self.player_number):
                        score += self.WEIGHT_THREE_IN_ROW  # 3 in a row (cumulative)

                        # Check length 4 (almost a line)
                        key4 = self._fast_geo.offset_key_fast(
                            start_key, dir_idx, 3, board_type
                        )
                        if key4 is None:
                            continue

                        if (key4 in markers and
                                markers[key4].player ==
                                self.player_number):
                            score += self.WEIGHT_FOUR_IN_ROW  # 4 in a row

        return score * self.WEIGHT_LINE_POTENTIAL

    def _evaluate_line_potential_numba(self, game_state: GameState) -> float:
        """Numba JIT-compiled line potential evaluation."""
        board = game_state.board
        board_type = board.type
        markers = board.markers

        # Prepare numpy arrays from markers
        player_positions, all_marker_keys = prepare_marker_arrays(
            markers, self.player_number
        )

        if len(player_positions) == 0:
            return 0.0

        # Determine board parameters
        board_type_is_hex = board_type == BoardType.HEXAGONAL
        board_size = board.size if hasattr(board, 'size') else 7  # Default hex radius

        # Call JIT-compiled function
        score = evaluate_line_potential_numba(
            player_positions,
            all_marker_keys,
            board_type_is_hex,
            board_size,
            self.WEIGHT_TWO_IN_ROW,
            self.WEIGHT_THREE_IN_ROW,
            self.WEIGHT_FOUR_IN_ROW,
        )

        return score * self.WEIGHT_LINE_POTENTIAL

    def _evaluate_victory_proximity(self, game_state: GameState) -> float:
        """Evaluate how close we are to winning (relative to opponents).

        Made symmetric: computes (my_proximity - max_opponent_proximity) so
        that the evaluation sums to approximately zero across all players.
        """
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        my_proximity = self._victory_proximity_base_for_player(game_state, my_player)

        # Find max opponent proximity for symmetric evaluation
        max_opponent_proximity = 0.0
        for p in game_state.players:
            if p.player_number != self.player_number:
                opp_proximity = self._victory_proximity_base_for_player(game_state, p)
                max_opponent_proximity = max(max_opponent_proximity, opp_proximity)

        # Symmetric: advantage over best opponent
        return (my_proximity - max_opponent_proximity) * self.WEIGHT_VICTORY_PROXIMITY

    def _evaluate_opponent_victory_threat(
        self,
        game_state: GameState,
    ) -> float:
        """
        Evaluate how much closer the leading opponent is to victory
        than we are.

        This mirrors the self victory proximity computation and
        compares our proximity score to the maximum proximity score
        among all opponents. A positive gap is treated as a threat
        and converted into a penalty.
        """
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0

        self_prox = self._victory_proximity_base_for_player(game_state, my_player)

        max_opp_prox = 0.0
        for p in game_state.players:
            if p.player_number == self.player_number:
                continue
            prox = self._victory_proximity_base_for_player(game_state, p)
            if prox > max_opp_prox:
                max_opp_prox = prox

        raw_gap = max_opp_prox - self_prox
        relative_threat = max(0.0, raw_gap)

        return -relative_threat * self.WEIGHT_OPPONENT_VICTORY_THREAT

    def _evaluate_marker_count(self, game_state: GameState) -> float:
        """Evaluate number of markers on board"""
        my_markers = 0
        for marker in game_state.board.markers.values():
            if marker.player == self.player_number:
                my_markers += 1

        return my_markers * self.WEIGHT_MARKER_COUNT

    def _get_visible_stacks(
        self,
        position: Position,
        game_state: GameState,
    ) -> list[RingStack]:
        """
        Compute line-of-sight visible stacks from a position.

        Optimized version: Uses raw coordinates instead of creating Position
        objects in the inner loop. This provides ~3x speedup for this function.

        Results are cached per-evaluation to avoid redundant computations
        when the same position is queried multiple times (e.g., vulnerability
        and overtake potential both iterate over the same stacks).
        """
        # Check cache first
        cache_key = position.to_key()
        if hasattr(self, '_visible_stacks_cache') and cache_key in self._visible_stacks_cache:
            return self._visible_stacks_cache[cache_key]

        visible: list[RingStack] = []
        board = game_state.board
        board_type = board.type
        stacks = board.stacks

        # Get directions from FastGeometry (cached)
        directions = self._fast_geo.get_los_directions(board_type)

        # Pre-compute bounds limits for inline checking
        is_hex = board_type == BoardType.HEXAGONAL
        if is_hex:
            # Canonical hex board uses cube radius = (size - 1) (RR-CANON-R001).
            # BoardState.size is 13 for the canonical radius-12 board.
            board_size = int(getattr(board, "size", 13) or 13)
            limit = board_size - 1
        elif board_type == BoardType.SQUARE8:
            limit = 8
        else:  # SQUARE19
            limit = 19

        curr_x = position.x
        curr_y = position.y
        # For hex, compute z from x,y (constraint: x + y + z = 0)
        if is_hex:
            curr_z = position.z if position.z is not None else -position.x - position.y
        else:
            curr_z = 0

        for dx, dy, dz in directions:
            x, y, z = curr_x, curr_y, curr_z

            while True:
                x += dx
                y += dy
                if is_hex:
                    z += dz
                    # Inline hex bounds check
                    if abs(x) > limit or abs(y) > limit or abs(z) > limit:
                        break
                    # Hex stack keys include cube z coordinate (Position.to_key()).
                    pos_key = f"{x},{y},{z}"
                else:
                    # Inline square bounds check
                    if x < 0 or x >= limit or y < 0 or y >= limit:
                        break
                    pos_key = f"{x},{y}"

                stack = stacks.get(pos_key)
                if stack is not None:
                    visible.append(stack)
                    break

        # Cache result for this evaluation
        if hasattr(self, '_visible_stacks_cache'):
            self._visible_stacks_cache[cache_key] = visible

        return visible

    def _evaluate_vulnerability(self, game_state: GameState) -> float:
        """
        Evaluate vulnerability of our stacks to overtaking captures.
        Considers relative cap heights of stacks in clear line of sight.
        """
        score = 0.0
        my_stacks = [
            s
            for s in game_state.board.stacks.values()
            if s.controlling_player == self.player_number
        ]

        for stack in my_stacks:
            visible_stacks = self._get_visible_stacks(
                stack.position,
                game_state
            )

            for adj_stack in visible_stacks:
                # Capture power is based on cap height per compact rules §10.1
                if (adj_stack.controlling_player != self.player_number
                        and adj_stack.cap_height > stack.cap_height):
                    diff = adj_stack.cap_height - stack.cap_height
                    score -= diff * 1.0

        return score * self.WEIGHT_VULNERABILITY

    def _evaluate_overtake_potential(self, game_state: GameState) -> float:
        """
        Evaluate our ability to overtake opponent stacks.
        Considers relative cap heights of stacks in clear line of sight.
        """
        score = 0.0
        my_stacks = [
            s
            for s in game_state.board.stacks.values()
            if s.controlling_player == self.player_number
        ]

        for stack in my_stacks:
            visible_stacks = self._get_visible_stacks(
                stack.position,
                game_state
            )

            for adj_stack in visible_stacks:
                # Capture power is based on cap height per compact rules §10.1
                if (adj_stack.controlling_player != self.player_number
                        and stack.cap_height > adj_stack.cap_height):
                    diff = stack.cap_height - adj_stack.cap_height
                    score += diff * 1.0

        return score * self.WEIGHT_OVERTAKE_POTENTIAL

    def _evaluate_territory_closure(self, game_state: GameState) -> float:
        """
        Evaluate how close we are to enclosing a territory.
        Uses a simplified metric: number of markers vs board size/density.
        """
        # This is a complex heuristic to implement perfectly without
        # pathfinding. As a proxy, we look at marker density and clustering.

        my_markers = [m for m in game_state.board.markers.values()
                      if m.player == self.player_number]

        if not my_markers:
            return 0.0

        # Calculate "clustering" - average distance between markers
        # Closer markers are more likely to form a closed loop
        total_dist = 0.0
        count = 0

        # Sample a few pairs to estimate density if too many markers
        # Deterministic subsampling: take first 10
        if len(my_markers) < 10:
            markers_to_check = my_markers
        else:
            markers_to_check = my_markers[:10]

        for i, m1 in enumerate(markers_to_check):
            for m2 in markers_to_check[i+1:]:
                dist = (
                    abs(m1.position.x - m2.position.x) +
                    abs(m1.position.y - m2.position.y)
                )
                if m1.position.z is not None and m2.position.z is not None:
                    dist += abs(m1.position.z - m2.position.z)

                total_dist += dist
                count += 1

        if count == 0:
            return 0.0

        avg_dist = total_dist / count

        # Lower average distance is better (more clustered)
        # We invert it for the score
        clustering_score = 10.0 / max(1.0, avg_dist)

        # Also reward total number of markers as a prerequisite for territory
        marker_count_score = len(my_markers) * 0.5

        return (
            (clustering_score + marker_count_score) *
            self.WEIGHT_TERRITORY_CLOSURE
        )

    # _evaluate_move is deprecated in favor of evaluate_position
    # on the simulated state

    def _get_center_positions(self, game_state: GameState) -> frozenset[str]:
        """Get center position keys for the board using FastGeometry cache."""
        return self._fast_geo.get_center_positions(game_state.board.type)

    def _evaluate_line_connectivity(self, game_state: GameState) -> float:
        """Evaluate connectivity of markers and gap potential.

        Weights used:
        - WEIGHT_CONNECTED_NEIGHBOR: Adjacent marker pairs in line directions
        - WEIGHT_GAP_POTENTIAL: Markers at distance 2 with empty gap between

        Note: WEIGHT_CONNECTED_NEIGHBOR is redundant with WEIGHT_TWO_IN_ROW in
        _evaluate_line_potential - both count the same adjacent marker pairs.
        See module docstring "Known Weight Redundancies" for details.

        Important: The original "connectivity = harder to capture" rationale
        is incorrect. RingRift has no Go-style capture mechanics. Markers
        cannot be surrounded/captured - only flipped by passing stacks.

        Optimized to use FastGeometry pre-computed offset tables.
        """
        score = 0.0
        board = game_state.board
        board_type = board.type
        markers = board.markers
        collapsed = board.collapsed_spaces
        stacks = board.stacks

        # Get number of directions for this board type
        num_directions = 8 if board_type != BoardType.HEXAGONAL else 6

        my_markers = [
            m for m in markers.values()
            if m.player == self.player_number
        ]

        for marker in my_markers:
            start_key = marker.position.to_key()
            for dir_idx in range(num_directions):
                # Use ultra-fast pre-computed offset lookup
                key1 = self._fast_geo.offset_key_fast(start_key, dir_idx, 1, board_type)
                if key1 is None:
                    continue

                key2 = self._fast_geo.offset_key_fast(start_key, dir_idx, 2, board_type)

                has_m1 = (
                    key1 in markers and
                    markers[key1].player == self.player_number
                )
                has_m2 = (
                    key2 is not None and
                    key2 in markers and
                    markers[key2].player == self.player_number
                )

                if has_m1:
                    score += self.WEIGHT_CONNECTED_NEIGHBOR  # Connected neighbor
                # Gap of 1, potential to connect if gap is empty or flippable
                if (has_m2 and not has_m1
                        and key1 not in collapsed and key1 not in stacks):
                    score += self.WEIGHT_GAP_POTENTIAL

        return score * self.WEIGHT_LINE_CONNECTIVITY

    def _evaluate_territory_safety(self, game_state: GameState) -> float:
        """
        Evaluate safety of potential territories.
        Penalizes if opponent has stacks near our marker clusters.
        """
        score = 0.0
        board = game_state.board

        # Identify clusters of markers (simplified)
        # For each of my markers, check distance to nearest opponent stack
        my_markers = [
            m for m in board.markers.values()
            if m.player == self.player_number
        ]
        opponent_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player != self.player_number
        ]

        if not my_markers or not opponent_stacks:
            return 0.0

        for marker in my_markers:
            min_dist = float('inf')
            for stack in opponent_stacks:
                # Manhattan distance approximation
                dist = (
                    abs(marker.position.x - stack.position.x) +
                    abs(marker.position.y - stack.position.y)
                )
                if (marker.position.z is not None and
                        stack.position.z is not None):
                    dist += abs(marker.position.z - stack.position.z)
                min_dist = min(min_dist, dist)

            # If opponent is very close (dist 1 or 2), penalty
            if min_dist <= 2:
                score -= (3.0 - min_dist)  # -2 for dist 1, -1 for dist 2

        return score * self.WEIGHT_TERRITORY_SAFETY

    def _evaluate_stack_mobility(self, game_state: GameState) -> float:
        """Evaluate mobility of individual stacks (relative to opponents).

        Made symmetric: computes (my_mobility - max_opponent_mobility) so
        that the evaluation sums to approximately zero across all players.
        """
        board = game_state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces

        def compute_mobility_for_player(player_num: int) -> float:
            """Compute raw mobility score for a player."""
            player_stacks = [
                s for s in stacks.values()
                if s.controlling_player == player_num
            ]
            mobility = 0.0
            for stack in player_stacks:
                pos_key = stack.position.to_key()
                adjacent_keys = self._get_adjacent_keys(pos_key, board_type)
                valid_moves = 0
                for adj_key in adjacent_keys:
                    if adj_key in collapsed:
                        continue
                    if adj_key in stacks:
                        target = stacks[adj_key]
                        if (target.controlling_player != player_num
                                and stack.cap_height >= target.cap_height):
                            valid_moves += 1
                        continue
                    valid_moves += 1
                mobility += valid_moves
                if valid_moves == 0:
                    mobility -= self.WEIGHT_BLOCKED_STACK_PENALTY
            return mobility

        my_mobility = compute_mobility_for_player(self.player_number)

        # Find max opponent mobility for symmetric evaluation
        max_opponent_mobility = 0.0
        for p in game_state.players:
            if p.player_number != self.player_number:
                opp_mobility = compute_mobility_for_player(p.player_number)
                max_opponent_mobility = max(max_opponent_mobility, opp_mobility)

        # Symmetric: advantage over best opponent
        return (my_mobility - max_opponent_mobility) * self.WEIGHT_STACK_MOBILITY

    def _evaluate_forced_elimination_risk(
        self,
        game_state: GameState,
    ) -> float:
        """
        Penalise positions where we control many stacks but have very few
        real actions (moves or placements), indicating forced-elimination risk.
        """
        board = game_state.board
        my_stacks = [
            s for s in board.stacks.values()
            if s.controlling_player == self.player_number
        ]
        controlled_stacks = len(my_stacks)
        if controlled_stacks == 0:
            return 0.0

        approx_actions = self._approx_real_actions_for_player(
            game_state,
            self.player_number,
        )
        ratio = approx_actions / max(1, controlled_stacks)

        if ratio >= 2.0:
            risk_factor = 0.0
        elif ratio >= 1.0:
            risk_factor = 2.0 - ratio
        else:
            risk_factor = 1.0 + (1.0 - ratio)

        return -risk_factor * self.WEIGHT_FORCED_ELIMINATION_RISK

    def _evaluate_lps_action_advantage(
        self,
        game_state: GameState,
    ) -> float:
        """
        Last-player-standing action advantage heuristic.

        In 3+ player games, reward being one of the few players with real
        actions left and penalise being the only player without actions.
        """
        players = game_state.players
        if len(players) <= 2:
            return 0.0

        actions = {
            p.player_number: self._approx_real_actions_for_player(
                game_state,
                p.player_number,
            )
            for p in players
        }
        self_actions = actions.get(self.player_number, 0)
        self_has = self_actions > 0

        opp_with_action = sum(
            1
            for p in players
            if p.player_number != self.player_number
            and actions.get(p.player_number, 0) > 0
        )

        if not self_has:
            advantage = -1.0
        else:
            total_opponents = len(players) - 1
            if total_opponents <= 0:
                advantage = 0.0
            else:
                inactive_fraction = (
                    total_opponents - opp_with_action
                ) / float(total_opponents)
                advantage = inactive_fraction

        return advantage * self.WEIGHT_LPS_ACTION_ADVANTAGE

    def _evaluate_multi_leader_threat(
        self,
        game_state: GameState,
    ) -> float:
        """
        Multi-player leader threat heuristic.

        In 3+ player games, penalise positions where a single opponent is
        much closer to victory than the other opponents.
        """
        players = game_state.players
        if len(players) <= 2:
            return 0.0

        prox_by_player = {
            p.player_number: self._victory_proximity_base_for_player(game_state, p)
            for p in players
        }

        opp_prox = [
            prox_by_player[p.player_number]
            for p in players
            if p.player_number != self.player_number
        ]
        if len(opp_prox) < 2:
            return 0.0

        opp_prox_sorted = sorted(opp_prox, reverse=True)
        opp_top1 = opp_prox_sorted[0]
        opp_top2 = opp_prox_sorted[1]

        leader_gap = max(0.0, opp_top1 - opp_top2)

        return -leader_gap * self.WEIGHT_MULTI_LEADER_THREAT

    def _get_adjacent_positions(
        self,
        position: Position,
        game_state: GameState
    ) -> list[Position]:
        """Get adjacent positions around a position.

        Note: This method returns Position objects for compatibility.
        For performance-critical code, use _get_adjacent_keys directly.
        """
        return BoardGeometry.get_adjacent_positions(
            position,
            game_state.board.type,
            game_state.board.size
        )

    def _get_adjacent_keys(self, pos_key: str, board_type) -> list[str]:
        """Get adjacent position keys using pre-computed FastGeometry tables.

        This is significantly faster than _get_adjacent_positions as it
        avoids creating Position objects.
        """
        return self._fast_geo.get_adjacent_keys(pos_key, board_type)

    def evaluate_swap_opening_bonus(
        self,
        game_state: GameState,
    ) -> float:
        """
        Evaluate the strategic value of P1's opening position.

        This method computes a bonus score that represents how valuable it
        would be for P2 to swap into P1's position. The bonus is based on:

        - How many P1 stacks occupy center positions (highest weight)
        - How many P1 stacks are adjacent to center positions
        - Total stack height of P1's stacks

        This method is called from ``select_move()`` when evaluating
        SWAP_SIDES moves to add a strategic bonus that makes the AI more
        likely to swap into advantageous P1 openings.

        Returns
        -------
        float
            Swap opening bonus score (0.0 if swap is not strategically
            valuable, positive otherwise based on P1's opening strength).
        """
        return self.swap_evaluator.evaluate_swap_opening_bonus(game_state)

    def _is_corner_position(
        self,
        position: Position,
        game_state: GameState,
    ) -> bool:
        """Check if a position is at a board corner.

        Delegates to SwapEvaluator for position classification.
        """
        return self.swap_evaluator.is_corner_position(position, game_state)

    def _is_edge_position(
        self,
        position: Position,
        game_state: GameState,
    ) -> bool:
        """Check if a position is on an edge (but not a corner).

        Delegates to SwapEvaluator for position classification.
        """
        return self.swap_evaluator.is_edge_position(position, game_state)

    def _is_strategic_diagonal_position(
        self,
        position: Position,
        game_state: GameState,
    ) -> bool:
        """Check if a position is on a key diagonal (one step from center).

        Delegates to SwapEvaluator for position classification.
        """
        return self.swap_evaluator.is_strategic_diagonal_position(
            position, game_state
        )

    def compute_opening_strength(
        self,
        position: Position,
        game_state: GameState,
    ) -> float:
        """Compute opening strength score for a position (0-1 scale).

        Delegates to SwapEvaluator. See SwapEvaluator.compute_opening_strength
        for full documentation.
        """
        return self.swap_evaluator.compute_opening_strength(position, game_state)

    def evaluate_swap_with_classifier(
        self,
        game_state: GameState,
    ) -> float:
        """Evaluate swap decision using the Opening Position Classifier.

        Delegates to SwapEvaluator. See SwapEvaluator.evaluate_swap_with_classifier
        for full documentation.
        """
        return self.swap_evaluator.evaluate_swap_with_classifier(game_state)

    def _evaluate_recovery_potential(
        self,
        game_state: GameState,
    ) -> float:
        """
        Evaluate recovery potential for all players (v1.6).

        Recovery (RR-CANON-R110–R115) allows temporarily eliminated players to
        slide markers to form lines, paying costs with buried ring extraction.
        This heuristic captures:

        1. Value of having recovery available as a strategic option
        2. Threat from opponents who have recovery potential
        3. Value of buried rings as recovery resources

        Returns:
            Score representing recovery strategic value (positive = good for us)
        """
        score = 0.0

        # Check our recovery status
        my_eligible = is_eligible_for_recovery(game_state, self.player_number)
        my_buried = count_buried_rings(
            game_state.board, self.player_number
        )

        if my_eligible:
            # Bonus for having recovery available
            score += self.WEIGHT_RECOVERY_ELIGIBILITY

            # Additional bonus if we actually have recovery moves
            if has_any_recovery_move(game_state, self.player_number):
                score += self.WEIGHT_RECOVERY_POTENTIAL
        else:
            # Small bonus for buried rings even if not currently eligible
            # (potential future recovery resource)
            score += my_buried * self.WEIGHT_BURIED_RING_VALUE * 0.3

        # Evaluate opponent recovery threats
        for player in game_state.players:
            if player.player_number == self.player_number:
                continue

            opp_eligible = is_eligible_for_recovery(
                game_state, player.player_number
            )
            if opp_eligible:
                # Penalty for opponent having recovery potential
                score -= self.WEIGHT_RECOVERY_ELIGIBILITY * 0.5

                # Extra penalty if opponent actually has recovery moves
                if has_any_recovery_move(game_state, player.player_number):
                    score -= self.WEIGHT_RECOVERY_THREAT

        return score

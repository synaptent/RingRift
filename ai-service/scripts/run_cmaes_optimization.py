#!/usr/bin/env python
"""CMA-ES optimization for HeuristicAI weights.

This script uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
to optimize heuristic weights through competitive self-play. The fitness
function evaluates candidate weights by playing games against a baseline
and measuring win rate, reusing the same `evaluate_fitness(...)` harness
that is also called from `run_genetic_heuristic_search.py`.

Usage examples
--------------

From the ``ai-service`` root::

    # Basic run with default hyperparameters
    python scripts/run_cmaes_optimization.py \
        --generations 50 \
        --population-size 20 \
        --games-per-eval 10 \
        --output logs/cmaes/optimized_weights.json

    # Quick test run
    python scripts/run_cmaes_optimization.py \
        --generations 3 \
        --population-size 4 \
        --games-per-eval 2 \
        --output logs/cmaes/test_weights.json

    # With custom baseline weights file
    python scripts/run_cmaes_optimization.py \
        --generations 50 \
        --population-size 20 \
        --games-per-eval 10 \
        --baseline logs/cmaes/previous_best.json \
        --output logs/cmaes/optimized_weights.json

    # Using multi-start evaluation from a state pool
    python scripts/run_cmaes_optimization.py \
        --generations 4 \
        --population-size 16 \
        --games-per-eval 32 \
        --board square8 \
        --eval-mode multi-start \
        --state-pool-id v1 \
        --output logs/cmaes/optimized_weights.multistart.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import time

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cma
except ImportError:  # pragma: no cover - handled via CLI guard
    # For library consumers (tests, helper scripts) we keep the module
    # importable even when `cma` is not installed, since core helpers like
    # `evaluate_fitness` and `evaluate_fitness_over_boards` do not depend on
    # the CMA-ES library. The CLI entrypoint will perform a runtime check and
    # emit a clear error message if CMA-ES is actually required.
    cma = None  # type: ignore[assignment]

from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MoveType,
    Player,
    TimeControl,
)
from app.ai.heuristic_ai import HeuristicAI  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)
from app.rules.default_engine import (  # type: ignore  # noqa: E402
    DefaultRulesEngine,
)
from app.training.eval_pools import (  # type: ignore  # noqa: E402
    load_state_pool,
)
from app.training.env import (  # type: ignore  # noqa: E402
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD,
)
from app.utils.progress_reporter import (  # noqa: E402
    OptimizationProgressReporter,
    ProgressReporter,
)


FitnessDebugCallback = Callable[
    [HeuristicWeights, HeuristicWeights, BoardType, Dict[str, Any]],
    None,
]


# Canonical order of weight keys for flattening/unflattening.
# This list is kept in lockstep with HEURISTIC_WEIGHT_KEYS defined in
# app.ai.heuristic_weights so that all optimisation tooling shares a single
# source of truth for the weight vector layout.
WEIGHT_KEYS: List[str] = list(HEURISTIC_WEIGHT_KEYS)

# Map CLI board names to BoardType enums for evaluation wiring.
BOARD_NAME_TO_TYPE: Dict[str, BoardType] = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hex": BoardType.HEXAGONAL,
}

# Lightweight diagnostic counter for pie-rule usage during a single
# evaluate_fitness() run. This is reset at the start of each evaluation
# and incremented whenever either AI selects a SWAP_SIDES move.
SWAP_SIDES_MOVE_COUNTER: int = 0


def weights_to_array(weights: HeuristicWeights) -> np.ndarray:
    """Flatten HeuristicWeights dict to a 1D numpy array.

    Parameters
    ----------
    weights:
        Dictionary of weight name -> value.

    Returns
    -------
    np.ndarray
        1D array of weight values in canonical order (WEIGHT_KEYS).
    """
    return np.array([weights[key] for key in WEIGHT_KEYS], dtype=np.float64)


def array_to_weights(arr: np.ndarray) -> HeuristicWeights:
    """Reconstruct HeuristicWeights dict from a 1D numpy array.

    Parameters
    ----------
    arr:
        1D array of weight values in canonical order (WEIGHT_KEYS).

    Returns
    -------
    HeuristicWeights
        Dictionary mapping weight names to values.
    """
    return {key: float(arr[i]) for i, key in enumerate(WEIGHT_KEYS)}


def create_heuristic_ai_with_weights(
    player_number: int,
    weights: HeuristicWeights,
    *,
    difficulty: int = 5,
    randomness: float = 0.0,
    rng_seed: Optional[int] = None,
    heuristic_eval_mode: Optional[str] = None,
) -> HeuristicAI:
    """Create a HeuristicAI instance with custom weights applied.

    Parameters
    ----------
    player_number:
        Player number (1 or 2).
    weights:
        Custom weights to apply to the AI.
    difficulty:
        AI difficulty level.
    randomness:
        Randomness parameter used for move selection (e.g. tie-breaking).
    rng_seed:
        Optional deterministic RNG seed for this AI instance.
    heuristic_eval_mode:
        Optional evaluation mode forwarded to HeuristicAI via
        ``AIConfig.heuristic_eval_mode``. Training harnesses use this to
        select between ``"full"`` and ``"light"`` heuristic evaluation
        based on board type.

    Returns
    -------
    HeuristicAI
        AI instance with the specified weights.
    """
    ai = HeuristicAI(
        player_number,
        AIConfig(
            difficulty=difficulty,
            think_time=0,
            randomness=randomness,
            rngSeed=rng_seed,
            heuristic_profile_id=None,
            heuristic_eval_mode=heuristic_eval_mode,
        ),
    )
    # Override weights on the instance
    for name, value in weights.items():
        setattr(ai, name, value)
    return ai


def create_game_state(
    board_type: BoardType = BoardType.SQUARE8,
) -> GameState:
    """Create an initial game state for self-play.

    Parameters
    ----------
    board_type:
        Type of board to use.

    Returns
    -------
    GameState
        Fresh game state ready for play.
    """
    if board_type == BoardType.SQUARE8:
        size = 8
        rings_per_player = 18
    elif board_type == BoardType.SQUARE19:
        size = 19
        rings_per_player = 36
    elif board_type == BoardType.HEXAGONAL:
        size = 11
        rings_per_player = 36
    else:
        size = 8
        rings_per_player = 18

    now = datetime.now()

    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    players = [
        Player(
            id=f"player{i}",
            username=f"AI {i}",
            type="ai",
            playerNumber=i,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in [1, 2]
    ]

    return GameState(
        id="cmaes-optimization",
        boardType=board_type,
        rngSeed=None,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=19,  # >50% of 36 total rings
        territoryVictoryThreshold=33,  # >50% of 64 spaces
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


def play_single_game_from_state(
    initial_state: GameState,
    candidate_weights: HeuristicWeights,
    opponent_weights: HeuristicWeights,
    candidate_plays_first: bool,
    max_moves: int = 500,
    *,
    randomness: float = 0.0,
    rng_seed_base: Optional[int] = None,
    heuristic_eval_mode: Optional[str] = None,
    per_move_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[int, int]:
    """Play a single game from a provided initial :class:`GameState`.

    This helper is shared by both the classic initial-position evaluation
    path and the multi-start evaluation path. It is deliberately free of any
    opponent-scheduling or state-pool concerns so that callers remain in
    control of those policies.
    """
    global SWAP_SIDES_MOVE_COUNTER
    # Derive per-player RNG seeds in a deterministic but side-independent way.
    use_randomness = randomness > 0.0
    if use_randomness and rng_seed_base is not None:
        candidate_seed: Optional[int] = rng_seed_base
        opponent_seed: Optional[int] = rng_seed_base + 1
    else:
        candidate_seed = None
        opponent_seed = None

    if candidate_plays_first:
        candidate_player = 1
        opponent_player = 2
    else:
        candidate_player = 2
        opponent_player = 1

    candidate_ai = create_heuristic_ai_with_weights(
        candidate_player,
        candidate_weights,
        randomness=randomness if use_randomness else 0.0,
        rng_seed=candidate_seed,
        heuristic_eval_mode=heuristic_eval_mode,
    )
    opponent_ai = create_heuristic_ai_with_weights(
        opponent_player,
        opponent_weights,
        randomness=randomness if use_randomness else 0.0,
        rng_seed=opponent_seed,
        heuristic_eval_mode=heuristic_eval_mode,
    )

    game_state = initial_state
    rules_engine = DefaultRulesEngine()
    move_count = 0

    while (
        game_state.game_status == GameStatus.ACTIVE and move_count < max_moves
    ):
        current_player = game_state.current_player
        current_ai = (
            candidate_ai if current_player == candidate_player else opponent_ai
        )
        current_ai.player_number = current_player

        move = current_ai.select_move(game_state)
        if not move:
            # No valid moves - opponent wins
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 2 if current_player == 1 else 1
            break

        if move.type == MoveType.SWAP_SIDES:
            SWAP_SIDES_MOVE_COUNTER += 1

        game_state = rules_engine.apply_move(game_state, move)
        move_count += 1

        # Optional per-move callback for fine-grained progress reporting.
        if per_move_callback is not None:
            per_move_callback(move_count)

    if game_state.game_status != GameStatus.FINISHED:
        # Draw due to move limit
        return (0, move_count)

    winner = game_state.winner
    if winner == candidate_player:
        return (1, move_count)
    elif winner is not None:
        return (-1, move_count)
    return (0, move_count)


def play_single_game(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    candidate_plays_first: bool,
    board_type: BoardType = BoardType.SQUARE8,
    max_moves: int = 500,
    *,
    randomness: float = 0.0,
    rng_seed_base: Optional[int] = None,
    heuristic_eval_mode: Optional[str] = None,
) -> Tuple[int, int]:
    """Play a single game between candidate and baseline weights.

    This thin wrapper constructs a fresh initial state for the given board
    type and delegates to :func:`play_single_game_from_state`. It preserves
    the historical semantics used by the training harness.
    """
    initial_state = create_game_state(board_type)
    return play_single_game_from_state(
        initial_state=initial_state,
        candidate_weights=candidate_weights,
        opponent_weights=baseline_weights,
        candidate_plays_first=candidate_plays_first,
        max_moves=max_moves,
        randomness=randomness,
        rng_seed_base=rng_seed_base,
        heuristic_eval_mode=heuristic_eval_mode,
    )


def evaluate_fitness(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    games_per_eval: int,
    board_type: BoardType = BoardType.SQUARE8,
    verbose: bool = False,
    *,
    opponent_mode: str = "baseline-only",
    incumbent_weights: Optional[HeuristicWeights] = None,
    max_moves: int = 500,
    debug_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    eval_mode: str = "initial-only",
    state_pool_id: Optional[str] = None,
    eval_randomness: float = 0.0,
    seed: Optional[int] = None,
    debug_callback: Optional[FitnessDebugCallback] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> float:
    """Evaluate fitness of candidate weights against one or more opponents.

    Fitness is computed as (wins + 0.5 * draws) / total_games.
    Higher values indicate stronger performance.

    Parameters
    ----------
    candidate_weights:
        Weights for the candidate AI.
    baseline_weights:
        Weights for the baseline AI (B0).
    games_per_eval:
        Total number of games to play per evaluation.
    board_type:
        Type of board to use.
    verbose:
        If True, print game results for debugging.
    opponent_mode:
        - "baseline-only": play all games against B0 (current baseline).
        - "baseline-plus-incumbent": split games between B0 and B1
          (current global best candidate), using an approximate 2:1 ratio
          (e.g. 80 vs B0, 40 vs B1 when games_per_eval=120).
    incumbent_weights:
        Weights for the incumbent opponent (B1). If None and
        opponent_mode == "baseline-plus-incumbent", falls back to B0.
    max_moves:
        Maximum moves per game before declaring a draw.
    eval_mode:
        - "initial-only": evaluate from the empty starting position only
          (historical behaviour; default).
        - "multi-start": evaluate from a fixed pool of mid-game states
          loaded via :mod:`app.training.eval_pools`.
    state_pool_id:
        Logical identifier for the evaluation state pool when using
        ``eval_mode='multi-start'``. If None, defaults to "v1".
    eval_randomness:
        Randomness parameter forwarded to HeuristicAI instances. ``0.0``
        keeps evaluation fully deterministic; small positive values enable
        controlled stochastic tie-breaking.
    seed:
        Optional base seed used to derive per-game RNG seeds when
        ``eval_randomness > 0.0``. When ``None``, a default base seed of 0
        is used.
    debug_hook:
        Optional legacy callback that receives a flat stats dict for this
        evaluation. It is called once per candidate after all games have
        been played. Kept for backward compatibility with existing
        diagnostics.
    debug_callback:
        Optional structured callback invoked once per candidate after
        evaluation. The callback receives:

        - ``candidate_weights``: the candidate weight dict passed in.
        - ``baseline_weights``: the baseline weight dict used for this run.
        - ``board_type``: the board used for evaluation.
        - ``stats``: a dict containing keys such as ``wins``, ``draws``,
          ``losses``, ``games``, ``games_per_eval``, ``fitness``,
          ``avg_moves``, and ``weight_l2_to_baseline``.

    Returns
    -------
    float
        Fitness score in [0, 1].
    """
    global SWAP_SIDES_MOVE_COUNTER

    if games_per_eval <= 0:
        return 0.0

    # Only track timing when progress reporting is enabled to avoid
    # overhead on hot evaluation paths.
    start_time: Optional[float] = None
    if progress_reporter is not None:
        start_time = time.time()

    if eval_mode not in ("initial-only", "multi-start"):
        raise ValueError(f"Unknown eval_mode: {eval_mode!r}")

    # Board-dependent heuristic evaluation mode: Square8 uses the full
    # structural evaluator, while large boards default to the lightweight
    # heuristic mode for faster evaluation.
    heuristic_eval_mode = TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.get(
        board_type,
        "full",
    )

    wins = 0
    draws = 0
    losses = 0
    total_moves = 0
    SWAP_SIDES_MOVE_COUNTER = 0

    # L2 distance between candidate and baseline weights in the canonical
    # HEURISTIC_WEIGHT_KEYS order. This is used purely for diagnostics and
    # does not affect optimisation behaviour.
    weight_l2 = 0.0
    try:
        candidate_vec = np.array(
            [float(candidate_weights[k]) for k in HEURISTIC_WEIGHT_KEYS],
            dtype=float,
        )
        baseline_vec = np.array(
            [float(baseline_weights[k]) for k in HEURISTIC_WEIGHT_KEYS],
            dtype=float,
        )
        weight_l2 = float(np.linalg.norm(candidate_vec - baseline_vec))
    except (KeyError, TypeError, ValueError):
        # In tests we sometimes use sentinel dicts with non-numeric values;
        # fall back to 0.0 rather than raising so that diagnostics remain
        # best-effort only.
        weight_l2 = 0.0

    # Build opponent schedule
    if opponent_mode == "baseline-only":
        opponents: List[HeuristicWeights] = [baseline_weights] * games_per_eval
    elif opponent_mode == "baseline-plus-incumbent":
        # Use incumbent if provided; otherwise fall back to baseline
        # for early generations.
        if incumbent_weights is None:
            incumbent_weights = baseline_weights

        # 2:1 split between baseline (B0) and incumbent (B1),
        # e.g. 80/40 for 120 games.
        games_vs_baseline = max(1, (2 * games_per_eval) // 3)
        if games_per_eval > 1:
            games_vs_baseline = min(games_vs_baseline, games_per_eval - 1)
        games_vs_incumbent = games_per_eval - games_vs_baseline

        opponents = (
            [baseline_weights] * games_vs_baseline
            + [incumbent_weights] * games_vs_incumbent
        )
    else:
        raise ValueError(f"Unknown opponent_mode: {opponent_mode!r}")

    pool_states: Optional[List[GameState]] = None
    pool_id = state_pool_id or "v1"
    if eval_mode == "multi-start":
        pool_states = load_state_pool(
            board_type=board_type,
            pool_id=pool_id,
            max_states=games_per_eval,
        )
        if not pool_states:
            raise ValueError(
                f"State pool is empty for board_type={board_type}, "
                f"pool_id={pool_id}"
            )

    # Randomness/seeding configuration: keep historical behaviour when
    # eval_randomness == 0.0 (no AI randomness and rng seeds unset), and use
    # a deterministic per-game seed schedule otherwise.
    use_randomness = eval_randomness > 0.0
    base_seed = 0 if seed is None else seed

    for i in range(games_per_eval):
        # Alternate who plays first for fairness across the overall schedule
        candidate_first = i % 2 == 0
        opponent_weights_for_game = opponents[i]
        game_seed = base_seed + i

        if eval_mode == "initial-only":
            result, move_count = play_single_game(
                candidate_weights,
                opponent_weights_for_game,
                candidate_first,
                board_type,
                max_moves,
                randomness=eval_randomness if use_randomness else 0.0,
                rng_seed_base=game_seed if use_randomness else None,
                heuristic_eval_mode=heuristic_eval_mode,
            )
        else:
            assert pool_states is not None  # for type-checkers
            base_state = pool_states[i % len(pool_states)]
            initial_state = base_state.model_copy(deep=True)

            move_progress_callback: Optional[Callable[[int], None]] = None
            if progress_reporter is not None and start_time is not None:

                def _per_move_progress_callback(
                    _move_count: int,
                    *,
                    _game_index: int = i,
                    _progress: ProgressReporter = progress_reporter,
                    _start_time: float = start_time,
                ) -> None:
                    # Emit time-based progress during long games (e.g. 19x19),
                    # without changing the completed-games counter.
                    games_completed = _game_index
                    elapsed = time.time() - _start_time
                    games_so_far = (
                        games_completed if games_completed > 0 else 1
                    )
                    avg_moves_so_far = (
                        total_moves / games_so_far
                        if total_moves > 0
                        else 0.0
                    )
                    sec_per_game = (
                        elapsed / games_so_far if games_so_far > 0 else 0.0
                    )
                    sec_per_move = (
                        elapsed / total_moves if total_moves > 0 else 0.0
                    )
                    _progress.update(
                        completed=games_completed,
                        extra_metrics={
                            "wins": wins,
                            "draws": draws,
                            "losses": losses,
                            "avg_moves": avg_moves_so_far,
                            "sec_per_game": sec_per_game,
                            "sec_per_move": sec_per_move,
                        },
                    )

                move_progress_callback = _per_move_progress_callback

            result, move_count = play_single_game_from_state(
                initial_state=initial_state,
                candidate_weights=candidate_weights,
                opponent_weights=opponent_weights_for_game,
                candidate_plays_first=candidate_first,
                max_moves=max_moves,
                randomness=eval_randomness if use_randomness else 0.0,
                rng_seed_base=game_seed if use_randomness else None,
                heuristic_eval_mode=heuristic_eval_mode,
                per_move_callback=move_progress_callback,
            )

        total_moves += move_count
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

        # Optional per-game progress reporting. This is only active when a
        # ProgressReporter instance is passed in, so default callers are
        # unaffected. We report games completed along with lightweight
        # diagnostics that help approximate throughput.
        if progress_reporter is not None and start_time is not None:
            games_completed = i + 1
            elapsed = time.time() - start_time
            avg_moves_so_far = (
                total_moves / games_completed if games_completed > 0 else 0.0
            )
            sec_per_game = (
                elapsed / games_completed if games_completed > 0 else 0.0
            )
            sec_per_move = elapsed / total_moves if total_moves > 0 else 0.0

            progress_reporter.update(
                completed=games_completed,
                extra_metrics={
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "avg_moves": avg_moves_so_far,
                    "sec_per_game": sec_per_game,
                    "sec_per_move": sec_per_move,
                },
            )

    avg_moves = total_moves / games_per_eval if games_per_eval > 0 else 0
    swap_sides_moves = SWAP_SIDES_MOVE_COUNTER

    if verbose:
        print(
            f"    Games: W={wins}, D={draws}, L={losses}, "
            f"avg_moves={avg_moves:.0f}, "
            f"weight_l2={weight_l2:.3f}, "
            f"swap_sides_moves={swap_sides_moves}"
        )

    fitness = (wins + 0.5 * draws) / games_per_eval

    # Emit a final per-evaluation summary on the reporter when enabled.
    if progress_reporter is not None and start_time is not None:
        elapsed = time.time() - start_time
        sec_per_game = elapsed / games_per_eval if games_per_eval > 0 else 0.0
        sec_per_move = elapsed / total_moves if total_moves > 0 else 0.0

        progress_reporter.finish(
            extra_metrics={
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "avg_moves": avg_moves,
                "sec_per_game": sec_per_game,
                "sec_per_move": sec_per_move,
            }
        )

    # Aggregate per-candidate evaluation statistics for optional diagnostics.
    stats: Dict[str, Any] = {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        # Preserve the historical key while also exposing a shorter alias
        # for new instrumentation.
        "games_per_eval": games_per_eval,
        "games": games_per_eval,
        "fitness": fitness,
        "avg_moves": avg_moves,
        # Historical name plus an explicit "to_baseline" alias for clarity.
        "weight_l2": weight_l2,
        "weight_l2_to_baseline": weight_l2,
        "eval_mode": eval_mode,
        "state_pool_id": pool_id,
        "eval_randomness": eval_randomness,
        "seed": seed,
        "opponent_mode": opponent_mode,
        # Pie-rule diagnostics: total SWAP_SIDES selections during this
        # evaluation, plus a per-game average for quick sanity checks.
        "swap_sides_moves": swap_sides_moves,
        "swap_sides_moves_per_game": (
            swap_sides_moves / games_per_eval if games_per_eval > 0 else 0.0
        ),
    }

    if debug_hook is not None:
        debug_hook(stats)

    if debug_callback is not None:
        debug_callback(candidate_weights, baseline_weights, board_type, stats)

    return fitness


def evaluate_fitness_over_boards(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    games_per_eval: int,
    boards: List[BoardType],
    *,
    opponent_mode: str = "baseline-only",
    max_moves: int = 200,
    verbose: bool = False,
    debug_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    eval_mode: str = "initial-only",
    state_pool_id: Optional[str] = None,
    seed: Optional[int] = None,
    eval_randomness: float = 0.0,
    debug_callback: Optional[FitnessDebugCallback] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: Optional[float] = None,
    enable_eval_progress: bool = True,
) -> Tuple[float, Dict[BoardType, float]]:
    """Evaluate candidate fitness averaged over multiple board types.

    This helper is a thin wrapper over :func:`evaluate_fitness` that evaluates
    the same candidate weights on a list of boards and returns:

    - the arithmetic mean of the per-board fitness values, and
    - a mapping from BoardType to per-board fitness.

    It is intentionally agnostic to the choice of evaluation mode and state
    pools; callers control those via ``eval_mode`` and ``state_pool_id``.
    """
    if not boards:
        raise ValueError("boards must contain at least one BoardType")

    per_board_fitness: Dict[BoardType, float] = {}

    for idx, board_type in enumerate(boards):
        board_seed = None if seed is None else seed + idx * 10_000

        # Optional per-board progress reporter. When enabled, we track
        # progress at the game level for this (candidate, board) pair.
        board_progress: Optional[ProgressReporter] = None
        if progress_label is not None and enable_eval_progress:
            context = f"{progress_label} | board={board_type.value}"
            if progress_interval_sec is None:
                interval = 10.0
            else:
                interval = progress_interval_sec
            board_progress = ProgressReporter(
                total_units=games_per_eval,
                unit_name="games",
                report_interval_sec=interval,
                context_label=context,
            )

        def _tag_stats(
            stats: Dict[str, Any],
            *,
            _board_type: BoardType = board_type,
            _board_index: int = idx,
            _board_seed: Optional[int] = board_seed,
        ) -> Dict[str, Any]:
            tagged = dict(stats)
            tagged["board_type"] = _board_type
            tagged["board_index"] = _board_index
            tagged["board_seed"] = _board_seed
            return tagged

        if debug_hook is not None:
            def board_debug_hook(
                stats: Dict[str, Any],
                *,
                _tag=_tag_stats,
            ) -> None:
                debug_hook(_tag(stats))
        else:
            board_debug_hook = None  # type: ignore[assignment]

        if debug_callback is not None:
            def board_debug_callback(
                candidate_w: HeuristicWeights,
                baseline_w: HeuristicWeights,
                bt: BoardType,
                stats: Dict[str, Any],
                *,
                _tag=_tag_stats,
            ) -> None:
                debug_callback(candidate_w, baseline_w, bt, _tag(stats))
        else:
            board_debug_callback = None  # type: ignore[assignment]

        fitness = evaluate_fitness(
            candidate_weights=candidate_weights,
            baseline_weights=baseline_weights,
            games_per_eval=games_per_eval,
            board_type=board_type,
            verbose=verbose,
            opponent_mode=opponent_mode,
            max_moves=max_moves,
            debug_hook=board_debug_hook,
            eval_mode=eval_mode,
            state_pool_id=state_pool_id,
            eval_randomness=eval_randomness,
            seed=board_seed,
            debug_callback=board_debug_callback,
            progress_reporter=board_progress,
        )
        per_board_fitness[board_type] = fitness

    aggregate = float(
        sum(per_board_fitness.values()) / float(len(per_board_fitness))
    )
    return aggregate, per_board_fitness


def evaluate_fitness_multiplayer(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    num_players: int,
    games_per_eval: int,
    boards: List[BoardType],
    state_pool_id: str = "v1",
    seed: Optional[int] = None,
) -> float:
    """Minimal rank-based fitness for 3p/4p heuristic evaluation.

    This helper is intentionally lightweight and is **not** wired into any
    CLI entrypoints yet. It exists so that future CMA-ES / GA experiments
    can evaluate candidate weights in genuine multi-player settings without
    redesigning the optimisation loop.

    Behaviour (stub, subject to refinement):

    - For each requested game, choose a board from ``boards`` in round-robin
      order and load the corresponding multi-player state pool via
      :func:`app.training.eval_pools.load_state_pool`, filtering by
      ``num_players``.
    - Sample an initial state from the pool and deep-copy it.
    - Randomly assign the candidate to one of the ``num_players`` seats;
      all other seats use ``baseline_weights``.
    - Play a single game from that state using :class:`DefaultRulesEngine`,
      with all players controlled by :class:`HeuristicAI` instances created
      via :func:`create_heuristic_ai_with_weights`.
    - At the end of the game, assign a score from the candidate's
      perspective:

        * +1 if the candidate is the sole winner.
        *  0 if there is no winner (draw or unresolved).
        * -1 otherwise.

    The function returns the arithmetic mean of these per-game scores in
    the range [-1, 1].

    Notes
    -----
    - This stub deliberately uses a simple win/lose/draw scoring scheme
      rather than a full ordinal ranking over all players. It is sufficient
      to wire multi-player evaluation into future optimisation harnesses
      without committing to a final scoring design.
    - Callers are responsible for selecting appropriate multi-player pools
      (for example, ``square19_3p_pool_v1`` or ``hex_4p_pool_v1``) via the
      ``state_pool_id`` parameter.
    """
    # Uses load_state_pool imported at module level for evaluation pools.

    if num_players <= 2:
        raise ValueError(
            "evaluate_fitness_multiplayer is intended for 3p/4p games; "
            f"got num_players={num_players!r}"
        )
    if games_per_eval <= 0:
        return 0.0
    if not boards:
        raise ValueError("boards must contain at least one BoardType")

    # Seed numpy RNG for deterministic but reproducible seat assignments.
    rng = np.random.default_rng(seed)

    # Preload pools for each board so that we can reuse them across games.
    pools: Dict[BoardType, List[GameState]] = {}
    for board in boards:
        states = load_state_pool(
            board_type=board,
            pool_id=state_pool_id,
            max_states=None,
            num_players=num_players,
        )
        if not states:
            raise ValueError(
                "Multi-player state pool is empty for "
                f"board_type={board}, pool_id={state_pool_id!r}, "
                f"num_players={num_players}"
            )
        pools[board] = states

    rules_engine = DefaultRulesEngine()

    total_score = 0.0
    games_played = 0

    for game_index in range(games_per_eval):
        board = boards[game_index % len(boards)]
        pool_states = pools[board]
        base_state = pool_states[game_index % len(pool_states)]
        game_state = base_state.model_copy(deep=True)

        # Choose candidate seat uniformly at random.
        candidate_seat_index = int(rng.integers(0, num_players))
        seat_player = game_state.players[candidate_seat_index]
        candidate_player_number = seat_player.player_number

        # Build per-player AIs.
        # For now we use the canonical evaluation mode mapping from the
        # training environment module so that per-board heuristic feature
        # tiers stay consistent with 2-player evaluation runs.
        heuristic_eval_mode = TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.get(
            board,
            "full",
        )

        ai_by_player: Dict[int, HeuristicAI] = {}
        for p in game_state.players:
            weights = (
                candidate_weights
                if p.player_number == candidate_player_number
                else baseline_weights
            )
            ai_by_player[p.player_number] = create_heuristic_ai_with_weights(
                p.player_number,
                weights,
                randomness=0.0,
                rng_seed=None,
                heuristic_eval_mode=heuristic_eval_mode,
            )

        moves_played = 0
        while (
            game_state.game_status == GameStatus.ACTIVE
            and moves_played < 500
        ):
            current_player = game_state.current_player
            ai = ai_by_player.get(current_player)
            if ai is None:
                # If we somehow lack an AI for the current player, treat this
                # as a terminal anomaly from the candidate's perspective.
                break

            ai.player_number = current_player
            move = ai.select_move(game_state)
            if not move:
                # No move found; treat as loss for the acting player and
                # end the game.
                game_state.game_status = GameStatus.FINISHED
                game_state.winner = None
                break

            game_state = rules_engine.apply_move(game_state, move)
            moves_played += 1

        # Score from the candidate's perspective.
        winner = getattr(game_state, "winner", None)
        if winner is None:
            score = 0.0
        elif winner == candidate_player_number:
            score = 1.0
        else:
            score = -1.0

        total_score += score
        games_played += 1

    return total_score / games_played if games_played > 0 else 0.0


def run_axis_aligned_multiplayer_eval(
    profiles_dir: str,
    baseline_weights: HeuristicWeights,
    num_players: int,
    boards: List[BoardType],
    state_pool_id: str,
    games_per_eval: int,
    seed: Optional[int],
    output_path: str,
    include_baseline: bool = True,
) -> None:
    """Evaluate axis-aligned profiles in a 3p/4p setting.

    This helper loads axis-aligned profiles from ``profiles_dir`` and scores
    each profile using :func:`evaluate_fitness_multiplayer` against the
    provided baseline weights. It writes a JSON summary to ``output_path``.
    """
    baseline_keys = set(baseline_weights.keys())
    profiles: List[AxisAlignedProfile] = []

    if not os.path.isdir(profiles_dir):
        raise ValueError(
            "Axis-aligned profiles dir does not exist or is not a directory: "
            f"{profiles_dir!r}"
        )

    for name in sorted(os.listdir(profiles_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(profiles_dir, name)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        weights_obj = payload.get("weights")
        if not isinstance(weights_obj, dict):
            raise ValueError(
                "Axis-aligned profile "
                f"{path!r} is missing a 'weights' object or it is not a dict"
            )

        weight_keys = set(weights_obj.keys())
        if weight_keys != baseline_keys:
            missing = sorted(baseline_keys - weight_keys)
            extra = sorted(weight_keys - baseline_keys)
            raise ValueError(
                "Axis-aligned profile "
                f"{path!r} weight keys mismatch baseline schema: "
                f"missing={missing}, extra={extra}"
            )

        weights: HeuristicWeights = {
            str(k): float(v) for k, v in weights_obj.items()
        }

        meta = payload.get("meta") or {}
        factor = meta.get("factor")
        sign = meta.get("sign")
        if isinstance(factor, str) and isinstance(sign, str):
            pid = f"{factor}_{sign}"
        else:
            pid = os.path.splitext(name)[0]

        profiles.append(
            AxisAlignedProfile(
                id=pid,
                weights=weights,
                meta=dict(meta),
                is_baseline=False,
            )
        )

    if include_baseline:
        profiles.insert(
            0,
            AxisAlignedProfile(
                id="baseline_v1_balanced",
                weights=dict(baseline_weights),
                meta={"source": "BASE_V1_BALANCED_WEIGHTS"},
                is_baseline=True,
            ),
        )

    results: List[Dict[str, Any]] = []
    for profile in profiles:
        raw_score = evaluate_fitness_multiplayer(
            candidate_weights=profile.weights,
            baseline_weights=baseline_weights,
            num_players=num_players,
            games_per_eval=games_per_eval,
            boards=boards,
            state_pool_id=state_pool_id,
            seed=seed,
        )
        fitness = (raw_score + 1.0) / 2.0
        results.append(
            {
                "id": profile.id,
                "is_baseline": profile.is_baseline,
                "raw_score": raw_score,
                "fitness": fitness,
                "meta": profile.meta,
            }
        )

    results.sort(key=lambda r: r["fitness"], reverse=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "meta": {
            "mode": "axis_aligned_multiplayer",
            "num_players": num_players,
            "boards": [b.value for b in boards],
            "state_pool_id": state_pool_id,
            "games_per_eval": games_per_eval,
            "profiles_dir": profiles_dir,
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(
        "Axis-aligned multi-player evaluation written to "
        f"{output_path}"
    )


def load_weights_from_file(path: str) -> HeuristicWeights:
    """Load weights from a JSON file.

    Parameters
    ----------
    path:
        Path to JSON file containing weights.

    Returns
    -------
    HeuristicWeights
        Loaded weights dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both flat format and nested "weights" format
    if "weights" in data:
        return data["weights"]
    return data


def save_weights_to_file(
    weights: HeuristicWeights,
    path: str,
    generation: Optional[int] = None,
    fitness: Optional[float] = None,
) -> None:
    """Save weights to a JSON file.

    Parameters
    ----------
    weights:
        Weights dictionary to save.
    path:
        Output file path.
    generation:
        Generation number (optional metadata).
    fitness:
        Fitness score (optional metadata).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    data = {
        "weights": weights,
        "timestamp": datetime.now().isoformat(),
    }
    if generation is not None:
        data["generation"] = generation
    if fitness is not None:
        data["fitness"] = fitness

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


@dataclass
class AxisAlignedProfile:
    """Axis-aligned profile wrapper for diagnostics."""

    id: str
    weights: HeuristicWeights
    meta: Dict[str, Any]
    is_baseline: bool = False


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES optimization."""

    generations: int
    population_size: int
    games_per_eval: int
    sigma: float
    output_path: str
    baseline_path: Optional[str]
    board_type: BoardType
    checkpoint_dir: Optional[str]
    seed: Optional[int] = None
    max_moves: int = 200
    opponent_mode: str = "baseline-only"
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    resume_from: Optional[str] = None
    eval_mode: str = "initial-only"
    state_pool_id: Optional[str] = None
    eval_boards: Optional[List[BoardType]] = None
    eval_randomness: float = 0.0
    # Number of players for the evaluation games. 2 uses the standard
    # 2-player evaluation loop; 3 or 4 uses evaluate_fitness_multiplayer().
    num_players: int = 2
    # When enabled, log per-candidate evaluation details (W/D/L, fitness,
    # L2 distance) to help diagnose plateau/0.5 behaviour. Disabled by
    # default to avoid noisy logs in normal runs.
    debug_plateau: bool = False
    # Minimum interval between optimization and evaluation progress log
    # lines (seconds). This remains configurable so callers can tune
    # verbosity while still honouring the "no >60s silence" guideline for
    # long-running jobs.
    progress_interval_sec: float = 10.0
    # When False, suppress per-board evaluation progress reporters and
    # rely solely on the generation-level optimisation reporter.
    enable_eval_progress: bool = True


def run_cmaes_optimization(config: CMAESConfig) -> HeuristicWeights:
    """Run CMA-ES optimization to find optimal heuristic weights.

    This driver supports both single-opponent evaluation against a static
    baseline and multi-opponent evaluation against a pool consisting of
    the baseline (B0) and the current global incumbent (B1).

    Parameters
    ----------
    config:
        CMA-ES configuration.

    Returns
    -------
    HeuristicWeights
        Optimized weights.
    """
    # Load baseline weights (B0)
    if config.baseline_path and os.path.exists(config.baseline_path):
        baseline_weights = load_weights_from_file(config.baseline_path)
        print(f"Loaded baseline weights from: {config.baseline_path}")
    else:
        baseline_weights = dict(BASE_V1_BALANCED_WEIGHTS)
        print("Using default BASE_V1_BALANCED_WEIGHTS as baseline")

    # Resolve run_id and run_dir
    run_id = config.run_id or datetime.now().strftime("cmaes_%Y%m%d_%H%M%S")

    run_dir = config.run_dir
    if not run_dir:
        if config.resume_from:
            # When resuming, treat resume_from as the canonical run directory.
            run_dir = config.resume_from
        else:
            # Derive run directory from the output path.
            output_dir = os.path.dirname(config.output_path) or "logs/cmaes"
            run_dir = os.path.join(output_dir, "runs", run_id)

    os.makedirs(run_dir, exist_ok=True)

    # Resolve checkpoint and generation summary directories
    checkpoint_dir = config.checkpoint_dir or os.path.join(
        run_dir, "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    generation_summaries_dir = os.path.join(run_dir, "generations")
    os.makedirs(generation_summaries_dir, exist_ok=True)

    # Determine the list of boards used for evaluation (defaulting to the
    # primary board_type when no explicit list is provided).
    eval_boards = config.eval_boards or [config.board_type]

    # Persist run metadata (additive JSON; schema is backward-compatible)
    run_meta_path = os.path.join(run_dir, "run_meta.json")
    run_meta: Dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "generations": config.generations,
        "population_size": config.population_size,
        "games_per_eval": config.games_per_eval,
        "sigma": config.sigma,
        "board": config.board_type.value,
        "eval_boards": [b.value for b in eval_boards],
        "max_moves": config.max_moves,
        "baseline_path": config.baseline_path,
        "seed": config.seed,
        "checkpoint_dir": checkpoint_dir,
        "generation_summaries_dir": generation_summaries_dir,
        "output_path": config.output_path,
        "resume_from": config.resume_from,
        "opponent_mode": config.opponent_mode,
        "eval_mode": config.eval_mode,
        "state_pool_id": config.state_pool_id,
        "eval_randomness": config.eval_randomness,
        "num_players": config.num_players,
        "debug_plateau": config.debug_plateau,
        "progress_interval_sec": config.progress_interval_sec,
        "enable_eval_progress": config.enable_eval_progress,
    }
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    # Initialize with baseline weights or resume from last checkpoint
    initial_weights = weights_to_array(baseline_weights)
    best_weights = baseline_weights
    best_fitness = -float("inf")
    resume_generation_offset = 0

    if config.resume_from:
        # Approximate resume: recentre CMA-ES on the last saved checkpoint.
        latest_gen = None
        latest_checkpoint_path = None

        try:
            for name in sorted(os.listdir(checkpoint_dir)):
                if not (
                    name.startswith("checkpoint_gen")
                    and name.endswith(".json")
                ):
                    continue
                gen_str = name[len("checkpoint_gen"):-len(".json")]
                try:
                    gen_num = int(gen_str)
                except ValueError:
                    continue
                latest_gen = gen_num
                latest_checkpoint_path = os.path.join(checkpoint_dir, name)
        except FileNotFoundError:
            latest_checkpoint_path = None

        if latest_checkpoint_path is not None:
            try:
                # Load weights
                best_weights = load_weights_from_file(latest_checkpoint_path)
                # Optionally recover fitness metadata
                with open(latest_checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_payload = json.load(f)
                if (
                    isinstance(checkpoint_payload, dict)
                    and "fitness" in checkpoint_payload
                ):
                    best_fitness = float(checkpoint_payload["fitness"])
                initial_weights = weights_to_array(best_weights)
                resume_generation_offset = latest_gen or 0
                print(
                    f"Resuming from checkpoint {latest_checkpoint_path} "
                    f"(generation {resume_generation_offset})"
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"WARNING: Failed to load resume checkpoint: {exc}")

    n_weights = len(WEIGHT_KEYS)

    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
        print(f"Random seed set to: {config.seed}")

    print("\nCMA-ES optimization starting:")
    print(f"  Run ID: {run_id}")
    print(f"  Run directory: {run_dir}")
    print(f"  Generations (this run): {config.generations}")
    print(f"  Population size: {config.population_size}")
    print(f"  Games per evaluation: {config.games_per_eval}")
    print(f"  Initial sigma: {config.sigma}")
    print(f"  Number of weights: {n_weights}")
    print(f"  Board type: {config.board_type.value}")
    boards_label = ",".join(b.value for b in eval_boards)
    print(f"  Boards: {boards_label}")
    if len(eval_boards) > 1:
        print("  Aggregate fitness: mean across boards")
    print(f"  Max moves per game: {config.max_moves}")
    print(f"  Number of players: {config.num_players}")
    print(f"  Opponent mode: {config.opponent_mode}")
    print(f"  Eval mode: {config.eval_mode}")
    print(f"  Eval randomness: {config.eval_randomness}")
    if config.eval_mode == "multi-start":
        print(f"  State pool id: {config.state_pool_id or 'v1'}")
        if (
            len(eval_boards) == 3
            and config.state_pool_id in (None, "v1")
        ):
            print(
                "  Evaluation preset: 2p-multi-board-multi-start "
                f"(games_per_eval={config.games_per_eval}, "
                f"eval_randomness={config.eval_randomness})"
            )
    if config.seed is not None:
        print(f"  Random seed: {config.seed}")
    if resume_generation_offset:
        print(f"  Resuming from generation offset: {resume_generation_offset}")
    print()

    # Initialize CMA-ES
    # We want to maximize fitness, but cma minimizes by default,
    # so we negate the fitness values.
    # Compute per-weight bounds that accommodate all baseline weights. The
    # baseline includes weights like WEIGHT_VICTORY_THRESHOLD_BONUS=1000 and
    # proximity factors at 50, so we need weight-specific upper bounds.
    # Use symmetric bounds to allow both positive and negative weights,
    # enabling the optimizer to explore inverted features (sensitivity
    # analysis showed some weights hurt when overweighted).
    max_abs_baseline_weight = max(abs(v) for v in baseline_weights.values())
    bound_magnitude = max(100, max_abs_baseline_weight * 2)  # 2x headroom

    cma_options = {
        "popsize": config.population_size,
        "maxiter": config.generations + resume_generation_offset,
        "bounds": [-bound_magnitude, bound_magnitude],  # Symmetric bounds
        "verbose": -9,  # Suppress cma's own logging
        # Disable ALL early stopping - use infinity to disable tolerances
        "tolfun": float("inf"),
        "tolx": float("inf"),
        "tolfunhist": float("inf"),
        "tolstagnation": int(1e9),  # Very high number
        "tolconditioncov": float("inf"),
    }
    if config.seed is not None:
        cma_options["seed"] = config.seed

    if cma is None:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "CMA-ES optimization requires the 'cma' package. "
            "Install it via 'pip install cma>=3.3.0' before running this CLI."
        )

    es = cast(Any, cma).CMAEvolutionStrategy(
        initial_weights.tolist(),
        config.sigma,
        cma_options,
    )

    print(f"CMA-ES initialized, will run for {config.generations} generations")

    # Initialize progress reporter for time-based progress output.
    progress_reporter = OptimizationProgressReporter(
        total_generations=config.generations,
        candidates_per_generation=config.population_size,
        report_interval_sec=config.progress_interval_sec,
    )

    # Run for a fixed number of generations (ignore CMA-ES stopping criteria)
    for local_generation in range(1, config.generations + 1):
        generation = resume_generation_offset + local_generation
        progress_reporter.start_generation(generation)

        # Sample candidates
        solutions = es.ask()

        # Evaluate fitness for each candidate (negate for minimization)
        fitnesses: List[float] = []
        gen_fitnesses: List[float] = []
        verbose_eval = generation == 1  # Debug first generation

        # Incumbent (B1) is the best-known weights at the *start* of
        # the generation.
        incumbent_for_generation = (
            best_weights
            if config.opponent_mode == "baseline-plus-incumbent"
            else None
        )

        boards_for_eval = eval_boards
        candidate_per_board_fitness: List[Dict[BoardType, float]] = []

        for idx, solution in enumerate(solutions):
            candidate_weights = array_to_weights(
                np.array(solution, dtype=float)
            )

            # Optional per-candidate debug logging for plateau diagnosis.
            if config.debug_plateau:
                def cmaes_debug_hook(
                    stats: Dict[str, Any],
                    *,
                    _generation: int = generation,
                    _idx: int = idx,
                ) -> None:
                    print(
                        "[CMAES] "
                        f"gen={_generation} idx={_idx} "
                        f"fitness={stats['fitness']:.4f} "
                        f"W={stats['wins']} D={stats['draws']} "
                        f"L={stats['losses']} "
                        f"l2={stats['weight_l2']:.3f} "
                        f"games={stats['games_per_eval']}"
                    )
            else:
                cmaes_debug_hook = None  # type: ignore[assignment]

            # Multi-player evaluation (3p/4p) uses a different fitness function.
            if config.num_players > 2:
                # evaluate_fitness_multiplayer returns a score in [-1, 1].
                # Normalise this to [0, 1] so that CMA-ES sees a consistent
                # fitness range with the 2-player win-rate metric.
                raw_score = evaluate_fitness_multiplayer(
                    candidate_weights=candidate_weights,
                    baseline_weights=baseline_weights,
                    num_players=config.num_players,
                    games_per_eval=config.games_per_eval,
                    boards=boards_for_eval,
                    state_pool_id=config.state_pool_id or "",
                    seed=config.seed,
                )
                fitness = (raw_score + 1.0) / 2.0
                # We do not have per-board breakdown from the multi-player
                # helper, so mirror the aggregate fitness across all boards
                # for reporting purposes.
                per_board_fitness = {
                    board: fitness for board in boards_for_eval
                }
            elif len(boards_for_eval) == 1:
                board_type = boards_for_eval[0]
                fitness = evaluate_fitness(
                    candidate_weights,
                    baseline_weights,
                    config.games_per_eval,
                    board_type,
                    verbose=(verbose_eval and idx == 0),
                    opponent_mode=config.opponent_mode,
                    incumbent_weights=incumbent_for_generation,
                    max_moves=config.max_moves,
                    debug_hook=cmaes_debug_hook,
                    eval_mode=config.eval_mode,
                    state_pool_id=config.state_pool_id,
                    eval_randomness=config.eval_randomness,
                    seed=config.seed,
                )
                per_board_fitness = {board_type: fitness}
            else:
                (
                    aggregate_fitness,
                    per_board_fitness,
                ) = evaluate_fitness_over_boards(
                    candidate_weights=candidate_weights,
                    baseline_weights=baseline_weights,
                    games_per_eval=config.games_per_eval,
                    boards=boards_for_eval,
                    opponent_mode=config.opponent_mode,
                    max_moves=config.max_moves,
                    verbose=(verbose_eval and idx == 0),
                    debug_hook=cmaes_debug_hook,
                    eval_mode=config.eval_mode,
                    state_pool_id=config.state_pool_id,
                    seed=config.seed,
                    eval_randomness=config.eval_randomness,
                    progress_label=f"CMAES gen={generation} cand={idx+1}",
                    progress_interval_sec=config.progress_interval_sec,
                    enable_eval_progress=config.enable_eval_progress,
                )
                fitness = aggregate_fitness

            candidate_per_board_fitness.append(per_board_fitness)
            fitnesses.append(-fitness)  # Negate for minimization
            gen_fitnesses.append(fitness)

            # Record candidate evaluation for progress reporting
            # games_per_eval * len(boards) games were played for this candidate
            games_this_candidate = config.games_per_eval * len(boards_for_eval)
            progress_reporter.record_candidate(
                candidate_idx=idx + 1,
                fitness=fitness,
                games_played=games_this_candidate,
            )

        # Update CMA-ES
        es.tell(solutions, fitnesses)

        # Track statistics (un-negated for reporting)
        mean_fitness = float(np.mean(gen_fitnesses))
        std_fitness = float(np.std(gen_fitnesses))
        max_fitness = float(np.max(gen_fitnesses))

        # Per-generation best candidate
        best_idx = int(np.argmax(gen_fitnesses))
        gen_best_fitness = gen_fitnesses[best_idx]
        gen_best_weights = array_to_weights(
            np.array(solutions[best_idx], dtype=float)
        )
        best_per_board = candidate_per_board_fitness[best_idx]
        best_per_board_serialized = {
            board.value: float(score)
            for board, score in best_per_board.items()
        }

        is_new_global_best = gen_best_fitness > best_fitness
        if is_new_global_best:
            best_fitness = gen_best_fitness
            best_weights = gen_best_weights

        print(
            f"Generation {generation:3d}: "
            f"mean={mean_fitness:.4f}, "
            f"std={std_fitness:.4f}, "
            f"best={max_fitness:.4f}, "
            f"overall_best={best_fitness:.4f}"
        )

        # Save checkpoint for the current global best
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_gen{generation:03d}.json"
        )
        save_weights_to_file(
            best_weights,
            checkpoint_path,
            generation=generation,
            fitness=best_fitness,
        )

        # Save generation summary
        summary_path = os.path.join(
            generation_summaries_dir, f"generation_{generation:03d}.json"
        )
        summary_payload: Dict[str, object] = {
            "generation": generation,
            "population_size": config.population_size,
            "games_per_eval": config.games_per_eval,
            "board": config.board_type.value,
            "eval_boards": [b.value for b in eval_boards],
            "opponent_mode": config.opponent_mode,
            "max_moves": config.max_moves,
            "mean_fitness": mean_fitness,
            "std_fitness": std_fitness,
            "max_fitness": max_fitness,
            "best_candidate": {
                "fitness": gen_best_fitness,
                "is_new_global_best": is_new_global_best,
                "checkpoint_path": checkpoint_path,
                "per_board_fitness": best_per_board_serialized,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, sort_keys=True)

        # Report generation completion with statistics
        progress_reporter.finish_generation(
            mean_fitness=mean_fitness,
            best_fitness=max_fitness,
            std_fitness=std_fitness,
        )

    # Emit final optimization summary
    progress_reporter.finish()

    final_generation = resume_generation_offset + config.generations
    print(
        f"\nCompleted {config.generations} generations "
        f"(last global generation {final_generation})"
    )
    print(f"Best fitness achieved: {best_fitness:.4f}")

    # Save final weights
    save_weights_to_file(
        best_weights,
        config.output_path,
        generation=final_generation,
        fitness=best_fitness,
    )
    print(f"Optimized weights saved to: {config.output_path}")

    return best_weights


def main():
    parser = argparse.ArgumentParser(
        description="CMA-ES optimization for HeuristicAI weights"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of CMA-ES generations (default: 50)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size per generation (default: 20)",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=10,
        help="Games to play per fitness evaluation (default: 10)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Initial step size for CMA-ES (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/cmaes/optimized_weights.json",
        help=(
            "Output target. If this ends with '.json' it is treated as the "
            "path to the final weights file (backward-compatible behaviour). "
            "Otherwise it is treated as a base directory under which a "
            "run-specific subdirectory will be created."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline weights file for comparison (optional)",
    )
    parser.add_argument(
        "--board",
        type=str,
        choices=["square8", "square19", "hex"],
        default="square8",
        help="Board type for self-play (default: square8)",
    )
    parser.add_argument(
        "--eval-boards",
        type=str,
        default="square8",
        help=(
            "Comma-separated list of board types to evaluate on, e.g. "
            "'square8' (default), 'square8,square19', or "
            "'square8,square19,hex'."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints after each generation (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help=(
            "Maximum moves per self-play game before declaring a draw "
            "(default: 200)."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Logical run identifier used to name the run directory. "
            "If omitted, a timestamp-based id is generated."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to an existing CMA-ES run directory to approximately "
            "resume from."
        ),
    )
    parser.add_argument(
        "--opponent-mode",
        type=str,
        choices=["baseline-only", "baseline-plus-incumbent"],
        default="baseline-only",
        help=(
            "Opponent pool mode for fitness evaluation "
            "(default: baseline-only)."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["initial-only", "multi-start"],
        default="multi-start",
        help=(
            "Evaluation mode for candidate fitness: 'initial-only' uses the "
            "empty starting position only; 'multi-start' samples starting "
            "states from a precomputed pool (default: 'multi-start')."
        ),
    )
    parser.add_argument(
        "--state-pool-id",
        type=str,
        default="v1",
        help=(
            "Identifier for the evaluation state pool when using "
            "eval-mode=multi-start (default: v1). For multi-player runs, "
            "pass an explicit id such as 'square19_3p_pool_v1' or "
            "'hex_4p_pool_v1' that matches configured entries in "
            "app.training.eval_pools.POOL_PATHS."
        ),
    )
    parser.add_argument(
        "--eval-randomness",
        type=float,
        default=0.02,
        help=(
            "Optional randomness parameter for heuristic evaluation. "
            "By default this is 0.02, which introduces light stochastic "
            "tie-breaking while remaining reproducible when a seed is set. "
            "Use 0.0 to keep evaluation fully deterministic."
        ),
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=10.0,
        help=(
            "Minimum seconds between optimization progress log lines "
            "(default: 10s). Increase for very long runs if logs are "
            "too chatty; decrease slightly if you need more frequent "
            "heartbeats, while still avoiding >60s silence."
        ),
    )
    parser.add_argument(
        "--debug-plateau",
        action="store_true",
        help=(
            "Enable detailed per-candidate evaluation logging "
            "(W/DL, fitness, L2 distance) to help diagnose plateau/0.5 "
            "behaviour. Disabled by default to keep logs compact."
        ),
    )
    parser.add_argument(
        "--disable-eval-progress",
        action="store_true",
        help=(
            "Disable per-board evaluation progress reporters and rely "
            "solely on the generation-level optimisation reporter. "
            "This is useful for very log-sensitive environments or "
            "when only outer-loop heartbeats are required. By default "
            "per-board evaluation progress is enabled; this flag "
            "turns it off."
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help=(
            "Number of players for evaluation games (default: 2). "
            "When set to 3 or 4, uses evaluate_fitness_multiplayer() "
            "which assigns the candidate to one random seat and uses "
            "baseline weights for all other players. Requires "
            "corresponding N-player state pools (for example, use "
            "--state-pool-id square19_3p_pool_v1 for 3p Square19 or "
            "--state-pool-id hex_4p_pool_v1 for 4p Hex)."
        ),
    )

    args = parser.parse_args()

    # Map board type string to enum
    board_type = BOARD_NAME_TO_TYPE[args.board]

    # Parse evaluation boards from CLI string.
    raw_names = [
        name.strip() for name in args.eval_boards.split(",") if name.strip()
    ]
    if not raw_names:
        raise ValueError(
            "At least one board must be specified in --eval-boards"
        )

    eval_boards: List[BoardType] = []
    for name in raw_names:
        try:
            eval_boards.append(BOARD_NAME_TO_TYPE[name])
        except KeyError:
            raise ValueError(f"Unknown board name in --eval-boards: {name!r}")

    # Resolve run_id
    run_id = args.run_id or datetime.now().strftime("cmaes_%Y%m%d_%H%M%S")

    # Derive run_dir, output_path, and checkpoint_dir.
    output_arg = args.output
    output_ext = os.path.splitext(output_arg)[1].lower()

    if args.resume_from:
        # Approximate resume from an existing run directory.
        run_dir = args.resume_from
        if output_ext == ".json":
            output_path = output_arg
        else:
            output_path = os.path.join(run_dir, "best_weights.json")
        checkpoint_dir = args.checkpoint_dir or os.path.join(
            run_dir, "checkpoints"
        )
    else:
        if output_ext == ".json":
            # Backward-compatible: treat --output as the final weights file.
            output_path = output_arg
            base_dir = os.path.dirname(output_arg) or "logs/cmaes"
            run_dir = os.path.join(base_dir, "runs", run_id)
        else:
            # Directory mode: create a per-run subdirectory and write
            # best_weights.json there.
            base_dir = output_arg
            run_dir = os.path.join(base_dir, "runs", run_id)
            output_path = os.path.join(run_dir, "best_weights.json")
        checkpoint_dir = args.checkpoint_dir or os.path.join(
            run_dir, "checkpoints"
        )

    # Resolve state_pool_id from CLI. For multi-player runs (num_players > 2),
    # callers are expected to provide an explicit pool id that matches
    # app.training.eval_pools.POOL_PATHS (for example, 'square19_3p_pool_v1'
    # or 'hex_4p_pool_v1').
    state_pool_id = args.state_pool_id

    config = CMAESConfig(
        generations=args.generations,
        population_size=args.population_size,
        games_per_eval=args.games_per_eval,
        sigma=args.sigma,
        output_path=output_path,
        baseline_path=args.baseline,
        board_type=board_type,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        max_moves=args.max_moves,
        opponent_mode=args.opponent_mode,
        run_id=run_id,
        run_dir=run_dir,
        resume_from=args.resume_from,
        eval_mode=args.eval_mode,
        state_pool_id=state_pool_id,
        eval_boards=eval_boards,
        eval_randomness=args.eval_randomness,
        num_players=args.num_players,
        debug_plateau=args.debug_plateau,
        progress_interval_sec=args.progress_interval_sec,
        enable_eval_progress=not args.disable_eval_progress,
    )

    run_cmaes_optimization(config)


if __name__ == "__main__":
    main()

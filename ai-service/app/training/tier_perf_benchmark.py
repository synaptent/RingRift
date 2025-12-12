"""Tier performance benchmark helpers for RingRift AI difficulty tiers.

This module provides a small, CI-friendly benchmark harness that measures
per-move latency for canonical ladder tiers. Budgets are defined in
app.config.perf_budgets and enforced by tests in ai-service/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

from app.models import BoardType, GameStatus
from app.training.env import TrainingEnvConfig, make_env
from app.training.tier_eval_config import (
    TierEvaluationConfig,
    get_tier_config,
)
from app.training.tier_eval_runner import (
    _create_ladder_ai_instance,
)
from app.config.perf_budgets import (
    TierPerfBudget,
    get_tier_perf_budget,
)


@dataclass
class TierPerfSample:
    """Raw latency samples for a single tier benchmark run."""

    tier_name: str
    difficulty: int
    board_type: BoardType
    num_players: int
    move_latencies_ms: List[float]


@dataclass
class TierPerfResult:
    """Aggregated latency metrics for a single tier."""

    tier_name: str
    average_ms: float
    p95_ms: float
    budget: TierPerfBudget
    num_samples: int


def _resolve_eval_config(
    tier_name: str,
    budget: TierPerfBudget,
) -> TierEvaluationConfig:
    """Resolve the TierEvaluationConfig corresponding to a perf budget.

    For canonical square8 2p tiers the evaluation config keys are simple
    difficulty names ("D4", "D6", "D8"), while perf budgets may also be
    keyed by more explicit names such as "D6_SQ8_2P". This helper keeps
    the mapping logic in one place.
    """
    key = tier_name.upper()
    try:
        return get_tier_config(key)
    except KeyError:
        base = f"D{budget.difficulty}".upper()
        return get_tier_config(base)


def _compute_p95(latencies_ms: List[float]) -> float:
    """Return a simple 95th percentile estimate for a small sample."""
    if not latencies_ms:
        return 0.0
    sorted_samples = sorted(latencies_ms)
    # Index based on 0.95 * N, clamped into a valid range.
    idx = int(len(sorted_samples) * 0.95) - 1
    if idx < 0:
        idx = 0
    if idx >= len(sorted_samples):
        idx = len(sorted_samples) - 1
    return sorted_samples[idx]


def run_tier_perf_benchmark(
    tier_name: str,
    num_games: int = 4,
    moves_per_game: int = 16,
    seed: int = 1,
) -> TierPerfResult:
    """Run a small latency benchmark for the given tier.

    Parameters
    ----------
    tier_name:
        Name of the tier to benchmark, for example "D6" or "D6_SQ8_2P".
    num_games:
        Number of self-play games to run. For CI tests this is kept
        deliberately small to avoid long runtimes at high difficulties.
    moves_per_game:
        Maximum number of moves to sample per game.
    seed:
        Base RNG seed used to derive per-game seeds.

    Returns
    -------
    TierPerfResult
        Aggregated average and p95 per-move latency along with the
        configured budget for this tier.
    """
    budget: TierPerfBudget = get_tier_perf_budget(tier_name)
    eval_config: TierEvaluationConfig = _resolve_eval_config(
        tier_name,
        budget,
    )

    env_config = TrainingEnvConfig(
        board_type=budget.board_type,
        num_players=budget.num_players,
        max_moves=None,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    latencies: List[float] = []

    for game_index in range(num_games):
        game_seed = (seed * 1_000_003 + game_index) & 0x7FFFFFFF

        # For performance benchmarking we use the candidate tier on both
        # seats so that every move samples the same difficulty profile.
        candidate_diff = eval_config.candidate_difficulty

        ai_p1 = _create_ladder_ai_instance(
            tier_config=eval_config,
            difficulty=candidate_diff,
            player_number=1,
            time_budget_ms=None,
            ai_type_override=None,
            rng_seed=game_seed,
        )
        ai_p2 = _create_ladder_ai_instance(
            tier_config=eval_config,
            difficulty=candidate_diff,
            player_number=2,
            time_budget_ms=None,
            ai_type_override=None,
            rng_seed=game_seed,
        )

        state = env.reset(seed=game_seed)
        done = False
        moves_sampled = 0

        while not done and moves_sampled < moves_per_game:
            current_player = state.current_player
            current_ai = ai_p1 if current_player == 1 else ai_p2

            t0 = perf_counter()
            move = current_ai.select_move(state)
            if move is None:
                # Per RR-CANON-R076: when get_valid_moves returns empty,
                # check for phase requirements that require bookkeeping moves
                requirement = GameEngine.get_phase_requirement(
                    state,
                    current_player,
                )
                if requirement is not None:
                    move = GameEngine.synthesize_bookkeeping_move(
                        requirement,
                        state,
                    )
                else:
                    # True "no moves" case - terminate immediately.
                    break
            # Mirror the /ai/move critical path by including evaluation.
            _ = current_ai.evaluate_position(state)
            t1 = perf_counter()

            latencies.append((t1 - t0) * 1000.0)

            state, _reward, done, info = env.step(move)
            moves_sampled += 1

            if state.game_status != GameStatus.ACTIVE:
                done = True
            moves_played = info.get("moves_played")
            if moves_played is not None:
                # Defensive: trust env's move counter when present.
                moves_sampled = min(moves_sampled, int(moves_played))

    if latencies:
        average_ms = sum(latencies) / float(len(latencies))
        p95_ms = _compute_p95(latencies)
    else:
        average_ms = 0.0
        p95_ms = 0.0

    return TierPerfResult(
        tier_name=budget.tier_name,
        average_ms=average_ms,
        p95_ms=p95_ms,
        budget=budget,
        num_samples=len(latencies),
    )

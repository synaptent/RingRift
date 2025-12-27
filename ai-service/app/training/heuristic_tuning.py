"""CMA-ES heuristic weight optimization module.

Provides utilities for optimizing heuristic weights using a simplified
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) approach.

Extracted from train.py to improve modularity (Dec 2025).
"""

import contextlib
import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from app.ai.heuristic_weights import (
    HEURISTIC_WEIGHT_KEYS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.training.eval_pools import run_heuristic_tier_eval
from app.training.tier_eval_config import (
    HEURISTIC_TIER_SPECS,
    HeuristicTierSpec,
)
from app.training.seed_utils import seed_all

__all__ = [
    "evaluate_heuristic_candidate",
    "run_cmaes_heuristic_optimization",
    "temporary_heuristic_profile",
]


def _flatten_heuristic_weights(
    profile: Mapping[str, float],
) -> tuple[list[str], list[float]]:
    """
    Deterministically flatten a heuristic weight profile into (keys, values).

    Keys are ordered according to HEURISTIC_WEIGHT_KEYS so that both CMA-ES
    and reconstruction remain stable across runs and consistent with other
    heuristic-training tooling.
    """
    keys: list[str] = list(HEURISTIC_WEIGHT_KEYS)
    values: list[float] = []
    for k in keys:
        try:
            values.append(float(profile[k]))
        except KeyError as exc:
            raise KeyError(
                f"Missing heuristic weight {k!r} in profile; all profiles "
                "used for optimisation must define the full "
                "HEURISTIC_WEIGHT_KEYS set."
            ) from exc
    return keys, values


def _reconstruct_heuristic_profile(
    keys: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    """Reconstruct a heuristic weight mapping from (keys, values)."""
    if len(keys) != len(values):
        raise ValueError(
            "Length mismatch reconstructing heuristic profile: "
            f"{len(keys)} keys vs {len(values)} values."
        )
    return {k: float(v) for k, v in zip(keys, values, strict=False)}


@contextlib.contextmanager
def temporary_heuristic_profile(
    profile_id: str,
    weights: Mapping[str, float],
):
    """
    Temporarily register a heuristic weight profile in the
    HEURISTIC_WEIGHT_PROFILES registry.

    This helper is intended for offline training/evaluation only (e.g.
    CMA-ES or search jobs) and must not be used on production code paths.
    """
    had_existing = profile_id in HEURISTIC_WEIGHT_PROFILES
    old_value = HEURISTIC_WEIGHT_PROFILES.get(profile_id)
    HEURISTIC_WEIGHT_PROFILES[profile_id] = dict(weights)
    try:
        yield
    finally:
        if had_existing:
            assert old_value is not None
            HEURISTIC_WEIGHT_PROFILES[profile_id] = old_value
        else:
            HEURISTIC_WEIGHT_PROFILES.pop(profile_id, None)


def _get_heuristic_tier_by_id(tier_id: str) -> HeuristicTierSpec:
    """Return the HeuristicTierSpec with the given id or raise ValueError."""
    for spec in HEURISTIC_TIER_SPECS:
        if spec.id == tier_id:
            return spec
    available = ", ".join(sorted(s.id for s in HEURISTIC_TIER_SPECS))
    raise ValueError(
        f"Unknown heuristic tier_id {tier_id!r}. "
        f"Available heuristic tiers: {available}"
    )


def evaluate_heuristic_candidate(
    tier_spec: HeuristicTierSpec,
    base_profile_id: str,
    keys: Sequence[str],
    candidate_vector: Sequence[float],
    rng_seed: int,
    games_per_candidate: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate a heuristic weight candidate via run_heuristic_tier_eval.

    Returns (fitness, raw_result_dict) where fitness is a scalar with
    higher values representing better performance.
    """
    # Rebuild mapping and register under a temporary candidate profile id.
    candidate_weights = _reconstruct_heuristic_profile(keys, candidate_vector)
    candidate_profile_id = f"cmaes_candidate_{tier_spec.id}"

    # Derive the concrete tier spec used for this evaluation. We keep all
    # structural fields but swap in the candidate/baseline profile ids so
    # that the tier harness routes AIs via the appropriate weight profiles.
    eval_tier = HeuristicTierSpec(
        id=tier_spec.id,
        name=tier_spec.name,
        board_type=tier_spec.board_type,
        num_players=tier_spec.num_players,
        eval_pool_id=tier_spec.eval_pool_id,
        num_games=tier_spec.num_games,
        candidate_profile_id=candidate_profile_id,
        baseline_profile_id=base_profile_id,
        description=tier_spec.description,
    )

    max_games = games_per_candidate or tier_spec.num_games

    with temporary_heuristic_profile(candidate_profile_id, candidate_weights):
        result = run_heuristic_tier_eval(
            tier_spec=eval_tier,
            rng_seed=rng_seed,
            max_games=max_games,
        )

    # Compute a simple scalar fitness from win/draw/loss and margins.
    games_played_raw = result.get("games_played") or 0
    games_played = max(1, int(games_played_raw))
    results = result.get("results") or {}
    wins = float(results.get("wins", 0.0))
    draws = float(results.get("draws", 0.0))
    win_rate = (wins + 0.5 * draws) / games_played

    margins = result.get("margins") or {}
    ring_margin = float(margins.get("ring_margin_mean") or 0.0)
    territory_margin = float(margins.get("territory_margin_mean") or 0.0)
    # Ring margin is primary; territory margin is down-weighted to avoid
    # over-optimising purely for space leads.
    margin_score = ring_margin + 0.25 * territory_margin

    fitness = float(win_rate + 0.01 * margin_score)
    return fitness, result


def run_cmaes_heuristic_optimization(
    tier_id: str,
    base_profile_id: str,
    generations: int = 5,
    population_size: int = 8,
    rng_seed: int = 1,
    games_per_candidate: int | None = None,
    evaluate_fn: Callable[..., tuple[float, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """
    Run a small CMA-ES-style optimisation loop over heuristic weights.

    The optimisation is offline-only and uses the heuristic eval-pool
    harness as its fitness function. It adapts the mean of a Gaussian
    search distribution over the heuristic weight vector while keeping a
    simple isotropic covariance (no full CMA matrix) for robustness.
    """
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    seed_all(rng_seed)
    py_rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed + 1)

    evaluate = evaluate_fn or evaluate_heuristic_candidate

    tier_spec = _get_heuristic_tier_by_id(tier_id)

    if base_profile_id not in HEURISTIC_WEIGHT_PROFILES:
        available = ", ".join(sorted(HEURISTIC_WEIGHT_PROFILES.keys()))
        raise ValueError(
            f"Unknown heuristic base_profile_id {base_profile_id!r}. "
            f"Available: {available}"
        )

    base_profile = HEURISTIC_WEIGHT_PROFILES[base_profile_id]
    keys, base_vector = _flatten_heuristic_weights(base_profile)

    dim = len(base_vector)
    mean = np.asarray(base_vector, dtype=float)
    # Initial step size chosen as a small fraction of the typical weight
    # magnitude so early generations explore but do not explode.
    sigma = 0.5

    history: list[dict[str, Any]] = []
    best_overall: dict[str, Any] | None = None
    best_overall_fitness = -float("inf")

    for gen in range(generations):
        candidates: list[dict[str, Any]] = []

        for _ in range(population_size):
            # Sample from an isotropic Gaussian around the current mean.
            perturbation = np_rng.standard_normal(dim)
            arr = mean + sigma * perturbation
            # Force a plain Python list[float] for downstream type-checkers.
            tmp = cast(Sequence[float], arr.tolist())
            vector: list[float] = [float(x) for x in tmp]

            eval_seed = py_rng.randint(1, 2**31 - 1)
            fitness, raw = evaluate(
                tier_spec=tier_spec,
                base_profile_id=base_profile_id,
                keys=keys,
                candidate_vector=vector,
                rng_seed=eval_seed,
                games_per_candidate=games_per_candidate,
            )
            fitness = float(fitness)
            candidate_entry = {
                "vector": vector,
                "fitness": fitness,
                "raw": raw,
            }
            candidates.append(candidate_entry)

            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_overall = {
                    "generation": gen,
                    "vector": vector,
                    "fitness": fitness,
                    "raw": raw,
                }

        # Sort population and update mean via weighted recombination of top mu.
        candidates.sort(key=lambda c: c["fitness"], reverse=True)
        mu = max(1, population_size // 2)
        top = candidates[:mu]

        weights_arr = np.array(
            [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)],
            dtype=float,
        )
        weights_arr /= weights_arr.sum()

        new_mean = np.zeros(dim, dtype=float)
        for w, cand in zip(weights_arr, top, strict=False):
            new_mean += w * np.asarray(cand["vector"], dtype=float)
        mean = new_mean

        mean_fitness = float(
            sum(c["fitness"] for c in candidates) / len(candidates)
        )
        history.append(
            {
                "generation": gen,
                "best_fitness": float(top[0]["fitness"]),
                "mean_fitness": mean_fitness,
            }
        )

        # Simple geometric decay of sigma to encourage convergence while
        # still leaving some exploration in later generations.
        sigma *= 0.9

    report: dict[str, Any] = {
        "run_type": "heuristic_cmaes_square8",
        "tier_id": tier_id,
        "base_profile_id": base_profile_id,
        "generations": generations,
        "population_size": population_size,
        "rng_seed": rng_seed,
        "games_per_candidate": games_per_candidate,
        "dimension": dim,
        "keys": list(keys),
        "history": history,
        "best": best_overall,
    }
    return report

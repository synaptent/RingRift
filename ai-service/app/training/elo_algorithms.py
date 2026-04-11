"""Pure Elo calculation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def is_random_participant(participant_id: str) -> bool:
    """Return True when a participant is a fixed random baseline."""
    pid_lower = participant_id.lower()
    if pid_lower.startswith("none:random"):
        return True
    if pid_lower in ("random", "baseline_random", "tier1_random"):
        return True
    if "random" in pid_lower and not any(
        token in pid_lower for token in ("heuristic", "minimax", "mcts", "descent", "neural")
    ):
        return True
    return False


def expected_scores(rating_a: float, rating_b: float) -> tuple[float, float]:
    """Compute pairwise expected scores."""
    exp_a = 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400))
    return exp_a, 1.0 - exp_a


def resolve_match_scores(
    participant_a: str,
    participant_b: str,
    winner: str | None,
) -> tuple[float, float]:
    """Convert a winner ID into pairwise scores."""
    if winner == participant_a:
        return 1.0, 0.0
    if winner == participant_b:
        return 0.0, 1.0
    return 0.5, 0.5


def adaptive_k_factor(games_a: int, games_b: int, base_k: float) -> float:
    """Scale K using the less-confident participant."""
    min_games = min(games_a, games_b)
    if min_games < 30:
        return base_k * 1.5
    if min_games < 100:
        return base_k * 1.25
    if min_games < 300:
        return base_k
    return base_k * 0.75


@dataclass
class EloComputation:
    """Computed pairwise Elo update."""

    score_a: float
    score_b: float
    change_a: float
    change_b: float
    new_rating_a: float
    new_rating_b: float


def compute_elo_update(
    *,
    participant_a: str,
    participant_b: str,
    winner: str | None,
    rating_a: float,
    rating_b: float,
    games_a: int,
    games_b: int,
    base_k: float,
    get_pinned_baseline_rating_fn: Callable[[str], float | None],
) -> EloComputation:
    """Compute the new ratings for a single pairwise result."""
    exp_a, exp_b = expected_scores(rating_a, rating_b)
    score_a, score_b = resolve_match_scores(participant_a, participant_b, winner)
    k = adaptive_k_factor(games_a, games_b, base_k)
    change_a = k * (score_a - exp_a)
    change_b = k * (score_b - exp_b)
    new_rating_a = rating_a + change_a
    new_rating_b = rating_b + change_b

    pinned_a = get_pinned_baseline_rating_fn(participant_a)
    pinned_b = get_pinned_baseline_rating_fn(participant_b)
    if pinned_a is not None:
        new_rating_a = float(pinned_a)
        change_a = 0.0
    if pinned_b is not None:
        new_rating_b = float(pinned_b)
        change_b = 0.0

    return EloComputation(
        score_a=score_a,
        score_b=score_b,
        change_a=change_a,
        change_b=change_b,
        new_rating_a=new_rating_a,
        new_rating_b=new_rating_b,
    )


__all__ = [
    "EloComputation",
    "adaptive_k_factor",
    "compute_elo_update",
    "expected_scores",
    "is_random_participant",
    "resolve_match_scores",
]

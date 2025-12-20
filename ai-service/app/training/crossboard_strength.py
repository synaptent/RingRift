"""Cross-board strength summary helpers.

This module contains small, dependency-free helpers for comparing tier/engine
strength across different board types without pulling in SciPy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def normalise_tier_name(tier: str) -> str:
    """Return a canonical tier name like ``"D6"`` or raise ``ValueError``."""
    name = tier.strip().upper()
    if name.startswith("D"):
        name = name[1:]
    if not name.isdigit():
        raise ValueError(f"Unsupported tier name {tier!r}; expected something like 'D6'.")
    return f"D{int(name)}"


def tier_number(tier: str) -> int:
    """Return the numeric difficulty for a tier label like ``"D8"``."""
    return int(normalise_tier_name(tier)[1:])


def rank_order_from_elos(elos: dict[str, float]) -> list[str]:
    """Return tier ids sorted from strongest to weakest by Elo."""
    items = [(normalise_tier_name(t), float(v)) for t, v in elos.items()]
    items.sort(key=lambda kv: (kv[1], tier_number(kv[0])), reverse=True)
    return [tier for tier, _ in items]


def rank_map(order: Sequence[str]) -> dict[str, int]:
    """Return 1-based rank map from an ordered tier list."""
    return {normalise_tier_name(tier): idx + 1 for idx, tier in enumerate(order)}


def spearman_rank_correlation(
    rank_a: dict[str, int],
    rank_b: dict[str, int],
    *,
    tiers: Iterable[str] | None = None,
) -> float:
    """Compute Spearman rank correlation (no tie handling).

    Args:
        rank_a: Mapping tier -> rank (1 = best).
        rank_b: Mapping tier -> rank (1 = best).
        tiers: Optional explicit tier set; defaults to intersection of keys.

    Returns:
        Correlation in [-1, 1]. Returns 0.0 when fewer than 2 tiers overlap.
    """
    if tiers is None:
        overlap = set(rank_a).intersection(rank_b)
    else:
        overlap = {normalise_tier_name(t) for t in tiers}
        overlap &= set(rank_a).intersection(rank_b)

    if len(overlap) < 2:
        return 0.0

    ordered = sorted(overlap, key=tier_number)
    n = len(ordered)
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0

    d_sq_sum = 0
    for tier in ordered:
        d = rank_a[tier] - rank_b[tier]
        d_sq_sum += d * d

    return 1.0 - (6.0 * d_sq_sum) / float(denom)


def inversion_count(elos: dict[str, float]) -> int:
    """Count inversions vs the expected monotone difficulty ordering.

    An inversion is a pair of tiers (i < j) where Elo(Di) > Elo(Dj), even
    though higher difficulty is expected to be stronger.
    """
    tiers = sorted((normalise_tier_name(t) for t in elos), key=tier_number)
    values = [float(elos[t]) for t in tiers]
    inv = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                inv += 1
    return inv


def summarize_crossboard_tier_strength(
    board_elos: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Compute a lightweight cross-board summary from per-board Elo maps."""
    boards = sorted(board_elos.keys())
    per_board_order = {
        board: rank_order_from_elos(elos) for board, elos in board_elos.items()
    }
    per_board_ranks = {
        board: rank_map(order) for board, order in per_board_order.items()
    }

    pairwise: list[dict[str, object]] = []
    for i, a in enumerate(boards):
        for b in boards[i + 1 :]:
            corr = spearman_rank_correlation(per_board_ranks[a], per_board_ranks[b])
            pairwise.append(
                {
                    "board_a": a,
                    "board_b": b,
                    "spearman": round(corr, 4),
                }
            )

    inversions = {
        board: inversion_count(elos) for board, elos in board_elos.items()
    }

    # Per-tier rank variance across boards (intersection only).
    tier_sets = [set(ranks.keys()) for ranks in per_board_ranks.values()]
    common_tiers = sorted(set.intersection(*tier_sets) if tier_sets else set(), key=tier_number)

    tier_rank_std: dict[str, float] = {}
    for tier in common_tiers:
        ranks = [per_board_ranks[b][tier] for b in boards]
        mean = sum(ranks) / len(ranks)
        var = sum((r - mean) ** 2 for r in ranks) / len(ranks)
        tier_rank_std[tier] = round(math.sqrt(var), 4)

    return {
        "boards": boards,
        "common_tiers": common_tiers,
        "pairwise_rank_correlation": pairwise,
        "inversion_counts": inversions,
        "tier_rank_std": tier_rank_std,
    }


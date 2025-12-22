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


# =============================================================================
# Cross-Board Configuration System
# =============================================================================

# All supported board/player configurations (9 total)
ALL_BOARD_CONFIGS: list[tuple[str, int]] = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]


def config_key(board: str, num_players: int) -> str:
    """Return a canonical config key like ``"square8_2p"``."""
    return f"{board}_{num_players}p"


def parse_config_key(key: str) -> tuple[str, int]:
    """Parse a config key like ``"square8_2p"`` into (board, num_players)."""
    parts = key.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].endswith("p"):
        raise ValueError(f"Invalid config key: {key!r}")
    board = parts[0]
    num_players = int(parts[1][:-1])
    return board, num_players


def aggregate_cross_board_elos(
    config_elos: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Aggregate Elo estimates across board/player configurations.

    Args:
        config_elos: Mapping from config key (e.g., "square8_2p") to Elo.
        weights: Optional per-config weights (defaults to equal weighting).

    Returns:
        Dictionary with aggregation statistics:
        - "mean_elo": Weighted mean Elo across configs
        - "min_elo": Minimum Elo across configs
        - "max_elo": Maximum Elo across configs
        - "std_elo": Standard deviation of Elos
        - "num_configs": Number of configs with Elo data
        - "weakest_config": Config key with lowest Elo
        - "strongest_config": Config key with highest Elo
    """
    if not config_elos:
        return {
            "mean_elo": 0.0,
            "min_elo": 0.0,
            "max_elo": 0.0,
            "std_elo": 0.0,
            "num_configs": 0,
            "weakest_config": None,
            "strongest_config": None,
        }

    configs = list(config_elos.keys())
    elos = [config_elos[c] for c in configs]

    if weights is None:
        # Equal weighting
        w = [1.0 / len(elos)] * len(elos)
    else:
        total = sum(weights.get(c, 1.0) for c in configs)
        w = [weights.get(c, 1.0) / total for c in configs]

    mean_elo = sum(e * wt for e, wt in zip(elos, w))
    min_elo = min(elos)
    max_elo = max(elos)

    # Weighted variance
    var = sum(wt * (e - mean_elo) ** 2 for e, wt in zip(elos, w))
    std_elo = math.sqrt(var)

    weakest_idx = elos.index(min_elo)
    strongest_idx = elos.index(max_elo)

    return {
        "mean_elo": round(mean_elo, 1),
        "min_elo": round(min_elo, 1),
        "max_elo": round(max_elo, 1),
        "std_elo": round(std_elo, 1),
        "num_configs": len(elos),
        "weakest_config": configs[weakest_idx],
        "strongest_config": configs[strongest_idx],
    }


def check_promotion_threshold(
    config_elos: dict[str, float],
    target_elo: float,
    *,
    min_configs: int = 9,
    allow_partial: bool = False,
    max_gap: float = 50.0,
) -> dict[str, object]:
    """Check if model meets promotion threshold across all configurations.

    Args:
        config_elos: Mapping from config key to Elo estimate.
        target_elo: Target Elo threshold (e.g., 2000).
        min_configs: Minimum number of configs required for promotion.
        allow_partial: If True, allow promotion if weakest config is within max_gap Elo.
        max_gap: Maximum Elo gap allowed for partial promotion (default 50).

    Returns:
        Dictionary with promotion decision:
        - "eligible": Whether model is eligible for promotion
        - "reason": Human-readable reason for decision
        - "configs_above_threshold": Number of configs meeting threshold
        - "configs_below_threshold": List of configs below threshold
        - "gap_to_threshold": Elo gap for weakest config
    """
    if len(config_elos) < min_configs:
        return {
            "eligible": False,
            "reason": f"Insufficient configs: {len(config_elos)} < {min_configs}",
            "configs_above_threshold": 0,
            "configs_below_threshold": list(config_elos.keys()),
            "gap_to_threshold": None,
        }

    above = []
    below = []
    min_gap = float("inf")

    for cfg, elo in config_elos.items():
        gap = target_elo - elo
        if gap <= 0:
            above.append(cfg)
        else:
            below.append(cfg)
            min_gap = min(min_gap, gap)

    if not below:
        return {
            "eligible": True,
            "reason": f"All {len(above)} configs exceed {target_elo} Elo",
            "configs_above_threshold": len(above),
            "configs_below_threshold": [],
            "gap_to_threshold": 0.0,
        }

    # Check partial promotion with configurable gap
    if allow_partial and min_gap <= max_gap:
        return {
            "eligible": True,
            "reason": f"Partial promotion: {len(above)} configs above, weakest within {max_gap:.0f} Elo",
            "configs_above_threshold": len(above),
            "configs_below_threshold": below,
            "gap_to_threshold": round(min_gap, 1),
        }

    return {
        "eligible": False,
        "reason": f"{len(below)} configs below threshold (gap: {min_gap:.1f} Elo)",
        "configs_above_threshold": len(above),
        "configs_below_threshold": below,
        "gap_to_threshold": round(min_gap, 1),
    }


def check_promotion_threshold_strict(
    config_elos: dict[str, float],
    target_elo: float = 2000.0,
    *,
    min_configs: int = 12,
    max_elo_gap: float = 25.0,
) -> dict[str, object]:
    """Strict promotion check: ALL configs must meet threshold.

    This is the recommended gate for 2000+ Elo tier promotion.
    Ensures balanced strength across all 12 board/player configurations.

    Args:
        config_elos: Mapping from config key to Elo estimate.
        target_elo: Target Elo threshold (default 2000).
        min_configs: Minimum configs required (default 12 for all configs).
        max_elo_gap: Maximum acceptable gap below threshold (default 25 Elo).

    Returns:
        Dictionary with promotion decision including detailed analysis.
    """
    if len(config_elos) < min_configs:
        return {
            "eligible": False,
            "reason": f"Missing configs: {len(config_elos)}/{min_configs}",
            "configs_evaluated": len(config_elos),
            "configs_required": min_configs,
            "gap_analysis": None,
        }

    below_threshold = []
    max_gap = 0.0

    for cfg, elo in config_elos.items():
        gap = target_elo - elo
        if gap > 0:
            below_threshold.append((cfg, round(gap, 1)))
            max_gap = max(max_gap, gap)

    # Sort by gap (worst first)
    below_threshold.sort(key=lambda x: x[1], reverse=True)

    if not below_threshold:
        return {
            "eligible": True,
            "reason": f"All {len(config_elos)} configs exceed {target_elo:.0f} Elo",
            "configs_evaluated": len(config_elos),
            "configs_required": min_configs,
            "gap_analysis": {"max_gap": 0.0, "configs_below": []},
        }

    if max_gap > max_elo_gap:
        return {
            "eligible": False,
            "reason": f"Weakest config gap: {max_gap:.0f} Elo (limit: {max_elo_gap:.0f})",
            "configs_evaluated": len(config_elos),
            "configs_required": min_configs,
            "gap_analysis": {
                "max_gap": round(max_gap, 1),
                "configs_below": below_threshold,
            },
        }

    return {
        "eligible": True,
        "reason": f"All configs within {max_elo_gap:.0f} Elo of threshold",
        "configs_evaluated": len(config_elos),
        "configs_required": min_configs,
        "gap_analysis": {
            "max_gap": round(max_gap, 1),
            "configs_below": below_threshold,
        },
    }


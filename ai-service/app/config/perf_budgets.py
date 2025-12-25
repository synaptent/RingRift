"""Performance budget configuration for tiered AI move latency.

Budgets are intentionally coarse guard rails derived from the canonical
ladder think_time_ms settings in app.main and app.config.ladder_config.
They are used by tier_perf_benchmark and CI tests to detect large
performance regressions in high-difficulty AIs without hard-coding
magic numbers in the tests themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.ladder_config import get_ladder_tier_config
from app.models import BoardType


@dataclass(frozen=True)
class TierPerfBudget:
    """Performance budget for a single ladder tier.

    Attributes
    ----------
    tier_name:
        Canonical identifier for the tier, e.g. "D6_SQ8_2P".
    difficulty:
        Logical difficulty level on the 1–10 ladder.
    board_type:
        Board geometry for this tier.
    num_players:
        Number of active players (currently 2 for these budgets).
    max_avg_move_ms:
        Soft upper bound for average per-move latency over a small
        benchmark sample.
    max_p95_move_ms:
        Soft upper bound for the 95th percentile per-move latency.
    notes:
        Free-form documentation for how this budget was chosen.
    """

    tier_name: str
    difficulty: int
    board_type: BoardType
    num_players: int
    max_avg_move_ms: float
    max_p95_move_ms: float
    notes: str = ""


def _build_square8_two_player_budgets() -> dict[str, TierPerfBudget]:
    """Construct perf budgets for Square8 2‑player D3/D4/D5/D6/D7/D8 tiers.

    Budgets are derived from the canonical LadderTierConfig entries for
    (difficulty, BoardType.SQUARE8, num_players=2). For each tier we
    allow a modest overhead on top of the configured think_time_ms to
    account for environment overhead and host variance while still being
    strict enough to catch large regressions.
    """
    budgets: dict[str, TierPerfBudget] = {}

    def _from_ladder(difficulty: int, label: str) -> TierPerfBudget:
        ladder_cfg = get_ladder_tier_config(difficulty, BoardType.SQUARE8, 2)
        think_time_ms = float(ladder_cfg.think_time_ms)
        # Allow ~10% overhead on average and ~25% on p95 relative to the
        # configured search-time budget. This keeps the budget aligned
        # with existing per-move time limits in the ladder while
        # remaining tolerant to normal variance across hosts.
        max_avg = think_time_ms * 1.10
        max_p95 = think_time_ms * 1.25
        notes = (
            "Derived from LadderTierConfig think_time_ms="
            f"{ladder_cfg.think_time_ms} for difficulty {difficulty} "
            f"on {ladder_cfg.board_type.value} {ladder_cfg.num_players}p. "
            "Budgets are coarse guard rails rather than hard UX limits."
        )
        return TierPerfBudget(
            tier_name=label,
            difficulty=difficulty,
            board_type=ladder_cfg.board_type,
            num_players=ladder_cfg.num_players,
            max_avg_move_ms=max_avg,
            max_p95_move_ms=max_p95,
            notes=notes,
        )

    d3 = _from_ladder(3, "D3_SQ8_2P")
    d4 = _from_ladder(4, "D4_SQ8_2P")
    d5 = _from_ladder(5, "D5_SQ8_2P")
    d6 = _from_ladder(6, "D6_SQ8_2P")
    d7 = _from_ladder(7, "D7_SQ8_2P")
    d8 = _from_ladder(8, "D8_SQ8_2P")

    for budget in (d3, d4, d5, d6, d7, d8):
        # Canonical name keyed by full tier descriptor.
        budgets[budget.tier_name.upper()] = budget
        # Convenience alias keyed by bare difficulty ("D4", "D6", "D8").
        short_name = f"D{budget.difficulty}"
        budgets[short_name.upper()] = budget

    return budgets


TIER_PERF_BUDGETS: dict[str, TierPerfBudget] = (
    _build_square8_two_player_budgets()
)


def get_tier_perf_budget(tier_name: str) -> TierPerfBudget:
    """Return the TierPerfBudget for the given tier name.

    The lookup is case-insensitive and accepts either the full tier name
    (for example "D6_SQ8_2P") or a bare difficulty such as "D6".
    """
    key = tier_name.upper()
    try:
        return TIER_PERF_BUDGETS[key]
    except KeyError as exc:  # pragma: no cover - defensive error path
        available = ", ".join(sorted(TIER_PERF_BUDGETS.keys()))
        raise KeyError(
            "Unknown tier perf budget "
            f"'{tier_name}'. Available tiers: {available}"
        ) from exc


__all__ = ["TIER_PERF_BUDGETS", "TierPerfBudget", "get_tier_perf_budget"]

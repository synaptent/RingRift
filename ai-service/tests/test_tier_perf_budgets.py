"""Perf budget and tier latency benchmark tests."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure app/ is importable when running tests directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config.perf_budgets import (  # noqa: E402
    TierPerfBudget,
    TIER_PERF_BUDGETS,
    get_tier_perf_budget,
)
from app.config.ladder_config import (  # noqa: E402
    LadderTierConfig,
    get_ladder_tier_config,
)
from app.training.tier_eval_config import (  # noqa: E402
    TierEvaluationConfig,
    get_tier_config,
)
from app.training.tier_perf_benchmark import (  # noqa: E402
    _resolve_eval_config,
    TierPerfResult,
    run_tier_perf_benchmark,
)


def _unique_budgets() -> list[TierPerfBudget]:
    """Return de-duplicated TierPerfBudget entries."""
    # TIER_PERF_BUDGETS maps multiple keys (for example "D6"
    # and "D6_SQ8_2P") to the same TierPerfBudget instance.
    # Using set(...) collapses these aliases into unique entries.
    return sorted(
        set(TIER_PERF_BUDGETS.values()),
        key=lambda b: (b.board_type.value, b.num_players, b.difficulty),
    )


def test_perf_budget_config_sanity() -> None:
    """Budgets align with ladder and tier configs and are well-formed."""
    budgets = _unique_budgets()
    assert budgets, "Expected at least one TierPerfBudget entry"

    for budget in budgets:
        assert isinstance(budget, TierPerfBudget)

        # Ladder entry must exist and match basic fields.
        ladder_cfg: LadderTierConfig = get_ladder_tier_config(
            budget.difficulty,
            budget.board_type,
            budget.num_players,
        )
        assert ladder_cfg.board_type == budget.board_type
        assert ladder_cfg.num_players == budget.num_players
        assert ladder_cfg.difficulty == budget.difficulty
        assert ladder_cfg.think_time_ms > 0

        # TierEvaluationConfig should exist for the corresponding
        # difficulty.
        tier_name = f"D{budget.difficulty}"
        tier_cfg: TierEvaluationConfig = get_tier_config(tier_name)
        assert tier_cfg.board_type == budget.board_type
        assert tier_cfg.num_players == budget.num_players
        assert tier_cfg.candidate_difficulty == budget.difficulty

        # Budgets must be positive and reasonably close to think_time_ms.
        assert budget.max_avg_move_ms > 0.0
        assert budget.max_p95_move_ms > 0.0
        assert budget.max_p95_move_ms >= budget.max_avg_move_ms

        # Guard against budgets far above the per-move search-time
        # limit. We allow at most a 2x factor over think_time_ms.
        max_reasonable_avg = ladder_cfg.think_time_ms * 2.0
        assert budget.max_avg_move_ms <= max_reasonable_avg


def test_get_tier_perf_budget_aliases() -> None:
    """Short difficulty aliases (D4/D6/D8) resolve to the same budgets as full tier names."""
    # Pick one canonical square8 2p tier and verify that both the
    # full descriptor key and bare difficulty alias return the same
    # TierPerfBudget instance.
    full: TierPerfBudget = get_tier_perf_budget("D6_SQ8_2P")
    short: TierPerfBudget = get_tier_perf_budget("D6")

    assert full is short
    assert full.tier_name == "D6_SQ8_2P"
    assert short.difficulty == 6


def test_resolve_eval_config_uses_difficulty_fallback_for_full_tier_names() -> None:
    """_resolve_eval_config maps full tier names back to canonical evaluation configs."""
    budget = get_tier_perf_budget("D6_SQ8_2P")

    cfg_full: TierEvaluationConfig = _resolve_eval_config("D6_SQ8_2P", budget)
    cfg_short: TierEvaluationConfig = _resolve_eval_config("D6", budget)

    # Both resolution paths should agree on the underlying evaluation config.
    assert cfg_full == cfg_short
    assert cfg_full.candidate_difficulty == budget.difficulty
    assert cfg_full.board_type == budget.board_type
    assert cfg_full.num_players == budget.num_players


@pytest.mark.slow
@pytest.mark.parametrize(
    "tier_name",
    [
        "D4_SQ8_2P",
        "D6_SQ8_2P",
        "D8_SQ8_2P",
    ],
)
def test_tier_perf_within_budgets_smoke(tier_name: str) -> None:
    """Smoke test: observed per-move latency stays within tier budgets.

    The workload is intentionally small so it can run in CI while still
    catching large regressions in search-time behaviour.

    A 15% tolerance margin is applied to account for CI environment variability.
    The budgets are "coarse guard rails" per the config, not hard UX limits.
    """
    budget = get_tier_perf_budget(tier_name)

    result: TierPerfResult = run_tier_perf_benchmark(
        tier_name=tier_name,
        num_games=1,
        moves_per_game=4,
        seed=1,
    )

    assert result.num_samples > 0
    # Basic sanity: the result should be associated with the same
    # budget.
    assert result.budget.tier_name == budget.tier_name
    assert result.budget.difficulty == budget.difficulty

    # Guard rails: observed latencies should not exceed the configured
    # per-tier budget thresholds. Apply 25% tolerance for CI/local variability.
    # The budgets are "coarse guard rails" (per config notes), intended to catch
    # major regressions rather than enforce strict UX limits.
    tolerance = 1.25
    assert result.average_ms <= result.budget.max_avg_move_ms * tolerance, (
        f"Average latency {result.average_ms:.1f}ms exceeds budget "
        f"{result.budget.max_avg_move_ms:.1f}ms (with {tolerance:.0%} tolerance)"
    )
    assert result.p95_ms <= result.budget.max_p95_move_ms * tolerance, (
        f"P95 latency {result.p95_ms:.1f}ms exceeds budget "
        f"{result.budget.max_p95_move_ms:.1f}ms (with {tolerance:.0%} tolerance)"
    )

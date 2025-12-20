"""Tests for the tier perf benchmark CLI helpers."""

from __future__ import annotations

import os
import sys

from typing import Any, Dict

import pytest

# Ensure app/ is importable when running tests directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from app.config.perf_budgets import TierPerfBudget  # noqa: E402
from app.models import BoardType  # noqa: E402
from app.training.tier_perf_benchmark import (  # noqa: E402
  TierPerfResult,
)
from scripts.run_tier_perf_benchmark import _eval_budget  # type: ignore[import]  # noqa: E402


def _make_dummy_budget(max_avg: float, max_p95: float) -> TierPerfBudget:
  return TierPerfBudget(
    tier_name="D6_SQ8_2P",
    difficulty=6,
    board_type=BoardType.SQUARE8,
    num_players=2,
    max_avg_move_ms=max_avg,
    max_p95_move_ms=max_p95,
    notes="test budget",
  )


def _make_result(avg: float, p95: float, budget: TierPerfBudget, samples: int = 10) -> TierPerfResult:
  return TierPerfResult(
    tier_name=budget.tier_name,
    average_ms=avg,
    p95_ms=p95,
    budget=budget,
    num_samples=samples,
  )


@pytest.mark.parametrize(
  "avg,p95,max_avg,max_p95,expected",
  [
    # Both metrics within budget → full pass.
    (50.0, 80.0, 100.0, 120.0, {"within_avg": True, "within_p95": True, "overall_pass": True}),
    # Average just above budget → fail overall, but p95 still passes.
    (105.0, 110.0, 100.0, 120.0, {"within_avg": False, "within_p95": True, "overall_pass": False}),
    # p95 just above budget → fail overall, average still passes.
    (90.0, 130.0, 100.0, 120.0, {"within_avg": True, "within_p95": False, "overall_pass": False}),
  ],
)
def test_eval_budget_flags_thresholds(
  avg: float,
  p95: float,
  max_avg: float,
  max_p95: float,
  expected: dict[str, Any],
) -> None:
  """_eval_budget should set within_avg / within_p95 / overall_pass consistently."""
  budget = _make_dummy_budget(max_avg, max_p95)
  result = _make_result(avg, p95, budget)

  evaluation = _eval_budget(result)

  assert evaluation["within_avg"] is expected["within_avg"]
  assert evaluation["within_p95"] is expected["within_p95"]
  assert evaluation["overall_pass"] is expected["overall_pass"]


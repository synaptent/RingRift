import os
import sys
from typing import List

import pytest

# Ensure app and scripts packages are importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from scripts.run_self_play_soak import (  # type: ignore  # noqa: E402
  GameRecord,
  _has_anomalies,
)


def _make_records(reasons: list[str]) -> list[GameRecord]:
  return [
    GameRecord(
      index=i,
      num_players=2,
      board_type="square8",
      engine_mode="mixed",
      seed=None,
      length=0,
      status="active",
      winner=None,
      termination_reason=reason,
    )
    for i, reason in enumerate(reasons)
  ]


def test_has_anomalies_false_for_normal_terminations() -> None:
  records = _make_records(
    [
      "status:finished",
      "status:completed",
      "max_moves_reached",
      "env_done_flag",
    ]
  )
  assert not _has_anomalies(records)


@pytest.mark.parametrize(
  "reason",
  [
    "no_legal_moves_for_current_player",
    "step_exception:RuntimeError",
    "step_exception:ValueError",
    "error_reset",
    "error_reset_unexpected_state",
  ],
)
def test_has_anomalies_true_for_invariant_and_engine_errors(reason: str) -> None:
  records = _make_records(["status:finished", reason])
  assert _has_anomalies(records)


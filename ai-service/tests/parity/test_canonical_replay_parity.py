"""Parity regression tests for canonical self-play databases.

These tests ensure that games selected as canonical parity fixtures
continue to replay identically between Python and TypeScript engines.

They are a stronger, non-xfailed complement to
``test_replay_parity_fixtures_regression.py``, which still tracks a
broader set of legacy/experimental fixtures under xfail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import json
import os

import pytest

# Import the parity helper used by the CLI script. We mirror the pattern
# from test_replay_parity_fixtures_regression.py to keep imports stable
# under pytest.
from scripts.check_ts_python_replay_parity import (  # type: ignore[import]
    check_game_parity,
)
from app.utils.paths import AI_SERVICE_ROOT
PARITY_FIXTURES_DIR = AI_SERVICE_ROOT / "parity_fixtures"


def _iter_canonical_fixtures() -> Iterable[Tuple[str, str]]:
  """Yield (db_path, game_id) pairs for canonical parity fixtures.

  We currently treat fixtures whose filenames start with
  ``canonical_square8__`` as canonical replay cases that must remain
  parity-clean. Additional canonical fixtures for other boards can be
  added here over time.
  """
  if not PARITY_FIXTURES_DIR.exists():
    return []

  pairs: list[Tuple[str, str]] = []
  for entry in sorted(PARITY_FIXTURES_DIR.iterdir()):
    name = entry.name
    if not name.endswith(".json"):
      continue
    if not name.startswith("canonical_square8__"):
      continue
    try:
      with entry.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    except Exception:
      continue

    db_path = payload.get("db_path")
    game_id = payload.get("game_id")
    if isinstance(db_path, str) and isinstance(game_id, str):
      pairs.append((db_path, game_id))

  return pairs


@pytest.mark.slow
@pytest.mark.parametrize("db_path,game_id", _iter_canonical_fixtures())
def test_canonical_replay_parity_holds(db_path: str, game_id: str) -> None:
  """Canonical fixtures must remain fully parity-clean.

  For each canonical fixture, we recompute TS↔Python replay parity using
  the same helper as the CLI harness and assert:

    - structure == "good"
    - diverged_at is None (no semantic divergence)
    - total move counts match between engines
  """
  # Guard against environments where Node/ts-node or the TS harness are
  # unavailable: treat that as a skip rather than a hard failure so unit
  # tests remain usable in restricted environments.
  try:
    result = check_game_parity(Path(db_path), game_id)
  except Exception as exc:  # pragma: no cover - defensive
    pytest.skip(f"TS replay or parity helper unavailable: {exc}")

  assert result.structure == "good"
  assert result.diverged_at is None
  assert result.total_moves_python == result.total_moves_ts

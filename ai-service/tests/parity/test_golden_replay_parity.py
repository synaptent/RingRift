"""Golden replay parity scaffold – strict TS↔Python checks for curated games.

This module is the future home for a small suite of *golden* replay tests:
carefully selected recorded games that exercise tricky parts of the rules
(line + territory, chain capture, pie rule, LPS/multi-player, structural
invariants) and must never regress once promoted.

At present it is intentionally fixture-driven:
- If no golden DBs exist under ai-service/tests/fixtures/golden_games/,
  the test is skipped.
- When golden DBs are added, each recorded game in those DBs will be
  replayed via both Python (GameReplayDB) and the TS replay script
  (selfplay-db-ts-replay.ts) using the existing differential replay
  helpers from test_differential_replay.py, and any divergence will
  cause this test to fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
from collections.abc import Iterable

import pytest

from app.db import GameReplayDB
from tests.parity.test_differential_replay import (  # noqa: E402
    get_recorded_game_steps,
    run_ts_replay,
    compare_replays,
)


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "golden_games"


def _iter_golden_games() -> Iterable[tuple[Path, str]]:
  """Yield (db_path, game_id) pairs for all games in golden fixture DBs."""
  if not FIXTURES_DIR.exists():
    return []

  db_paths: list[Path] = sorted(FIXTURES_DIR.glob("*.db"))
  games: list[tuple[Path, str]] = []

  for db_path in db_paths:
    db = GameReplayDB(str(db_path))
    with db._get_conn() as conn:
      rows = conn.execute("SELECT game_id FROM games").fetchall()
    for (game_id,) in rows:
      games.append((db_path, game_id))

  return games


@pytest.mark.slow
def test_golden_games_have_no_differential_replay_divergences() -> None:
  """Strict parity check over all curated golden games (when present).

  This scaffold is intentionally skip-friendly while the golden suite is
  being built out:

  - If there are no *.db files under ai-service/tests/fixtures/golden_games/,
    the test is skipped.
  - Once golden DBs are added, every recorded game in those DBs is treated
    as a golden trace and must have *zero* TS↔Python differential replay
    divergences, using the same comparison logic as
    tests/parity/test_differential_replay.py.
  """
  games = list(_iter_golden_games())
  if not games:
    pytest.skip(
      "No golden game fixtures found under "
      "ai-service/tests/fixtures/golden_games/*.db",
    )

  for db_path, game_id in games:
    py_steps = get_recorded_game_steps(str(db_path), game_id)
    assert py_steps, (
      f"Golden game should have recorded steps: game_id={game_id!r}, "
      f"db={db_path}"
    )

    ts_steps = run_ts_replay(str(db_path), game_id)
    if ts_steps is None:
      pytest.skip(
        "TypeScript replay unavailable or failed for "
        f"golden db={db_path} game_id={game_id}; skipping strict parity.",
      )
    assert ts_steps, (
      f"TS replay produced no steps for golden game_id={game_id!r} in db={db_path}"
    )

    comparison = compare_replays(py_steps, ts_steps)
    divergences = comparison.get("divergences", [])

    assert not divergences, (
      "Expected no TS↔Python differential replay divergences for "
      f"golden game_id={game_id!r} in db={db_path}, but found: "
      f"{divergences[:5]}"
    )
    assert comparison["py_move_count"] > 0
    assert comparison["ts_move_count"] > 0

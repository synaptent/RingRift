"""Regression harness for TS↔Python replay parity fixtures.

This test suite turns the JSON fixtures produced by
``scripts/check_ts_python_replay_parity.py --emit-fixtures-dir parity_fixtures``
into pytest cases.

Intent:
  - Each fixture captures a concrete (db_path, game_id) where historical
    TS↔Python replay diverged.
  - The *eventual* goal is that replay parity holds for all of these games,
    i.e. ``check_game_parity`` reports no semantic divergence.
  - Today, these tests are marked xfail because the divergences are still
    present; once the Python rules engine is fully aligned with the canonical
    rules spec and TS shared engine, the xfail can be removed to turn these
    into strict regression guards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple
from collections.abc import Iterable

import pytest

# Import the parity helper used by the CLI script
from scripts.check_ts_python_replay_parity import (  # type: ignore[import]
    check_game_parity,
)
from app.utils.paths import AI_SERVICE_ROOT
PARITY_FIXTURES_DIR = AI_SERVICE_ROOT / "parity_fixtures"


def _iter_parity_fixtures() -> Iterable[tuple[str, str]]:
    """Yield (db_path, game_id) pairs from parity fixture JSON files.

    Each fixture written by ``check_ts_python_replay_parity.py`` contains:
      - db_path: absolute path to the GameReplayDB file
      - game_id: the game identifier within that DB
    """
    if not PARITY_FIXTURES_DIR.exists():
        return []

    pairs: list[tuple[str, str]] = []
    for entry in sorted(PARITY_FIXTURES_DIR.iterdir()):
        if not entry.name.endswith(".json"):
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
@pytest.mark.xfail(
    reason=(
        "Known TS↔Python replay divergences in self-play DBs. "
        "Once the Python GameEngine is fully aligned with the canonical "
        "rules spec and TS shared engine, these tests should pass and "
        "the xfail marker can be removed."
    ),
    strict=False,
)
@pytest.mark.parametrize("db_path,game_id", _iter_parity_fixtures())
def test_replay_parity_fixture_eventually_converges(db_path: str, game_id: str) -> None:
    """For each recorded divergence fixture, assert eventual parity.

    We re-use the ``check_game_parity`` helper from the CLI parity script
    to recompute parity for the given (db_path, game_id). The *desired*
    invariant is:

      - no semantic divergence (diverged_at is None), and
      - Python and TS agree on total move counts.

    At present this is xfailed because divergences are still being worked
    through; once they are fixed, this test will start to pass and provide
    a regression guard for replay parity on all historically-problematic
    games.
    """
    # Guard against environments where Node/ts-node or the TS harness are
    # unavailable: in that case, the underlying helper will raise when
    # it cannot spawn the TS replay process. We treat that as a skip.
    try:
        result = check_game_parity(Path(db_path), game_id)
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"TS replay or parity helper unavailable: {exc}")

    # Structure must be good (initial state and snapshots usable).
    assert result.structure == "good"

    # Desired final property once parity is fixed: no semantic divergence
    # and TS/Python agree on total move counts.
    assert result.diverged_at is None
    assert result.total_moves_python == result.total_moves_ts


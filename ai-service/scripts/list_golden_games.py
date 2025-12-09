#!/usr/bin/env python
"""List games contained in the golden replay fixture databases.

This helper inspects all GameReplayDB SQLite files under
ai-service/tests/fixtures/golden_games/ and prints a compact summary for
each recorded game:

- db_path
- game_id
- board_type
- num_players
- winner
- termination_reason
- total_moves

It is intended as a convenience when:
- choosing a game_id to wire into the env-driven golden test
  (RINGRIFT_PARITY_GOLDEN_DB / RINGRIFT_PARITY_GOLDEN_GAME_ID), or
- deciding which golden traces to reference in documentation.

Usage (from ai-service/):

    PYTHONPATH=. python scripts/list_golden_games.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import GameReplayDB  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "ai-service" / "tests" / "fixtures" / "golden_games"

    if not fixtures_dir.exists():
        print(f"[list_golden_games] No fixtures directory at {fixtures_dir}", file=sys.stderr)
        sys.exit(0)

    db_paths: List[Path] = sorted(fixtures_dir.glob("*.db"))
    if not db_paths:
        print(f"[list_golden_games] No *.db files found under {fixtures_dir}", file=sys.stderr)
        sys.exit(0)

    rows = []
    for db_path in db_paths:
        db = GameReplayDB(str(db_path))
        with db._get_conn() as conn:
            games = conn.execute(
                """
                SELECT game_id,
                       board_type,
                       num_players,
                       winner,
                       termination_reason,
                       total_moves
                FROM games
                """,
            ).fetchall()

        for g in games:
            rows.append(
                {
                    "db_path": str(db_path),
                    "game_id": g["game_id"],
                    "board_type": g["board_type"],
                    "num_players": g["num_players"],
                    "winner": g["winner"],
                    "termination_reason": g["termination_reason"],
                    "total_moves": g["total_moves"],
                },
            )

    # Print as JSON to stdout for easy piping/grepping.
    json.dump(rows, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

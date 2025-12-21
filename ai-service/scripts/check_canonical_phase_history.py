from __future__ import annotations

"""
Canonical phase-history checker for GameReplayDB databases.

This script replays every recorded game from a given SQLite DB using the
Python GameEngine.apply_move path, which now enforces a strict
phase→MoveType invariant. Any game that violates the canonical
per-phase move taxonomy (action / skip / no-action) will cause a hard
failure.

Usage:

  PYTHONPATH=. python -m ai-service.scripts.check_canonical_phase_history \\
      --db data/games/canonical_square8_2p.db

Exit codes:
  0 – all games satisfied the canonical phase-history invariant
  1 – at least one game violated the invariant
"""

import argparse
import sys

from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine


def check_db(db_path: str, game_ids: list[str] | None = None) -> int:
    db = GameReplayDB(db_path)

    if game_ids is None:
        with db._get_conn() as conn:  # type: ignore[attr-defined]
            rows = conn.execute("SELECT game_id FROM games").fetchall()
            game_ids = [row["game_id"] for row in rows]

    violations = 0

    for game_id in game_ids:
        # Start from initial state
        state = db.get_initial_state(game_id)
        if state is None:
            print(f"[CANONICAL-PHASE] game {game_id}: missing initial state", file=sys.stderr)
            violations += 1
            continue

        moves = db.get_moves(game_id)
        for idx, move in enumerate(moves):
            try:
                state = GameEngine.apply_move(state, move, trace_mode=True)
            except Exception as e:
                print(
                    f"[CANONICAL-PHASE] game {game_id}: "
                    f"violation at move {idx} type={move.type} player={move.player}: {e}",
                    file=sys.stderr,
                )
                violations += 1
                break

    return 0 if violations == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check canonical phase-history invariants for a replay DB")
    parser.add_argument("--db", required=True, help="Path to GameReplayDB SQLite file")
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help="Specific game_id(s) to check (can be provided multiple times). When omitted, all games are checked.",
    )
    args = parser.parse_args(argv)

    return check_db(args.db, args.game_ids)


if __name__ == "__main__":
    raise SystemExit(main())

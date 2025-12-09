#!/usr/bin/env python
"""
inspect_replay_db.py
=====================

Lightweight CLI for inspecting GameReplayDB SQLite databases.

Features:
- List games with filters (board type, players, winner, termination, source, move count).
- Show aggregated DB stats (total games/moves, counts by board/status/termination).
- Classify structural recording quality per game (good / mid_snapshot / internal_inconsistent)
  using the same logic as cleanup_useless_replay_dbs.py.

Usage (from ai-service/):

  PYTHONPATH=. python scripts/inspect_replay_db.py --db data/games/selfplay.db --stats

  PYTHONPATH=. python scripts/inspect_replay_db.py \\
    --db data/games/selfplay.db \\
    --board-type square8 --num-players 2 --min-moves 10 --limit 20

  PYTHONPATH=. python scripts/inspect_replay_db.py \\
    --db logs/cmaes/runs/xyz/games.db --classify-structure --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from app.db.game_replay import GameReplayDB, BoardType

try:
    # Reuse the existing structural classifier for consistency with cleanup_useless_replay_dbs.
    from scripts.cleanup_useless_replay_dbs import classify_game_structure
except Exception:  # pragma: no cover - defensive fallback
    classify_game_structure = None  # type: ignore[assignment]


def _parse_board_type(name: Optional[str]) -> Optional[BoardType]:
    if not name:
        return None
    name = name.lower()
    try:
        return BoardType(name)
    except ValueError:
        raise SystemExit(f"Unknown board type: {name!r} (expected square8|square19|hexagonal)")


def _print_stats(db: GameReplayDB) -> None:
    stats = db.get_stats()

    print("=== GameReplayDB Stats ===")
    print(f"Schema version: {stats.get('schema_version', '?')}")
    print(f"Total games:   {stats.get('total_games', 0)}")
    print(f"Total moves:   {stats.get('total_moves', 0)}")
    print()
    print("Games by board type:")
    for bt, count in sorted(stats.get("games_by_board_type", {}).items()):
        print(f"  {bt}: {count}")
    print()
    print("Games by status:")
    for status, count in sorted(stats.get("games_by_status", {}).items()):
        print(f"  {status}: {count}")
    print()
    print("Games by termination reason:")
    for reason, count in sorted(stats.get("games_by_termination", {}).items()):
        label = reason or "(none)"
        print(f"  {label}: {count}")
    print()


def _format_row(cols: List[str], widths: List[int]) -> str:
    return "  ".join(col.ljust(w) for col, w in zip(cols, widths))


def _print_games_table(games: List[Dict[str, Any]]) -> None:
    if not games:
        print("No games matched the specified filters.")
        return

    headers = [
        "game_id",
        "board",
        "players",
        "status",
        "winner",
        "reason",
        "moves",
        "source",
        "created_at",
    ]

    rows: List[List[str]] = []
    for g in games:
        rows.append(
            [
                str(g.get("game_id", "")),
                str(g.get("board_type", "")),
                str(g.get("num_players", "")),
                str(g.get("game_status", "")),
                str(g.get("winner", "")),
                str(g.get("termination_reason", "")),
                str(g.get("total_moves", "")),
                str(g.get("source", "")),
                str(g.get("created_at", "")),
            ]
        )

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if len(val) > widths[i]:
                widths[i] = len(val)

    print(_format_row(headers, widths))
    print(_format_row(["-" * len(h) for h in headers], widths))
    for row in rows:
        print(_format_row(row, widths))
    print(f"\nTotal listed: {len(games)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a GameReplayDB SQLite database (games.db or selfplay.db).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.getenv("GAME_REPLAY_DB_PATH", "data/games/selfplay.db"),
        help="Path to GameReplayDB SQLite file (default: GAME_REPLAY_DB_PATH or data/games/selfplay.db).",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal"],
        help="Filter by board type.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter by number of players.",
    )
    parser.add_argument(
        "--winner",
        type=int,
        help="Filter by winning player seat (1-based).",
    )
    parser.add_argument(
        "--termination-reason",
        type=str,
        help="Filter by termination_reason.",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Filter by source tag (e.g. selfplay_soak, cmaes_optimization).",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        help="Minimum total_moves.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        help="Maximum total_moves.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of games to list (default: 50).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset into result set for paging.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print aggregated database stats before listing games.",
    )
    parser.add_argument(
        "--classify-structure",
        action="store_true",
        help=(
            "Classify structural recording quality per listed game using the same "
            "rules as cleanup_useless_replay_dbs.py (good/mid_snapshot/internal_inconsistent)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of a human-readable table.",
    )

    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        raise SystemExit(f"Database not found: {db_path}")

    db = GameReplayDB(db_path)

    if args.stats:
        _print_stats(db)

    board_type = _parse_board_type(args.board_type)

    filters: Dict[str, Any] = {
        "board_type": board_type,
        "num_players": args.num_players,
        "winner": args.winner,
        "termination_reason": args.termination_reason,
        "source": args.source,
        "min_moves": args.min_moves,
        "max_moves": args.max_moves,
        "limit": args.limit,
        "offset": args.offset,
    }

    games = db.query_games(**filters)

    structure_by_game: Dict[str, str] = {}
    if args.classify_structure and classify_game_structure is not None:
        for meta in games:
            game_id = meta.get("game_id")
            if not game_id:
                continue
            try:
                structure, _reason = classify_game_structure(db, game_id)  # type: ignore[operator]
            except Exception:  # pragma: no cover - defensive
                structure = "error"
            structure_by_game[game_id] = structure

    if args.json:
        payload: Dict[str, Any] = {
            "db_path": os.path.abspath(db_path),
            "filters": {
                "board_type": args.board_type,
                "num_players": args.num_players,
                "winner": args.winner,
                "termination_reason": args.termination_reason,
                "source": args.source,
                "min_moves": args.min_moves,
                "max_moves": args.max_moves,
                "limit": args.limit,
                "offset": args.offset,
            },
            "total_listed": len(games),
            "games": games,
        }
        if structure_by_game:
            payload["structure"] = structure_by_game
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_games_table(games)
        if structure_by_game:
            print("\nStructural classification (per game_id):")
            for gid, cls in sorted(structure_by_game.items()):
                print(f"  {gid}: {cls}")


if __name__ == "__main__":  # pragma: no cover
    main()

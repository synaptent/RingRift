#!/usr/bin/env python3
"""Backward-compatible auto-promotion entrypoint.

The project now promotes models by publishing stable *best* aliases and syncing
them to the cluster (instead of rewriting ladder config files in-place).

This script remains as a thin wrapper so existing daemons/cronjobs can keep
calling `auto_promote_best_models.py` while the underlying implementation lives
in `scripts/model_promotion_manager.py`.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"


def _best_model_for_config(board_type: str, num_players: int) -> Optional[Dict[str, Any]]:
    if not ELO_DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(ELO_DB_PATH))
    try:
        cur = conn.cursor()
        row = cur.execute(
            """
            SELECT model_id, rating, games_played
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            ORDER BY rating DESC
            LIMIT 1
            """,
            (board_type, int(num_players)),
        ).fetchone()
        if not row:
            return None
        return {"model_id": row[0], "rating": float(row[1]), "games_played": int(row[2])}
    finally:
        conn.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted (no changes).")
    parser.add_argument("--run", action="store_true", help="Publish + sync best-model aliases.")
    parser.add_argument("--min-games", type=int, default=20, help="Minimum Elo games required (default: 20).")
    parser.add_argument("--board", choices=["square8", "square19", "hexagonal"], default=None)
    parser.add_argument("--players", type=int, choices=[2, 3, 4], default=None)
    parser.add_argument("--restart-p2p", action="store_true", help="Restart p2p orchestrator after sync.")
    parser.add_argument("--update-sandbox-config", action="store_true", help="Write src/shared/config/ai_models.json.")

    args = parser.parse_args(argv)

    if not args.run:
        # Default to dry-run.
        board_types = [args.board] if args.board else ["square8", "square19", "hexagonal"]
        player_counts = [args.players] if args.players else [2, 3, 4]
        for board in board_types:
            for players in player_counts:
                best = _best_model_for_config(board, players)
                if not best:
                    continue
                if best["games_played"] < int(args.min_games):
                    continue
                print(
                    f"{board} {players}p: {best['model_id']} "
                    f"(Elo={best['rating']:.0f}, games={best['games_played']})"
                )
        return 0

    cmd = [
        sys.executable,
        str((AI_SERVICE_ROOT / "scripts" / "model_promotion_manager.py").resolve()),
        "--full-pipeline",
        "--min-games",
        str(int(args.min_games)),
    ]
    if args.restart_p2p:
        cmd.append("--restart-p2p")
    if args.update_sandbox_config:
        cmd.append("--update-sandbox-config")

    proc = subprocess.run(cmd, cwd=str(AI_SERVICE_ROOT), text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())


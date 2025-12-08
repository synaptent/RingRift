#!/usr/bin/env python
from __future__ import annotations

"""
End-to-end canonical self-play generator and gate.

This script ties together three pieces for a single board type:

  1. Run a small canonical self-play soak to populate a GameReplayDB.
  2. Run the TS↔Python replay parity harness on that DB.
  3. Run the lightweight canonical history validator over every game.

A database is considered "canonical" for training only if:
  - The parity gate passes (no structural issues, no semantic divergence),
  - The canonical history validator reports zero issues for all games, and
  - At least one game was recorded.

Current implementation:
  - Uses scripts/run_canonical_selfplay_parity_gate.py for step (1) + (2).
  - Uses app.rules.history_validation.validate_canonical_history_for_game
    over GameReplayDB for step (3).

Typical usage (from ai-service/):

  PYTHONPATH=. python scripts/generate_canonical_selfplay.py \\
    --board-type square19 \\
    --num-games 50 \\
    --db data/games/canonical_square19.db \\
    --summary db_health.canonical_square19.json

The summary JSON includes:
  - board_type, db_path
  - parity_gate (raw summary from run_canonical_selfplay_parity_gate)
  - canonical_history (games_checked, non_canonical_games, sample_issues)
  - canonical_ok (boolean)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.db.game_replay import GameReplayDB
from app.rules.history_validation import validate_canonical_history_for_game


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: List[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Ensure PYTHONPATH points at ai-service root when invoked from repo root.
    env.setdefault("PYTHONPATH", str(AI_SERVICE_ROOT))
    # Keep OpenMP usage conservative by default.
    env.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or AI_SERVICE_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    return proc


def run_selfplay_and_parity(
    board_type: str,
    num_games: int,
    db_path: Path,
    num_players: int,
    hosts: str | None = None,
) -> Dict[str, Any]:
    """
    Delegate to run_canonical_selfplay_parity_gate.py to:
      - run a small canonical self-play soak, and
      - run TS↔Python parity on the resulting DB.
    """
    summary_path = db_path.with_suffix(db_path.suffix + ".parity_gate.json")

    cmd = [
        sys.executable,
        "scripts/run_canonical_selfplay_parity_gate.py",
        "--board-type",
        board_type,
        "--num-games",
        str(num_games),
        "--num-players",
        str(num_players),
        "--db",
        str(db_path),
        "--summary",
        str(summary_path),
    ]
    if hosts:
        cmd += ["--hosts", hosts]

    proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT)

    parity_summary: Dict[str, Any]
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                parity_summary = json.load(f)
        except Exception:
            parity_summary = {
                "error": "failed_to_load_parity_summary_file",
                "summary_path": str(summary_path),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
    else:
        # Fallback to parsing stdout if the summary file was not written.
        try:
            parity_summary = json.loads(proc.stdout)
        except Exception:
            parity_summary = {
                "error": "failed_to_parse_parity_summary_stdout",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

    parity_summary["returncode"] = proc.returncode
    return parity_summary


def run_canonical_history_check(db_path: Path) -> Dict[str, Any]:
    """Run validate_canonical_history_for_game over all games in the DB."""
    db = GameReplayDB(str(db_path))

    with db._get_conn() as conn:  # type: ignore[attr-defined]
        rows = conn.execute("SELECT game_id FROM games").fetchall()
        game_ids = [row["game_id"] for row in rows]

    issues_by_game: Dict[str, List[Dict[str, Any]]] = {}

    for gid in game_ids:
        report = validate_canonical_history_for_game(db, gid)
        if not report.is_canonical:
            issues_by_game[gid] = [
                {
                    "move_number": issue.move_number,
                    "phase": issue.phase,
                    "move_type": issue.move_type,
                    "reason": issue.reason,
                }
                for issue in report.issues
            ]

    games_checked = len(game_ids)
    non_canonical_games = len(issues_by_game)

    # For brevity, surface at most a few sample issues.
    sample_issues: Dict[str, Any] = {}
    for gid, issues in list(issues_by_game.items())[:5]:
        sample_issues[gid] = issues[:5]

    return {
        "games_checked": games_checked,
        "non_canonical_games": non_canonical_games,
        "sample_issues": sample_issues,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a canonical self-play GameReplayDB for a board type and "
            "gate it on TS↔Python parity plus canonical history constraints."
        )
    )
    parser.add_argument(
        "--board-type",
        required=True,
        choices=["square8", "square19", "hexagonal"],
        help="Board type to run self-play on.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=32,
        help="Number of self-play games to run (default: 32).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Number of players for self-play (default: 2).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help=(
            "Path to the GameReplayDB SQLite file to write. "
            "Defaults to data/games/canonical_<board>.db."
        ),
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to write the combined canonical summary JSON.",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="Comma-separated hosts for distributed self-play soak; when set, delegates to run_distributed_selfplay_soak.",
    )

    args = parser.parse_args(argv)

    board_type: str = args.board_type
    num_games: int = args.num_games
    num_players: int = args.num_players
    hosts: str | None = args.hosts

    if args.db:
        db_path = Path(args.db).resolve()
    else:
        db_name = f"canonical_{board_type}_{num_players}p.db"
        db_path = (AI_SERVICE_ROOT / "data" / "games" / db_name).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        parity_summary = run_selfplay_and_parity(board_type, num_games, db_path, num_players, hosts)
    except Exception as e:  # pragma: no cover - debug hook
        payload = {
            "board_type": board_type,
            "num_players": num_players,
            "db_path": str(db_path),
            "num_games_requested": num_games,
            "error": str(e),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise

    # Determine if the parity gate itself passed.
    passed_gate = bool(parity_summary.get("passed_canonical_parity_gate"))
    parity_rc = int(parity_summary.get("soak_returncode", 0) or 0)

    canonical_history = {}
    canonical_ok = False

    if db_path.exists() and passed_gate and parity_rc == 0:
        canonical_history = run_canonical_history_check(db_path)
        games_checked = int(canonical_history.get("games_checked", 0) or 0)
        non_canonical = int(canonical_history.get("non_canonical_games", 0) or 0)
        canonical_ok = passed_gate and parity_rc == 0 and games_checked > 0 and non_canonical == 0

    summary: Dict[str, Any] = {
        "board_type": board_type,
        "num_players": num_players,
        "db_path": str(db_path),
        "num_games_requested": num_games,
        "parity_gate": parity_summary,
        "canonical_history": canonical_history,
        "canonical_ok": bool(canonical_ok),
    }

    if args.summary:
        summary_path = Path(args.summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if canonical_ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

"""
Lightweight invariant soak across TS orchestrator and Python GameEngine.

This script is intended to be CI-friendly and fast:
  - For each available canonical_* GameReplayDB, it replays a small sample of
    games via Python GameEngine.apply_move(trace_mode=True) and asserts that
    no invariants fail (any exception aborts the run).
  - Optionally, it invokes the TS orchestrator soak script
    (scripts/run-orchestrator-soak.ts) in a short CI profile and treats any
    invariant violations or crashes as failure.

Usage (from ai-service/):

  # Python-only invariant soak on canonical DBs
  PYTHONPATH=. python scripts/run_invariant_soak.py

  # Include TS orchestrator soak as well
  PYTHONPATH=. python scripts/run_invariant_soak.py --with-ts
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List

from app.db import GameReplayDB
from app.game_engine import GameEngine


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(f"[invariant-soak] {msg}")


def _python_soak_for_db(db_path: Path, max_games: int) -> bool:
    """Replay up to max_games from db_path and ensure no engine invariant fails."""
    _log(f"Python soak: {db_path}")
    db = GameReplayDB(str(db_path))

    with db._get_conn() as conn:  # type: ignore[attr-defined]
        rows = conn.execute("SELECT game_id FROM games").fetchall()
        game_ids = [row["game_id"] for row in rows]

    if not game_ids:
        _log(f"  Skipping (no games found).")
        return True

    random.shuffle(game_ids)
    sample_ids = game_ids[: max_games or len(game_ids)]

    for gid in sample_ids:
        _log(f"  Replaying game {gid}")
        state = db.get_initial_state(gid)
        if state is None:
            _log(f"    ERROR: missing initial state for {gid}")
            return False

        moves = db.get_moves(gid)
        try:
            for mv in moves:
                state = GameEngine.apply_move(state, mv, trace_mode=True)
        except Exception as exc:
            _log(f"    ERROR: invariant violation while replaying {gid}: {exc}")
            return False

    _log("  OK")
    return True


def run_python_invariant_soak(max_games_per_db: int) -> bool:
    """Run Python invariant soak across all canonical_*.db under data/games."""
    games_dir = AI_SERVICE_ROOT / "data" / "games"
    if not games_dir.exists():
        _log("No data/games directory; skipping Python soak.")
        return True

    db_paths = sorted(p for p in games_dir.glob("canonical_*.db") if p.is_file())
    if not db_paths:
        _log("No canonical_*.db files found; nothing to soak.")
        return True

    all_ok = True
    for db_path in db_paths:
        ok = _python_soak_for_db(db_path, max_games=max_games_per_db)
        if not ok:
            all_ok = False
            break
    return all_ok


def run_ts_invariant_soak() -> bool:
    """
    Invoke the TS orchestrator soak in a short CI profile.

    This delegates to scripts/run-orchestrator-soak.ts and relies on its
    own invariant checks (isANMState, S-invariant, etc.). Any non-zero
    exit code is treated as a failure.
    """
    _log("TS soak: scripts/run-orchestrator-soak.ts --profile=ci-short")

    cmd = [
        sys.executable,
        "-m",
        "node",
    ]

    # Prefer invoking via npx ts-node if available; fall back to node if the
    # script is compiled. We do not attempt to detect this dynamically here;
    # callers can wrap this script in a small shell wrapper if needed.
    # To avoid environment flakiness in this harness, we instead call `node`
    # via a subprocess directly.
    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")
    env.setdefault("NODE_ENV", "production")

    # Use `node` directly; in many environments this will be available on PATH.
    proc = subprocess.run(
        [
            "node",
            "--require",
            "ts-node/register",
            "scripts/run-orchestrator-soak.ts",
            "--profile",
            "ci-short",
            "--fail-on-violation",
        ],
        cwd=str(AI_SERVICE_ROOT.parent),
        env=env,
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        _log("TS orchestrator soak failed.")
        _log(proc.stdout)
        _log(proc.stderr)
        return False

    _log("TS orchestrator soak OK.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a small invariant soak across canonical_* DBs (Python) and optionally TS orchestrator."
    )
    parser.add_argument(
        "--max-games-per-db",
        type=int,
        default=3,
        help="Maximum number of games per canonical DB to replay in Python (default: 3).",
    )
    parser.add_argument(
        "--with-ts",
        action="store_true",
        help="Also run the TS orchestrator soak (scripts/run-orchestrator-soak.ts --profile=ci-short).",
    )
    args = parser.parse_args(argv)

    ok_py = run_python_invariant_soak(max_games_per_db=args.max_games_per_db)
    ok_ts = True

    if args.with_ts:
        ok_ts = run_ts_invariant_soak()

    return 0 if ok_py and ok_ts else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

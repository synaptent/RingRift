#!/usr/bin/env python
"""
Canonical self-play + parity gate driver.

This script automates a minimal end-to-end check for a given board type:

  1. Run a small Python self-play soak using the canonical GameEngine
     and record completed games into a GameReplayDB.
  2. Run the TS↔Python replay parity harness on that DB.
  3. Emit a compact JSON summary describing whether the DB passes the
     "canonical parity gate" (no structural issues, no semantic divergence).

Typical usage (from ai-service/):

  PYTHONPATH=. python scripts/run_canonical_selfplay_parity_gate.py \\
    --board-type square8 \\
    --num-games 20 \\
    --db data/games/selfplay_square8_parity_gate.db \\
    --summary parity_gate.square8.json

The intent is to make it easy to:
  - Generate fresh, canonical self-play DBs per board type, and
  - Gate training pipelines on those DBs passing basic parity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: list[str], cwd: Path | None = None, env_overrides: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess and return the completed process."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or AI_SERVICE_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    return proc


def run_selfplay_soak(board_type: str, num_games: int, db_path: Path, seed: int, max_moves: int) -> Dict[str, Any]:
    """Run a small Python self-play soak and record games to db_path."""
    logs_dir = AI_SERVICE_ROOT / "logs" / "selfplay"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / f"soak.{board_type}.parity_gate.summary.json"
    jsonl_path = logs_dir / f"soak.{board_type}.parity_gate.jsonl"

    cmd = [
        sys.executable,
        "scripts/run_self_play_soak.py",
        "--num-games",
        str(num_games),
        "--board-type",
        board_type,
        "--engine-mode",
        "mixed",
        "--num-players",
        "2",
        "--max-moves",
        str(max_moves),
        "--seed",
        str(seed),
        "--record-db",
        str(db_path),
        "--log-jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_path),
    ]

    # Enable strict invariant by default so soak respects ANM constraints.
    env_overrides = {
        "RINGRIFT_STRICT_NO_MOVE_INVARIANT": "1",
        "PYTHONPATH": str(AI_SERVICE_ROOT),
        # Keep OpenMP usage conservative for long-running soaks and
        # avoid environment-specific SHM issues on some platforms.
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
    }

    proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT, env_overrides=env_overrides)
    result: Dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "summary_path": str(summary_path),
    }
    return result


def run_parity_check(db_path: Path) -> Dict[str, Any]:
    """Run the TS↔Python parity harness on a single DB and return the parsed summary."""
    cmd = [
        sys.executable,
        "scripts/check_ts_python_replay_parity.py",
        "--db",
        str(db_path),
    ]
    env_overrides = {"PYTHONPATH": str(AI_SERVICE_ROOT)}
    proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT, env_overrides=env_overrides)

    summary: Dict[str, Any]
    try:
        summary = json.loads(proc.stdout)
    except Exception:
        summary = {
            "error": "failed_to_parse_parity_summary",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    summary["returncode"] = proc.returncode
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical Python self-play for a board type and gate the resulting GameReplayDB on TS↔Python parity."
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
        default=20,
        help="Number of self-play games to run for the gate (default: 20).",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the GameReplayDB SQLite file to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for the soak run.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game before forced termination (default: 200).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to write the parity gate JSON summary. When omitted, prints to stdout only.",
    )

    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    soak_result = run_selfplay_soak(args.board_type, args.num_games, db_path, args.seed, args.max_moves)

    parity_summary = run_parity_check(db_path)

    # Basic gate: soak must succeed and parity must be clean.
    #
    # We require:
    #   - soak_returncode == 0 (Python self-play soak did not abort), and
    #   - no structural issues and no semantic divergences, and
    #   - at least one game was parity-checked.
    passed = False
    soak_rc = soak_result.get("returncode")
    if (
        soak_rc == 0
        and "error" not in parity_summary
        and parity_summary.get("returncode") == 0
    ):
        struct = int(parity_summary.get("games_with_structural_issues", 0))
        sem = int(parity_summary.get("games_with_semantic_divergence", 0))
        total_checked = int(parity_summary.get("total_games_checked", 0))
        passed = struct == 0 and sem == 0 and total_checked > 0

    gate_summary: Dict[str, Any] = {
        "board_type": args.board_type,
        "db_path": str(db_path),
        "num_games": args.num_games,
        "seed": args.seed,
        "max_moves": args.max_moves,
        "soak_returncode": soak_result.get("returncode"),
        "parity_summary": parity_summary,
        "passed_canonical_parity_gate": bool(passed),
    }

    if args.summary:
        summary_path = Path(args.summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(gate_summary, f, indent=2, sort_keys=True)

    print(json.dumps(gate_summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

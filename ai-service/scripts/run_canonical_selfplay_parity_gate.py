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
from typing import Any, Dict, List

from app.models import BoardType
from app.training.env import get_theoretical_max_moves


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(
    cmd: list[str], cwd: Path | None = None, env_overrides: Dict[str, str] | None = None
) -> subprocess.CompletedProcess:
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


def run_selfplay_soak(
    board_type: str, num_games: int, db_path: Path, seed: int, max_moves: int, num_players: int
) -> Dict[str, Any]:
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
        str(num_players),
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
        # Enforce TS↔Python parity during recording; abort early on any divergence.
        # This prevents non-canonical games (e.g., actor mismatches) from entering
        # the canonical DB in the first place.
        "RINGRIFT_PARITY_VALIDATION": "strict",
        # Ensure host applies required bookkeeping/no-op moves for the same actor
        # in line/territory phases, matching TS orchestration.
        "RINGRIFT_FORCE_BOOKKEEPING_MOVES": "1",
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
    """Run the TS↔Python parity harness on a single DB and return the parsed summary.

    This always invokes the parity script in **canonical** mode with
    ``view = post_move`` so that:

      - ``passed_canonical_parity_gate`` in the returned JSON reflects the
        canonical gate status for this DB, and
      - the process return code is non-zero whenever the canonical gate fails.
    """
    cmd = [
        sys.executable,
        "scripts/check_ts_python_replay_parity.py",
        "--db",
        str(db_path),
        "--mode",
        "canonical",
        "--view",
        "post_move",
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


def run_parity_checks(db_paths: list[Path]) -> Dict[str, Any]:
    """Run parity on multiple DBs and aggregate results."""
    summaries: list[Dict[str, Any]] = []
    all_pass = True
    for db_path in db_paths:
        summary = run_parity_check(db_path)
        summaries.append({"db": str(db_path), "summary": summary})
        rc = summary.get("returncode", 1)
        struct = int(summary.get("games_with_structural_issues", 0))
        sem = int(summary.get("games_with_semantic_divergence", 0))
        total_checked = int(summary.get("total_games_checked", 0))
        if rc != 0 or struct > 0 or sem > 0 or total_checked == 0:
            all_pass = False
    return {"all_pass": all_pass, "per_db": summaries}


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
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Number of players for the self-play soak (default: 2).",
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
        default=0,
        help=(
            "Maximum moves per game before forced termination. "
            "Use 0 to auto-select the theoretical max for the board/player count "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="Comma-separated hosts for distributed self-play soak; when set, delegates to run_distributed_selfplay_soak for the specified board/num-players. Default: None (local soak).",
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

    parity_summary: Dict[str, Any] | Dict[str, Any]
    soak_result: Dict[str, Any] = {}
    dbs_to_check: list[Path] = [db_path]

    # Auto-select max_moves when not provided.
    if args.max_moves and args.max_moves > 0:
        max_moves = args.max_moves
    else:
        bt_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_enum = bt_map[args.board_type]
        max_moves = get_theoretical_max_moves(board_enum, args.num_players)

    if args.hosts:
        hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
        output_dir = db_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # Delegate to distributed soak runner with filters
        try:
            output_dir_arg = str(output_dir.relative_to(AI_SERVICE_ROOT))
        except ValueError:
            output_dir_arg = str(output_dir)

        cmd = [
            sys.executable,
            "scripts/run_distributed_selfplay_soak.py",
            "--games-per-config",
            str(args.num_games),
            "--hosts",
            ",".join(hosts),
            "--output-dir",
            output_dir_arg,
            "--board-types",
            args.board_type,
            "--num-players",
            str(args.num_players),
            "--base-seed",
            str(args.seed),
            "--max-parallel-per-host",
            "2",
        ]
        proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT)
        soak_result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "distributed": True,
        }
        # Collect DBs produced for this config
        dbs_to_check = list(output_dir.glob(f"selfplay_{args.board_type}_{args.num_players}p_*.db"))
        if not dbs_to_check:
            parity_summary = {"error": "no_db_produced", "returncode": 1}
        else:
            parity_summary = run_parity_checks(dbs_to_check)
    else:
        soak_result = run_selfplay_soak(
            args.board_type, args.num_games, db_path, args.seed, max_moves, args.num_players
        )
        parity_summary = run_parity_check(db_path)

    # Basic gate: soak must succeed and the canonical parity gate must pass.
    #
    # We require:
    #   - soak_returncode == 0 (Python self-play soak did not abort), and
    #   - parity harness reports passed_canonical_parity_gate == True, and
    #   - underlying parity process exits with code 0.
    #
    # For distributed runs (args.hosts), run_parity_checks() aggregates the
    # per-DB parity summaries; we treat all_pass == True as a requirement that
    # every DB passed its canonical parity gate.
    passed = False
    soak_rc = soak_result.get("returncode")
    if soak_rc == 0 and "error" not in parity_summary:
        if args.hosts:
            passed = bool(parity_summary.get("all_pass"))
        else:
            parity_rc = int(parity_summary.get("returncode", 1))
            passed_gate = bool(parity_summary.get("passed_canonical_parity_gate"))
            passed = parity_rc == 0 and passed_gate

    gate_summary: Dict[str, Any] = {
        "board_type": args.board_type,
        "num_players": args.num_players,
        "db_path": str(db_path),
        "db_paths_checked": [str(p) for p in dbs_to_check],
        "num_games": args.num_games,
        "seed": args.seed,
        "max_moves": max_moves,
        "hosts": args.hosts.split(",") if args.hosts else None,
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

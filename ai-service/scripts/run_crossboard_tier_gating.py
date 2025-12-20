#!/usr/bin/env python
"""Cross-board tier gating wrapper for RingRift difficulty tiers.

This script composes three pipeline steps into one:

1) Optional cross-board non-regression check via run_parity_promotion_gate.py.
2) Canonical square8 2p tier gate + perf via run_full_tier_gating.py.
3) Aggregate into a single cross-board promotion report.

The parity promotion gate is treated as a *pre-gate*: if it runs and fails,
promotion is blocked regardless of the tier-specific decision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


PARITY_GATE_FILENAME = "parity_promotion_gate.json"
CROSSBOARD_GATE_FILENAME = "crossboard_gate_report.json"
FULL_GATE_FILENAME = "gate_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run optional cross-board parity gate, then full tier gating, "
            "and emit a combined cross-board promotion report."
        ),
    )
    parser.add_argument(
        "--tier",
        required=True,
        help="Difficulty tier name (e.g. D2, D4, D6, D8).",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Candidate id (must match training_report.json).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Tier run directory containing training_report.json.",
    )
    parser.add_argument(
        "--board",
        default="square8",
        help="Board identifier for full tier gating (default: square8).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players for full tier gating (default: 2).",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip perf benchmark even when a TierPerfBudget exists.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use lightweight configs suitable for CI smoke runs.",
    )

    # Parity promotion gate args (optional).
    parser.add_argument(
        "--parity-player1",
        type=str,
        default=None,
        help="AI type for candidate in parity gate (e.g. neural_network).",
    )
    parser.add_argument(
        "--parity-player2",
        type=str,
        default=None,
        help="AI type for baseline in parity gate (e.g. neural_network).",
    )
    parser.add_argument(
        "--parity-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for parity candidate.",
    )
    parser.add_argument(
        "--parity-checkpoint2",
        type=str,
        default=None,
        help="Optional checkpoint path for parity baseline.",
    )
    parser.add_argument(
        "--parity-cmaes-weights",
        type=str,
        default=None,
        help="Optional CMA-ES weights path for parity gate.",
    )
    parser.add_argument(
        "--parity-minimax-depth",
        type=int,
        default=3,
        help="Minimax depth when parity AI type is minimax.",
    )
    parser.add_argument(
        "--parity-boards",
        nargs="*",
        default=["square8", "square19", "hex"],
        help="Boards to include in parity gate (default: square8 square19 hex).",
    )
    parser.add_argument(
        "--parity-games-per-matrix",
        type=int,
        default=200,
        help="Games per board matrix for parity gate (default: 200).",
    )
    parser.add_argument(
        "--parity-max-moves",
        type=int,
        default=200,
        help="Max moves per game in parity gate (default: 200).",
    )
    parser.add_argument(
        "--parity-min-ci-lower-bound",
        type=float,
        default=0.5,
        help="Non-inferiority CI lower-bound threshold (default: 0.5).",
    )
    parser.add_argument(
        "--parity-seed",
        type=int,
        default=1,
        help="Seed for parity gate (default: 1).",
    )
    return parser.parse_args(argv)


def _run_parity_gate(args: argparse.Namespace, run_dir: str) -> dict[str, Any] | None:
    if not args.parity_player1 or not args.parity_player2:
        return None

    out_path = os.path.join(run_dir, PARITY_GATE_FILENAME)
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_parity_promotion_gate.py"),
        "--player1",
        args.parity_player1,
        "--player2",
        args.parity_player2,
        "--seed",
        str(args.parity_seed),
        "--games-per-matrix",
        str(4 if args.demo else args.parity_games_per_matrix),
        "--max-moves",
        str(args.parity_max_moves),
        "--min-ci-lower-bound",
        str(args.parity_min_ci_lower_bound),
        "--output-json",
        out_path,
        "--boards",
        *args.parity_boards,
    ]
    if args.parity_checkpoint:
        cmd.extend(["--checkpoint", args.parity_checkpoint])
    if args.parity_checkpoint2:
        cmd.extend(["--checkpoint2", args.parity_checkpoint2])
    if args.parity_cmaes_weights:
        cmd.extend(["--cmaes-weights", args.parity_cmaes_weights])
    if args.parity_minimax_depth:
        cmd.extend(["--minimax-depth", str(args.parity_minimax_depth)])

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_full_gate(args: argparse.Namespace, run_dir: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_full_tier_gating.py"),
        "--tier",
        args.tier,
        "--candidate-id",
        args.candidate_id,
        "--run-dir",
        run_dir,
        "--board",
        args.board,
        "--num-players",
        str(args.num_players),
    ]
    if args.no_perf:
        cmd.append("--no-perf")
    if args.demo:
        cmd.append("--demo")

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    gate_path = os.path.join(run_dir, FULL_GATE_FILENAME)
    with open(gate_path, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_gate_reports(
    parity_report: dict[str, Any] | None,
    full_gate_report: dict[str, Any],
) -> dict[str, Any]:
    parity_pass = True
    if parity_report is not None:
        parity_pass = bool(parity_report.get("gate", {}).get("overall_pass"))

    tier_decision = str(full_gate_report.get("final_decision", "reject")).lower()
    final_decision = "promote" if parity_pass and tier_decision == "promote" else "reject"

    return {
        "tier": full_gate_report.get("tier"),
        "board": full_gate_report.get("board"),
        "num_players": full_gate_report.get("num_players"),
        "candidate_id": full_gate_report.get("candidate_id"),
        "parity_gate": {
            "run": parity_report is not None,
            "overall_pass": parity_pass,
            "result_path": PARITY_GATE_FILENAME if parity_report is not None else None,
        },
        "tier_gate": {
            "overall_pass": bool(full_gate_report.get("evaluation", {}).get("overall_pass")),
            "perf_pass": full_gate_report.get("perf", {}).get("overall_pass"),
            "final_decision": tier_decision,
            "result_path": FULL_GATE_FILENAME,
        },
        "final_decision": final_decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    parity_report = _run_parity_gate(args, run_dir)
    full_gate_report = _run_full_gate(args, run_dir)

    combined = combine_gate_reports(parity_report, full_gate_report)
    out_path = os.path.join(run_dir, CROSSBOARD_GATE_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, sort_keys=True)

    print(f"Wrote cross-board gate report to {out_path}")
    return 0 if combined["final_decision"] == "promote" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


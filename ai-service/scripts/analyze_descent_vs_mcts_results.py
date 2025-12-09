#!/usr/bin/env python3
"""
Aggregate and summarize results from run_descent_vs_mcts_experiment.py runs.

Scans one or more root directories for run_*/results.json files and prints a
compact table comparing:
  - Descent vs MCTS final losses
  - Descent vs heuristic win rate
  - MCTS vs heuristic win rate
  - Descent vs MCTS head-to-head win rate

Usage (from ai-service/):

  PYTHONPATH=. python scripts/analyze_descent_vs_mcts_results.py \
    --root experiments/descent_vs_mcts_sq8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunSummary:
    path: Path
    timestamp: str
    num_games: int
    epochs: int
    eval_games: int
    descent_loss: Optional[float]
    mcts_loss: Optional[float]
    descent_vs_heuristic: Optional[float]
    mcts_vs_heuristic: Optional[float]
    descent_vs_mcts: Optional[float]


def load_run(results_path: Path) -> Optional[RunSummary]:
    try:
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    cfg = data.get("config", {})
    descent = data.get("descent", {})
    mcts = data.get("mcts", {})
    cmp_ = data.get("comparison", {})

    def _get_win_rate(section: dict, key: str) -> Optional[float]:
        sub = section.get(key)
        if not isinstance(sub, dict):
            return None
        val = sub.get("win_rate")
        return float(val) if isinstance(val, (int, float)) else None

    return RunSummary(
        path=results_path.parent,
        timestamp=str(data.get("timestamp", "")),
        num_games=int(cfg.get("num_games", 0) or 0),
        epochs=int(cfg.get("epochs", cfg.get("epochs_per_iter", 0) or 0)),
        eval_games=int(cfg.get("eval_games", 0) or 0),
        descent_loss=float(descent.get("final_loss")) if "final_loss" in descent else None,
        mcts_loss=float(mcts.get("final_loss")) if "final_loss" in mcts else None,
        descent_vs_heuristic=_get_win_rate(descent, "vs_heuristic"),
        mcts_vs_heuristic=_get_win_rate(mcts, "vs_heuristic"),
        descent_vs_mcts=(
            float(cmp_.get("descent_vs_mcts", {}).get("descent_win_rate"))
            if isinstance(cmp_.get("descent_vs_mcts", {}).get("descent_win_rate"), (int, float))
            else None
        ),
    )


def find_runs(root: Path) -> List[RunSummary]:
    runs: List[RunSummary] = []
    for results_path in root.rglob("results.json"):
        summary = load_run(results_path)
        if summary is not None:
            runs.append(summary)
    return runs


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "   n/a "
    return f"{x*100:6.1f}%"


def format_loss(x: Optional[float]) -> str:
    if x is None:
        return "  n/a "
    return f"{x:6.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Descent vs MCTS experiment results.")
    parser.add_argument(
        "--root",
        type=str,
        action="append",
        required=True,
        help=("Root directory to scan for run_*/results.json. Can be specified " "multiple times."),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="timestamp",
        choices=["timestamp", "descent_vs_mcts", "descent_vs_heuristic", "mcts_vs_heuristic"],
        help="Sort key for runs (default: timestamp).",
    )
    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.root]
    all_runs: List[RunSummary] = []
    for root in roots:
        if not root.exists():
            print(f"[warn] root {root} does not exist, skipping")
            continue
        all_runs.extend(find_runs(root))

    if not all_runs:
        print("No runs found.")
        return

    key = args.sort_by
    if key == "timestamp":
        all_runs.sort(key=lambda r: r.timestamp)
    elif key == "descent_vs_mcts":
        all_runs.sort(key=lambda r: (r.descent_vs_mcts or 0.0), reverse=True)
    elif key == "descent_vs_heuristic":
        all_runs.sort(key=lambda r: (r.descent_vs_heuristic or 0.0), reverse=True)
    elif key == "mcts_vs_heuristic":
        all_runs.sort(key=lambda r: (r.mcts_vs_heuristic or 0.0), reverse=True)

    print(
        "run_dir".ljust(32),
        "games",
        "epochs",
        "des_loss",
        "mcts_loss",
        "des_vs_heur",
        "mcts_vs_heur",
        "des_vs_mcts",
    )
    print("-" * 100)
    for r in all_runs:
        name = r.path.name
        print(
            name.ljust(32),
            f"{r.num_games:5d}",
            f"{r.epochs:6d}",
            format_loss(r.descent_loss),
            format_loss(r.mcts_loss),
            format_pct(r.descent_vs_heuristic),
            format_pct(r.mcts_vs_heuristic),
            format_pct(r.descent_vs_mcts),
        )


if __name__ == "__main__":
    main()

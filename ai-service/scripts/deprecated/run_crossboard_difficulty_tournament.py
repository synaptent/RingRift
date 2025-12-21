#!/usr/bin/env python
"""Run cross-board difficulty-tier tournaments and summarize consistency.

This script is a thin wrapper around `scripts/run_distributed_tournament.py`
that:

1) Runs a full (or small demo) D1–D10 tournament per board, and
2) Aggregates the per-board Elo rankings into a cross-board consistency report
   (pairwise Spearman rank correlation + inversion counts).

It is intentionally orchestration-only so you can run different boards on
different machines and then aggregate via `--board-report board=path`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.training.crossboard_strength import (
    normalise_tier_name,
    summarize_crossboard_tier_strength,
)


def _parse_tiers(tiers_spec: str) -> list[str]:
    spec = tiers_spec.strip()
    if not spec:
        raise ValueError("tiers must be non-empty")

    if "," not in spec and "-" in spec:
        start, end = (s.strip() for s in spec.split("-", 1))
        start_tier = normalise_tier_name(start)
        end_tier = normalise_tier_name(end)
        start_n = int(start_tier[1:])
        end_n = int(end_tier[1:])
        if end_n < start_n:
            raise ValueError(f"Invalid tier range: {tiers_spec!r}")
        return [f"D{i}" for i in range(start_n, end_n + 1)]

    return [normalise_tier_name(part) for part in spec.split(",") if part.strip()]


def _parse_boards(boards_spec: str) -> list[str]:
    boards = [b.strip().lower() for b in boards_spec.split(",") if b.strip()]
    if not boards:
        raise ValueError("boards must be non-empty")
    # Normalise common aliases.
    normalized = []
    for b in boards:
        if b in {"hexagonal", "hex"}:
            normalized.append("hex")
        elif b in {"sq8", "square8"}:
            normalized.append("square8")
        elif b in {"sq19", "square19"}:
            normalized.append("square19")
        else:
            normalized.append(b)
    return normalized


def _parse_board_reports(items: list[str]) -> dict[str, str]:
    reports: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --board-report {item!r}; expected format board=path"
            )
        board, path = item.split("=", 1)
        board = board.strip().lower()
        path = path.strip()
        if not board or not path:
            raise ValueError(
                f"Invalid --board-report {item!r}; expected format board=path"
            )
        reports[board] = path
    return reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run difficulty-tier tournaments across multiple boards and emit a "
            "single cross-board summary JSON."
        )
    )
    parser.add_argument(
        "--boards",
        type=str,
        default="square8,square19,hex",
        help="Comma-separated boards to run (default: square8,square19,hex).",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="D1-D10",
        help="Tier list or range (e.g. D1-D10 or D1,D2,D4,D6) (default: D1-D10).",
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=100,
        help="Games per tier matchup (default: 100).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers per-board tournament (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base seed forwarded to per-board tournaments (default: 1).",
    )
    parser.add_argument(
        "--think-time-scale",
        type=float,
        default=1.0,
        help="Multiply ladder think_time_ms by this factor (default: 1.0).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Max moves per game before declaring draw (default: 300).",
    )
    parser.add_argument(
        "--wilson-confidence",
        type=float,
        default=0.95,
        help="Wilson CI confidence for decisive matchups (default: 0.95).",
    )
    parser.add_argument(
        "--nn-model-id",
        type=str,
        default=None,
        help=(
            "Optional override for CNN model id used by MCTS/Descent tiers "
            "(defaults to LadderTierConfig.model_id)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tournaments",
        help="Directory for per-board reports and combined summary.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional explicit output path for the combined JSON report.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Demo mode: lower the compute footprint (defaults to "
            "--games-per-matchup=4 and --think-time-scale=0.05 unless overridden)."
        ),
    )
    parser.add_argument(
        "--board-report",
        action="append",
        default=[],
        help=(
            "Aggregate existing per-board tournament reports instead of running. "
            "Repeatable, format: board=path."
        ),
    )
    return parser.parse_args(argv)


def _load_report(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_elos(report: dict[str, Any]) -> dict[str, float]:
    elos: dict[str, float] = {}
    for row in report.get("rankings", []):
        tier = row.get("tier")
        if not tier:
            continue
        elos[normalise_tier_name(str(tier))] = float(row.get("elo", 0.0))
    return elos


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    boards = _parse_boards(args.boards)
    tiers = _parse_tiers(args.tiers)

    games_per_matchup = args.games_per_matchup
    think_time_scale = args.think_time_scale
    if args.demo:
        if args.games_per_matchup == 100:
            games_per_matchup = 4
        if args.think_time_scale == 1.0:
            think_time_scale = 0.05

    os.makedirs(args.output_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    board_report_paths: dict[str, str] = {}
    board_reports: dict[str, dict[str, Any]] = {}

    provided = _parse_board_reports(args.board_report)
    if provided:
        board_report_paths.update(provided)
        for board, path in provided.items():
            board_reports[board] = _load_report(path)
    else:
        for board in boards:
            report_path = os.path.join(
                args.output_dir,
                f"difficulty_tournament_{board}_{ts}.json",
            )
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "run_distributed_tournament.py"),
                "--board",
                board,
                "--tiers",
                ",".join(tiers),
                "--games-per-matchup",
                str(games_per_matchup),
                "--workers",
                str(args.workers),
                "--output-dir",
                args.output_dir,
                "--seed",
                str(args.seed),
                "--think-time-scale",
                str(think_time_scale),
                "--max-moves",
                str(args.max_moves),
                "--wilson-confidence",
                str(args.wilson_confidence),
                "--output-report",
                report_path,
            ]
            if args.nn_model_id:
                cmd.extend(["--nn-model-id", args.nn_model_id])

            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            board_report_paths[board] = report_path
            board_reports[board] = _load_report(report_path)

    board_elos = {board: _extract_elos(report) for board, report in board_reports.items()}
    crossboard = summarize_crossboard_tier_strength(board_elos)

    combined = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "boards": boards,
            "tiers": tiers,
            "games_per_matchup": games_per_matchup,
            "workers": args.workers,
            "seed": args.seed,
            "think_time_scale": think_time_scale,
            "max_moves": args.max_moves,
            "wilson_confidence": args.wilson_confidence,
            "nn_model_id": args.nn_model_id,
            "demo": bool(args.demo),
        },
        "board_reports": {
            board: {
                "path": path,
                "tournament_id": board_reports[board].get("tournament_id"),
                "summary": board_reports[board].get("summary"),
            }
            for board, path in board_report_paths.items()
        },
        "crossboard_summary": crossboard,
    }

    out_path = args.output_json or os.path.join(
        args.output_dir, f"crossboard_difficulty_tournament_{ts}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, sort_keys=True)

    print(f"Wrote cross-board tournament summary to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


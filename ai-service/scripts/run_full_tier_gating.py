#!/usr/bin/env python
"""Combined tier gating + perf benchmark wrapper for ladder tiers.

This script orchestrates the core checks described in the
AI_TIER_TRAINING_AND_PROMOTION_PIPELINE for a single difficulty tier:

1) Load ``training_report.json`` from a tier training run directory and
   validate that ``tier`` and ``candidate_id`` match the CLI arguments.
2) Run the canonical difficulty-tier gate via ``run_tier_gate.py`` in
   difficulty mode for the given candidate.
3) Run the small tier perf benchmark for tiers with configured budgets
   (D3/D4/D5/D6/D7/D8) unless ``--no-perf`` is supplied.
4) Optionally perform cross-tier sanity probes (currently stubbed).
5) Aggregate everything into a ``gate_report.json`` plus an updated
   ``status.json`` under the same ``--run-dir``.

The script is safe to run in ``--demo`` mode, which uses very small
game counts for evaluation and perf to keep CI/local smoke tests fast.
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

from app.config.perf_budgets import (
    get_tier_perf_budget,
)
from app.training.tier_perf_benchmark import (
    TierPerfResult,
    run_tier_perf_benchmark,
)

# Filenames within a run directory
TRAINING_REPORT_NAME = "training_report.json"
TIER_EVAL_FILENAME = "tier_eval_result.json"
PROMOTION_PLAN_FILENAME = "promotion_plan.json"
TIER_PERF_FILENAME = "tier_perf_report.json"
GATE_REPORT_NAME = "gate_report.json"
STATUS_NAME = "status.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for full tier gating."""
    parser = argparse.ArgumentParser(
        description=(
            "Run difficulty-tier gating and (optionally) perf benchmark for a "
            "ladder tier candidate based on an existing training run "
            "directory."
        ),
    )
    parser.add_argument(
        "--tier",
        required=True,
        help="Difficulty tier name (e.g. D2-D10).",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Candidate id to gate (must match training_report.json).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=("Path to a run directory produced by " "run_tier_training_pipeline.py containing training_report.json."),
    )
    parser.add_argument(
        "--board",
        default="square8",
        help=("Board identifier for future multi-board support " "(default: square8)."),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2).",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip perf benchmark even when a TierPerfBudget exists.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=("Use lightweight configs (few games, small perf sample) suitable " "for CI and local smoke tests."),
    )
    return parser.parse_args(argv)


def _load_training_report(run_dir: str) -> dict[str, Any]:
    """Load training_report.json from *run_dir* or exit with an error."""
    path = os.path.join(run_dir, TRAINING_REPORT_NAME)
    if not os.path.exists(path):
        raise SystemExit(
            f"training_report.json not found in run dir {run_dir!r}; "
            "ensure run_tier_training_pipeline.py has completed."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_tier_gate_cli(
    tier: str,
    candidate_id: str,
    run_dir: str,
    seed: int | None,
    num_games_override: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke run_tier_gate.py in difficulty-tier mode.

    Returns the TierEvaluationResult payload and the promotion plan
    payload as Python dicts. Both JSON files are written into *run_dir*.
    """
    eval_path = os.path.join(run_dir, TIER_EVAL_FILENAME)
    plan_path = os.path.join(run_dir, PROMOTION_PLAN_FILENAME)

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_tier_gate.py"),
        "--tier",
        tier,
        "--candidate-model-id",
        candidate_id,
        "--output-json",
        eval_path,
        "--promotion-plan-out",
        plan_path,
    ]
    # In non-demo runs, the candidate id should refer to a concrete artefact
    # produced by the tier training pipeline. Enable candidate loading so the
    # gate does not accidentally evaluate the production ladder while
    # emitting a promotion plan for a label-only candidate.
    if num_games_override is None:
        cmd.append("--use-candidate-artifact")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if num_games_override is not None:
        cmd.extend(["--num-games", str(num_games_override)])

    subprocess.run(cmd, check=True)

    with open(eval_path, encoding="utf-8") as f:
        eval_payload = json.load(f)
    with open(plan_path, encoding="utf-8") as f:
        plan_payload = json.load(f)
    return eval_payload, plan_payload


def _eval_perf_budget(result: TierPerfResult) -> dict[str, Any]:
    """Evaluate whether a perf benchmark result is within its tier budget."""
    within_avg = result.average_ms <= result.budget.max_avg_move_ms
    within_p95 = result.p95_ms <= result.budget.max_p95_move_ms
    overall = within_avg and within_p95
    return {
        "within_avg": within_avg,
        "within_p95": within_p95,
        "overall_pass": overall,
    }


def _run_perf_if_available(
    tier: str,
    run_dir: str,
    demo: bool,
) -> tuple[TierPerfResult | None, dict[str, Any] | None, str | None]:
    """Run perf benchmark when a TierPerfBudget exists for *tier*.

    Returns (TierPerfResult or None, evaluation dict or None,
    relative_result_path or None).
    """
    try:
        # Probe for a configured budget; we ignore the returned object
        # here and rely on run_tier_perf_benchmark for details.
        get_tier_perf_budget(tier)
    except KeyError:
        return None, None, None

    num_games = 1 if demo else 4
    moves_per_game = 4 if demo else 16
    seed = 1

    result = run_tier_perf_benchmark(
        tier_name=tier,
        num_games=num_games,
        moves_per_game=moves_per_game,
        seed=seed,
    )
    eval_dict = _eval_perf_budget(result)

    perf_path = os.path.join(run_dir, TIER_PERF_FILENAME)
    payload: dict[str, Any] = {
        "tier_name": result.tier_name,
        "difficulty": result.budget.difficulty,
        "board_type": result.budget.board_type.value,
        "num_players": result.budget.num_players,
        "metrics": {
            "average_ms": result.average_ms,
            "p95_ms": result.p95_ms,
        },
        "budget": {
            "max_avg_move_ms": result.budget.max_avg_move_ms,
            "max_p95_move_ms": result.budget.max_p95_move_ms,
        },
        "evaluation": eval_dict,
    }
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return result, eval_dict, TIER_PERF_FILENAME


def _update_status_json(
    run_dir: str,
    tier: str,
    board: str,
    num_players: int,
    candidate_id: str,
) -> None:
    """Create or update status.json for the run directory."""
    status_path = os.path.join(run_dir, STATUS_NAME)
    status: dict[str, Any] = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
        except Exception:  # pragma: no cover - defensive
            status = {}

    status["tier"] = tier
    status["board"] = board
    status["num_players"] = num_players
    status["candidate_id"] = candidate_id

    training = status.get("training") or {}
    training.setdefault("status", "completed")
    training.setdefault("report_path", TRAINING_REPORT_NAME)
    status["training"] = training

    # Automated gate block: mark as completed and record artefact paths.
    auto_gate = status.get("automated_gate") or {}
    auto_gate["status"] = "completed"
    # Always record canonical filenames for tier eval and promotion plan once
    # gating has run, even if earlier steps initialised these fields to None.
    # Tests and downstream tooling rely on these being concrete strings after a
    # successful automated gate.
    auto_gate["eval_json"] = TIER_EVAL_FILENAME
    auto_gate["promotion_plan"] = PROMOTION_PLAN_FILENAME
    status["automated_gate"] = auto_gate

    # Perf block: mark as completed if a perf report exists in the run dir.
    perf = status.get("perf") or {}
    perf_path = os.path.join(run_dir, TIER_PERF_FILENAME)
    if os.path.exists(perf_path):
        perf["status"] = "completed"
        # As with automated_gate, overwrite any placeholder/None value with the
        # actual perf report filename once it exists.
        perf["perf_json"] = TIER_PERF_FILENAME
    else:
        perf.setdefault("status", "not_started")
        perf.setdefault("perf_json", None)
    status["perf"] = perf

    # Human calibration block remains a stub for now; the pipeline doc
    # treats calibration as a follow-up step once automated gates pass.
    human = status.get("human_calibration") or {
        "required": True,
        "status": "pending",
        "min_games": 50,
    }
    status["human_calibration"] = human

    # Backwards-compatible alias used by existing tests and tooling.
    gating = status.get("gating") or {}
    gating["status"] = auto_gate["status"]
    gating["report_path"] = GATE_REPORT_NAME
    status["gating"] = gating

    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the full tier gating pipeline."""
    args = parse_args(argv)
    tier = args.tier.upper()

    if tier not in {"D2", "D4", "D6", "D8", "D9", "D10"}:
        raise SystemExit(f"Unsupported tier {args.tier!r}; expected one of " "D2, D4, D6, D8, D9, D10.")

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    training_report = _load_training_report(run_dir)
    report_tier = str(training_report.get("tier", "")).upper()
    report_candidate = training_report.get("candidate_id")

    if report_tier != tier:
        raise SystemExit(
            "Tier mismatch between CLI and training_report.json: " f"CLI tier={tier!r}, report tier={report_tier!r}"
        )
    if report_candidate != args.candidate_id:
        raise SystemExit(
            "candidate_id mismatch between CLI and training_report.json: "
            f"CLI id={args.candidate_id!r}, report id={report_candidate!r}"
        )

    # Prefer the board/num_players recorded in the training report.
    board_from_report = training_report.get("board") or args.board
    num_players_from_report = int(training_report.get("num_players") or args.num_players)

    if board_from_report != args.board:
        print(
            "Warning: CLI board does not match training_report.json "
            f"(cli={args.board!r}, report={board_from_report!r}); "
            "using the report value in gate_report.json."
        )
    if num_players_from_report != args.num_players:
        print(
            "Warning: CLI num_players does not match training_report.json "
            f"(cli={args.num_players}, report={num_players_from_report}); "
            "using the report value in gate_report.json."
        )

    # Difficulty-tier evaluation and promotion plan.
    seed = 1
    num_games_override = 4 if args.demo else None
    tier_eval, promotion_plan = _run_tier_gate_cli(
        tier=tier,
        candidate_id=args.candidate_id,
        run_dir=run_dir,
        seed=seed,
        num_games_override=num_games_override,
    )
    gate_pass = bool(tier_eval.get("overall_pass"))

    # Perf benchmark (when budgets exist and not explicitly disabled).
    perf_result: TierPerfResult | None = None
    perf_eval: dict[str, Any] | None = None
    perf_result_path: str | None = None
    perf_run = False

    if not args.no_perf:
        perf_result, perf_eval, perf_result_path = _run_perf_if_available(
            tier=tier,
            run_dir=run_dir,
            demo=args.demo,
        )
        perf_run = perf_result is not None

    perf_pass = True
    if perf_run and perf_eval is not None:
        perf_pass = bool(perf_eval.get("overall_pass"))

    # Cross-tier sanity checks are currently stubbed; the structure is
    # left in place for future tiny tournaments.
    cross_tier_sanity: dict[str, Any] = {
        "run": False,
    }

    # Final decision: require both the difficulty gate and perf (when run)
    # to pass, and the promotion plan to recommend promotion.
    plan_decision = promotion_plan.get("decision", "reject")
    if gate_pass and perf_pass and plan_decision == "promote":
        final_decision = "promote"
    else:
        final_decision = "reject"

    created_at = datetime.now(timezone.utc).isoformat()

    evaluation_block: dict[str, Any] = {
        "result_path": TIER_EVAL_FILENAME,
        "promotion_plan_path": PROMOTION_PLAN_FILENAME,
        "overall_pass": gate_pass,
    }

    perf_block: dict[str, Any] = {
        "run": perf_run,
        "result_path": perf_result_path,
        "overall_pass": (perf_eval.get("overall_pass") if perf_run and perf_eval else None),
    }

    gate_report: dict[str, Any] = {
        "tier": tier,
        "board": board_from_report,
        "num_players": num_players_from_report,
        "candidate_id": args.candidate_id,
        "evaluation": evaluation_block,
        "perf": perf_block,
        "cross_tier_sanity": cross_tier_sanity,
        "final_decision": final_decision,
        "created_at": created_at,
    }

    gate_path = os.path.join(run_dir, GATE_REPORT_NAME)
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2, sort_keys=True)
    print(f"Wrote combined gate report to {gate_path}")

    _update_status_json(
        run_dir=run_dir,
        tier=tier,
        board=board_from_report,
        num_players=num_players_from_report,
        candidate_id=args.candidate_id,
    )

    # Exit code: 0 only when both gate and perf (if run) pass.
    exit_ok = gate_pass and (not perf_run or perf_pass)
    return 0 if exit_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

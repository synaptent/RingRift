#!/usr/bin/env python
"""Tiered training pipeline orchestrator for Square-8 2-player ladder tiers.

This script trains (or in demo mode, stubs) a candidate for a given
difficulty tier (D2/D4/D6/D8) and writes a structured
``training_report.json`` plus a small ``status.json`` into the specified
run directory.

The implementation is intentionally light-weight:

- In ``--demo`` mode no heavy self-play or neural training is performed;
  instead we generate plausible candidate ids and record configuration
  stubs and minimal metrics suitable for CI and local smoke tests.
- Non-demo code paths are structured so they can be wired to real
  training loops (heuristic CMA-ES for D2, search persona tuning for
  D4, and neural training for D6/D8) without changing the JSON schema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType  # noqa: E402
from app.training.env import TrainingEnvConfig  # noqa: E402
from app.training.config import TrainConfig  # noqa: E402
from app.training.config import (  # noqa: E402
    get_training_config_for_board,
)
from app.training.train import train_model  # noqa: E402
from app.training.seed_utils import seed_all  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the tier training pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Train (or stub-train) a candidate for a Square-8 2-player "
            "difficulty tier and emit training_report.json in --run-dir."
        ),
    )
    parser.add_argument(
        "--tier",
        required=True,
        help="Difficulty tier (D2, D4, D6, D8).",
    )
    parser.add_argument(
        "--board",
        default="square8",
        help="Board identifier (currently only 'square8' is supported).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (currently only 2 is supported).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Output directory for training_report.json and status.json. "
            "Will be created if it does not exist."
        ),
    )
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=None,
        help=(
            "Optional explicit candidate id. When omitted an id is "
            "generated based on tier/board/num_players."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use extremely small/safe configs and avoid heavy training. "
            "Intended for CI and local smoke tests."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible demo runs.",
    )
    return parser.parse_args(argv)


def _board_prefix(board: str) -> str:
    """Return a short, filesystem-friendly board prefix for candidate ids."""
    b = board.lower()
    if b in {"square8", "sq8"}:
        return "sq8"
    return b.replace(" ", "_")


def _generate_candidate_id(
    tier: str,
    board: str,
    num_players: int,
    demo: bool,
    now: datetime,
) -> str:
    """Generate a simple candidate id string.

    For example (demo mode)::

        sq8_2p_d4_demo_20251205_140000
    """
    tier_name = tier.upper()
    if not tier_name.startswith("D") or not tier_name[1:].isdigit():
        raise ValueError(
            f"Unsupported tier name {tier!r}; expected D2/D4/D6/D8."
        )
    suffix = "demo" if demo else "cand"
    ts = now.strftime("%Y%m%d_%H%M%S")
    prefix = _board_prefix(board)
    return f"{prefix}_{num_players}p_{tier_name.lower()}_{suffix}_{ts}"


def _build_env_summary(
    board: str,
    num_players: int,
    seed: int | None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of TrainingEnvConfig."""
    # For now we only support Square-8 2-player; keep env canonical.
    board_enum = BoardType.SQUARE8
    env_cfg = TrainingEnvConfig(
        board_type=board_enum,
        num_players=num_players,
        max_moves=None,
        reward_mode="terminal",
        seed=seed,
    )
    data = asdict(env_cfg)
    # Convert enum to its public string representation.
    data["board_type"] = env_cfg.board_type.value
    return data


def _run_d2_training(
    args: argparse.Namespace,
    candidate_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tier-specific training stub for D2 (heuristic baseline)."""
    seed = args.seed or 1
    training_params: Dict[str, Any] = {
        "mode": "heuristic_stub_demo" if args.demo else "heuristic_stub",
        "trainer": "heuristic_cmaes",
        "tier_spec_id": "sq8_heuristic_baseline_v1",
        "base_profile_id": "heuristic_v1_balanced",
        "ladder_heuristic_profile_id": "heuristic_v1_2p",
        "candidate_profile_id": candidate_id,
        "generations": 1 if args.demo else 5,
        "population_size": 4 if args.demo else 8,
        "games_per_candidate": 4 if args.demo else 24,
        "seed": seed,
    }

    metrics: Dict[str, Any] = {
        "training_steps": 0,
        "loss": None,
        "extra": {
            "note": (
                "demo mode; no real heuristic optimisation run"
                if args.demo
                else (
                    "stub configuration; wire to "
                    "run_cmaes_heuristic_optimization for full runs"
                )
            ),
        },
    }

    # Non-demo path is intentionally left as a structural stub for now to
    # avoid surprising long-running jobs. When needed, this branch can call
    # run_cmaes_heuristic_optimization(...) and fold its report into metrics.
    return training_params, metrics


def _run_d4_training(
    args: argparse.Namespace,
    candidate_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tier-specific training stub for D4 (minimax search persona)."""
    seed = args.seed or 1
    # In the current implementation we treat D4 primarily as a search-
    # configuration tuning problem rather than heavy self-play.
    training_params: Dict[str, Any] = {
        "mode": "search_persona_demo" if args.demo else "search_persona",
        "persona_id": candidate_id,
        "heuristic_profile_id": "heuristic_v1_2p",
        "search_depth": 4,
        "use_iterative_deepening": True,
        "enable_pruning": True,
        "seed": seed,
    }
    metrics: Dict[str, Any] = {
        "training_steps": 0,
        "loss": None,
        "extra": {
            "note": (
                "demo mode; no real tournaments run; persona config only"
            ),
        },
    }
    return training_params, metrics


def _run_neural_tier_training(
    args: argparse.Namespace,
    candidate_id: str,
    tier_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Shared helper for D6/D8 neural tiers.

    In demo mode this simply records a minimal TrainConfig snapshot without
    invoking heavy training loops. For full runs (non-demo) it calls
    train_model(...) with a small-but-real configuration using the
    canonical Square-8 training preset.
    """
    seed = args.seed or 42
    board_enum = BoardType.SQUARE8

    # Start from the board-specific preset so we stay consistent with other
    # tooling.
    base_config = get_training_config_for_board(board_enum, TrainConfig())
    base_config.seed = seed
    base_config.model_id = candidate_id

    # Map tier to a notional logical difficulty for logging purposes.
    logical_difficulty: int | None
    if tier_name[1:].isdigit():
        logical_difficulty = int(tier_name[1:])
    else:
        logical_difficulty = None

    training_params: Dict[str, Any] = {
        "mode": "neural_demo" if args.demo else "neural_full",
        "train_config": {
            "board_type": base_config.board_type.value,
            "model_id": base_config.model_id,
            "batch_size": base_config.batch_size,
            "epochs_per_iter": base_config.epochs_per_iter,
            "learning_rate": base_config.learning_rate,
        },
        "logical_difficulty": logical_difficulty,
    }

    metrics: Dict[str, Any] = {
        "training_steps": 0,
        "loss": None,
        "extra": {},
    }

    if args.demo:
        # In demo mode we explicitly avoid invoking torch-based training to
        # keep CI runs fast and deterministic.
        training_params["note"] = (
            "demo mode; neural training loop not executed; model_id is "
            "reserved only"
        )
        return training_params, metrics

    # Full (non-demo) run: call into train_model with a conservative config.
    seed_all(seed)
    data_path = os.path.join(
        base_config.data_dir,
        "square8_selfplay_demo.npz",
    )
    save_path = os.path.join(
        base_config.model_dir,
        f"{candidate_id}.pth",
    )
    checkpoint_dir = os.path.join(
        base_config.model_dir,
        "checkpoints",
        candidate_id,
    )

    train_model(
        config=base_config,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=5,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=base_config.epochs_per_iter,
        warmup_epochs=0,
        lr_scheduler="none",
    )

    # We do not currently plumb through detailed loss curves from train_model;
    # callers can inspect training logs/checkpoints directly if needed.
    metrics["training_steps"] = base_config.epochs_per_iter
    metrics["extra"] = {
        "checkpoint_path": save_path,
    }
    return training_params, metrics


def _update_status_json(
    run_dir: str,
    tier: str,
    board: str,
    num_players: int,
    candidate_id: str,
) -> None:
    """Create or update a lightweight status.json for the run directory."""
    status_path = os.path.join(run_dir, "status.json")
    status: Dict[str, Any] = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        except Exception:  # pragma: no cover - defensive
            status = {}

    status["tier"] = tier
    status["board"] = board
    status["num_players"] = num_players
    status["candidate_id"] = candidate_id
    # Training block mirrors the pipeline doc: status + report reference.
    training = status.get("training") or {}
    training["status"] = "completed"
    training["report_path"] = "training_report.json"
    status["training"] = training

    # Automated gate / perf / human calibration blocks are initialised
    # here so later orchestration (run_full_tier_gating.py) can fill
    # them in without having to create structure from scratch.
    auto_gate = status.get("automated_gate") or {}
    auto_gate.setdefault("status", "not_started")
    auto_gate.setdefault("eval_json", None)
    auto_gate.setdefault("promotion_plan", None)
    status["automated_gate"] = auto_gate

    perf = status.get("perf") or {}
    perf.setdefault("status", "not_started")
    perf.setdefault("perf_json", None)
    status["perf"] = perf

    human = status.get("human_calibration") or {
        "required": True,
        "status": "pending",
        "min_games": 50,
    }
    status["human_calibration"] = human

    # Backwards-compatible alias used by some tests and tooling.
    gating = status.get("gating") or {}
    gating.setdefault("status", auto_gate["status"])
    gating.setdefault("report_path", None)
    status["gating"] = gating

    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tier_name = args.tier.upper()
    if tier_name not in {"D2", "D4", "D6", "D8"}:
        raise SystemExit(
            f"Unsupported tier {args.tier!r}; expected one of D2, D4, D6, D8."
        )

    board_norm = args.board.lower()
    if board_norm not in {"square8", "sq8"}:
        print(
            f"Warning: board {args.board!r} is not the canonical square8; "
            "proceeding but training configuration is only validated for "
            "Square-8 2-player."
        )
    if args.num_players != 2:
        print(
            f"Warning: num_players={args.num_players} is not supported; "
            "this pipeline is designed for 2-player only."
        )

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    now = datetime.now(timezone.utc)
    candidate_id = args.candidate_id or _generate_candidate_id(
        tier=tier_name,
        board=args.board,
        num_players=args.num_players,
        demo=args.demo,
        now=now,
    )

    env_summary = _build_env_summary(
        board=args.board,
        num_players=args.num_players,
        seed=args.seed if args.demo else None,
    )

    if tier_name == "D2":
        training_params, metrics = _run_d2_training(args, candidate_id)
    elif tier_name == "D4":
        training_params, metrics = _run_d4_training(args, candidate_id)
    elif tier_name in {"D6", "D8"}:
        training_params, metrics = _run_neural_tier_training(
            args, candidate_id, tier_name
        )
    else:  # pragma: no cover - guarded above
        raise SystemExit(f"Unsupported tier {tier_name!r}.")

    report: Dict[str, Any] = {
        "tier": tier_name,
        "board": args.board,
        "num_players": args.num_players,
        "candidate_id": candidate_id,
        "config": {
            "env": env_summary,
            "training_params": training_params,
        },
        "metrics": metrics,
        "created_at": now.isoformat(),
    }

    report_path = os.path.join(run_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Wrote training report to {report_path}")

    _update_status_json(
        run_dir=run_dir,
        tier=tier_name,
        board=args.board,
        num_players=args.num_players,
        candidate_id=candidate_id,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

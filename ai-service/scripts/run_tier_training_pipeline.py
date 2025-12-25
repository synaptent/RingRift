#!/usr/bin/env python3
"""
Tier-based training pipeline for AI difficulty ladder.

This script trains AI models for specific difficulty tiers (D2-D10)
using different training modes based on the tier configuration.

Usage:
    python scripts/run_tier_training_pipeline.py --tier D6 --board square8 --num-players 2
    python scripts/run_tier_training_pipeline.py --tier D8 --config config/tier_training_pipeline.square8_2p.json

For gating after training, see: python scripts/run_full_tier_gating.py --help

Training modes by tier (configurable):
    D2-D3: heuristic_cmaes - CMA-ES optimization of heuristic weights
    D4-D5: search_persona - search persona snapshot (minimax)
    D7: search_persona - search persona snapshot (heuristic-only MCTS)
    D6/D8-D10: neural - Neural network training with increasing strength
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.ladder_config import get_ladder_tier_config
from app.models import AIType, BoardType

BOARD_TYPE_BY_ARG = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
    "hex8": BoardType.HEXAGONAL,
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _heuristic_profile_key(board: str, num_players: int) -> str:
    board_abbrev = {
        "square8": "sq8",
        "square19": "sq19",
        "hexagonal": "hex",
        "hex8": "hex",
        "hex": "hex",
    }.get(board, board[:3])
    return f"heuristic_v1_{board_abbrev}_{num_players}p"


def _build_candidate_id(args: argparse.Namespace) -> str:
    token = uuid.uuid4().hex[:8]
    return f"{args.tier.lower()}_{args.board}_{args.num_players}p_{_utc_stamp()}_{token}"


def _build_run_dir(args: argparse.Namespace) -> Path:
    run_tag = f"{args.tier}_{args.board}_{args.num_players}p_{_utc_stamp()}"
    return args.output_dir / run_tag


def _get_ladder_config(args: argparse.Namespace):
    try:
        board_type = BOARD_TYPE_BY_ARG[args.board]
        difficulty = int(args.tier[1:])
        return get_ladder_tier_config(difficulty, board_type, args.num_players)
    except Exception:
        return None


def _minimax_depth_for_difficulty(difficulty: int) -> int:
    if difficulty >= 9:
        return 5
    if difficulty >= 7:
        return 4
    if difficulty >= 4:
        return 3
    return 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AI models for a specific difficulty tier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        help="Difficulty tier to train (D2=easiest, D10=hardest).",
    )
    parser.add_argument(
        "--board",
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type (default: square8).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to tier training config JSON (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "tier_training",
        help="Output directory for training artifacts.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to training data NPZ file (for neural tiers).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for neural tiers (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for neural training (default: 256).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with minimal training for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    return parser.parse_args(argv)


def load_full_config(config_path: Path | None) -> dict[str, Any]:
    """Load the full tier training config JSON if provided."""
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_tier_config(
    config_path: Path | None,
    tier: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load tier-specific configuration and return (tier_config, full_config)."""
    full_config = load_full_config(config_path)
    if full_config:
        tier_block = full_config.get("tiers", {}).get(tier, {})
        if tier_block:
            return tier_block, full_config

    defaults = {
        "D2": {"training": {"mode": "heuristic_cmaes"}},
        "D3": {"training": {"mode": "heuristic_cmaes"}},
        "D4": {"training": {"mode": "search_persona"}},
        "D5": {"training": {"mode": "search_persona"}},
        "D6": {"training": {"mode": "neural"}},
        "D7": {"training": {"mode": "search_persona"}},
        "D8": {"training": {"mode": "neural"}},
        "D9": {"training": {"mode": "neural"}},
        "D10": {"training": {"mode": "neural"}},
    }
    return defaults.get(tier, {"training": {"mode": "neural"}}), full_config


def _resolve_cmaes_workers(training_cfg: dict[str, Any]) -> str | None:
    workers = training_cfg.get("cmaes_workers")
    if workers:
        return str(workers)
    env_workers = os.getenv("RINGRIFT_CMAES_WORKERS")
    if env_workers:
        return env_workers
    return None


def _select_cmaes_mode(
    training_cfg: dict[str, Any],
    full_config: dict[str, Any],
) -> str:
    requested = str(training_cfg.get("cmaes_mode") or "auto")
    if requested.lower() != "auto":
        return requested

    prefs = full_config.get("cmaes_preferences", {})
    pref_order = prefs.get("preference_order", [])
    workers = _resolve_cmaes_workers(training_cfg)

    for mode in pref_order:
        if mode in ("gpu_distributed_iterative", "distributed_iterative") and not workers:
            continue
        return mode

    return "iterative"


def _build_cmaes_command(
    args: argparse.Namespace,
    run_dir: Path,
    training_cfg: dict[str, Any],
    full_config: dict[str, Any],
    mode: str,
) -> list[str] | None:
    prefs = full_config.get("cmaes_preferences", {})
    scripts = prefs.get("scripts", {})
    params = prefs.get("default_params", {}).get(mode, {})
    workers = _resolve_cmaes_workers(training_cfg)

    script_entry = scripts.get(mode)
    if not script_entry:
        return None

    script_parts = shlex.split(str(script_entry))
    script_name = script_parts[0]
    script_path = PROJECT_ROOT / "scripts" / script_name
    cmd: list[str] = [sys.executable, str(script_path)] + script_parts[1:]

    if mode in ("iterative", "distributed_iterative"):
        cmd.extend([
            "--board",
            args.board,
            "--num-players",
            str(args.num_players),
            "--output-dir",
            str(run_dir),
            "--seed",
            str(args.seed),
        ])
        if params.get("generations_per_iter") is not None:
            cmd.extend(["--generations-per-iter", str(params["generations_per_iter"])])
        if params.get("max_iterations") is not None:
            cmd.extend(["--max-iterations", str(params["max_iterations"])])
        if params.get("improvement_threshold") is not None:
            cmd.extend(["--improvement-threshold", str(params["improvement_threshold"])])
        if params.get("plateau_generations") is not None:
            cmd.extend(["--plateau-generations", str(params["plateau_generations"])])
        if params.get("population_size") is not None:
            cmd.extend(["--population-size", str(params["population_size"])])
        if params.get("games_per_eval") is not None:
            cmd.extend(["--games-per-eval", str(params["games_per_eval"])])

        profiles_path = training_cfg.get("profiles_path")
        if profiles_path:
            cmd.extend(["--profiles-path", str(profiles_path)])

        if mode == "distributed_iterative":
            if "--distributed" not in cmd:
                cmd.append("--distributed")
            if workers:
                cmd.extend(["--workers", workers])

    elif mode == "gpu_distributed_iterative":
        if not workers:
            return None
        cmd.extend([
            "--mode",
            "coordinator",
            "--board",
            args.board,
            "--num-players",
            str(args.num_players),
            "--output-dir",
            str(run_dir),
            "--workers",
            workers,
        ])
        generations = params.get("generations") or params.get("generations_per_iter")
        if generations is not None:
            cmd.extend(["--generations", str(generations)])
        if params.get("population_size") is not None:
            cmd.extend(["--population-size", str(params["population_size"])])
        if params.get("games_per_eval") is not None:
            cmd.extend(["--games-per-eval", str(params["games_per_eval"])])

    elif mode == "single_stage":
        output_path = run_dir / "optimized_weights.json"
        cmd.extend([
            "--board",
            args.board,
            "--num-players",
            str(args.num_players),
            "--output",
            str(output_path),
        ])
        if params.get("generations") is not None:
            cmd.extend(["--generations", str(params["generations"])])
        if params.get("population_size") is not None:
            cmd.extend(["--population-size", str(params["population_size"])])
        if params.get("games_per_eval") is not None:
            cmd.extend(["--games-per-eval", str(params["games_per_eval"])])

    return cmd


def run_neural_training(
    args: argparse.Namespace,
    tier_config: dict[str, Any],
    candidate_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Run neural network training for the tier."""
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{candidate_id}.pth"

    cmd = [
        sys.executable,
        "-m",
        "app.training.train",
        "--board-type",
        args.board,
        "--num-players",
        str(args.num_players),
        "--epochs",
        str(args.epochs if not args.demo else 2),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--save-path",
        str(model_path),
    ]

    if args.data_path:
        cmd.extend(["--data-path", str(args.data_path)])

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running neural training for tier {args.tier}...")
    print(f'Command: {" ".join(cmd)}')

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    success = result.returncode == 0

    if success and model_path.exists():
        shutil.copy2(model_path, run_dir / "model.pth")

    return {
        "mode": "neural",
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "success": success,
        "candidate_model_id": candidate_id,
        "model_path": str(model_path),
    }


def run_heuristic_cmaes(
    args: argparse.Namespace,
    tier_config: dict[str, Any],
    full_config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Run CMA-ES optimization for heuristic weights."""
    print(f"Running CMA-ES optimization for tier {args.tier}...")

    run_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = tier_config.get("training", {})
    mode = _select_cmaes_mode(training_cfg, full_config)

    cmd = _build_cmaes_command(args, run_dir, training_cfg, full_config, mode)
    if not cmd:
        return {
            "mode": "heuristic_cmaes",
            "success": False,
            "note": f"Unable to resolve CMA-ES command for mode {mode}.",
        }

    print(f"CMA-ES mode: {mode}")
    print(f'Command: {" ".join(cmd)}')

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    return {
        "mode": "heuristic_cmaes",
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "success": result.returncode == 0,
        "cmaes_mode": mode,
    }


def run_search_persona(
    args: argparse.Namespace,
    tier_config: dict[str, Any],
    candidate_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Generate a search persona snapshot for minimax/MCTS tiers."""
    run_dir.mkdir(parents=True, exist_ok=True)

    ladder_cfg = _get_ladder_config(args)
    difficulty = int(args.tier[1:])
    ai_type = ladder_cfg.ai_type if ladder_cfg else AIType.MCTS

    persona_config: dict[str, Any] = {
        "tier": args.tier,
        "candidate_id": candidate_id,
        "ai_type": ai_type.value if isinstance(ai_type, AIType) else str(ai_type),
        "difficulty": difficulty,
        "think_time_ms": ladder_cfg.think_time_ms if ladder_cfg else None,
        "randomness": ladder_cfg.randomness if ladder_cfg else None,
        "use_neural_net": ladder_cfg.use_neural_net if ladder_cfg else False,
    }

    training_cfg = tier_config.get("training", {})

    if ai_type == AIType.MINIMAX:
        persona_config.update(
            {
                "search_type": "minimax",
                "max_depth": _minimax_depth_for_difficulty(difficulty),
                "use_incremental_search": True,
            }
        )
    else:
        default_sims = 100 if difficulty <= 4 else 200 if difficulty <= 6 else 300
        persona_config.update(
            {
                "search_type": "mcts",
                "mcts_simulations": training_cfg.get("mcts_simulations", default_sims),
                "exploration_constant": training_cfg.get("exploration_constant", 1.4),
                "temperature": training_cfg.get("temperature", 0.5),
            }
        )

    with open(run_dir / "search_persona.json", "w", encoding="utf-8") as f:
        json.dump(persona_config, f, indent=2)

    return {
        "mode": "search_persona",
        "run_dir": str(run_dir),
        "success": True,
        "persona_config": persona_config,
    }


def main(argv: list[str] | None = None) -> int:
    """Main entry point for tier training pipeline."""
    args = parse_args(argv)

    print("=" * 60)
    print("Tier Training Pipeline")
    print("=" * 60)
    print(f"Tier: {args.tier}")
    print(f"Board: {args.board}")
    print(f"Players: {args.num_players}")
    print(f"Demo mode: {args.demo}")
    print("=" * 60)

    tier_config, full_config = load_tier_config(args.config, args.tier)
    training_mode = tier_config.get("training", {}).get("mode", "neural")

    ladder_cfg = _get_ladder_config(args)
    if training_mode == "heuristic_cmaes":
        candidate_id = (
            ladder_cfg.model_id if ladder_cfg and ladder_cfg.model_id else _heuristic_profile_key(args.board, args.num_players)
        )
    elif training_mode == "search_persona":
        candidate_id = ladder_cfg.model_id if ladder_cfg and ladder_cfg.model_id else _build_candidate_id(args)
    else:
        candidate_id = _build_candidate_id(args)

    run_dir = _build_run_dir(args)

    print(f"Training mode: {training_mode}")
    print(f"Candidate ID: {candidate_id}")

    if training_mode == "heuristic_cmaes":
        result = run_heuristic_cmaes(args, tier_config, full_config, run_dir)
    elif training_mode == "search_persona":
        result = run_search_persona(args, tier_config, candidate_id, run_dir)
    else:
        result = run_neural_training(args, tier_config, candidate_id, run_dir)

    training_report = {
        "tier": args.tier,
        "candidate_id": candidate_id,
        "candidate_model_id": candidate_id if training_mode == "neural" else None,
        "candidate_profile_id": candidate_id if training_mode == "heuristic_cmaes" else None,
        "board": args.board,
        "num_players": args.num_players,
        "training_mode": training_mode,
        "demo": args.demo,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }

    if "run_dir" in result:
        report_path = Path(result["run_dir"]) / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(training_report, f, indent=2)
        print(f"\nTraining report saved to: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"Training {'completed successfully' if result.get('success') else 'failed'}")
    print(f"Candidate ID: {candidate_id}")
    print(f"{'=' * 60}")

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

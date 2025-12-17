#!/usr/bin/env python3
"""Unified Tournament Runner for RingRift AI Evaluation.

This script consolidates all tournament functionality into a single entry point
with multiple modes, each preserving the unique features of the original scripts:

Modes:
  basic        - Simple AI vs AI tournament (from run_ai_tournament.py)
  models       - Neural network model evaluation (from run_model_elo_tournament.py)
  distributed  - Multi-tier difficulty ladder (from run_distributed_tournament.py)
  ssh          - SSH-distributed tournament (from run_ssh_distributed_tournament.py)
  eval         - Evaluation pool tournament (from run_eval_tournaments.py)
  diverse      - All board/player configs (from run_diverse_tournaments.py)
  weights      - Heuristic weight profiles (from run_axis_aligned_tournament.py)
  crossboard   - Cross-board analysis (from run_crossboard_difficulty_tournament.py)

Usage:
    # Basic tournament
    python run_tournament.py basic --p1 heuristic --p2 mcts --games 100

    # Model evaluation
    python run_tournament.py models --board square8 --players 2 --continuous

    # Distributed ladder
    python run_tournament.py distributed --tiers D1-D5 --workers 4

    # All configs
    python run_tournament.py diverse --orchestrator local

Configuration is loaded from config/unified_loop.yaml for consistent settings.
Results are persisted to the unified Elo database (data/unified_elo.db).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup path
AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.unified_config import get_config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for tournament execution."""
    level = logging.DEBUG if verbose else logging.INFO
    try:
        from app.core.logging_config import setup_logging as _setup_logging
        _setup_logging("run_tournament", level=level, log_dir="logs")
    except ImportError:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all modes."""
    parser = argparse.ArgumentParser(
        description="Unified Tournament Runner for RingRift AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", type=str, help="Config file path (default: config/unified_loop.yaml)")
    parser.add_argument("--output-dir", type=str, default="tournament_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Create subparsers for each mode
    subparsers = parser.add_subparsers(dest="mode", help="Tournament mode")

    # Basic mode (from run_ai_tournament.py)
    basic = subparsers.add_parser("basic", help="Basic AI vs AI tournament")
    basic.add_argument("--p1", type=str, default="heuristic", help="Player 1 AI type")
    basic.add_argument("--p1-diff", type=int, default=5, help="Player 1 difficulty")
    basic.add_argument("--p2", type=str, default="mcts", help="Player 2 AI type")
    basic.add_argument("--p2-diff", type=int, default=5, help="Player 2 difficulty")
    basic.add_argument("--board", type=str, default="square8", help="Board type")
    basic.add_argument("--games", type=int, default=100, help="Number of games")
    basic.add_argument("--max-moves", type=int, default=10000, help="Max moves per game")

    # Models mode (from run_model_elo_tournament.py)
    models = subparsers.add_parser("models", help="Neural network model evaluation")
    models.add_argument("--board", type=str, default="square8", help="Board type")
    models.add_argument("--players", type=int, default=2, help="Number of players")
    models.add_argument("--games", type=int, help="Games per matchup (default from config)")
    models.add_argument("--models-dir", type=str, help="Models directory")
    models.add_argument("--include-baselines", action="store_true", help="Include baseline AIs")
    models.add_argument("--continuous", action="store_true", help="Run continuously")
    models.add_argument("--interval", type=int, default=3600, help="Continuous interval (seconds)")
    models.add_argument("--all-configs", action="store_true", help="Run all board/player configs")
    models.add_argument("--elo-matchmaking", action="store_true", help="Match by Elo rating")

    # Distributed mode (from run_distributed_tournament.py)
    distributed = subparsers.add_parser("distributed", help="Multi-tier difficulty ladder")
    distributed.add_argument("--board", type=str, default="square8", help="Board type")
    distributed.add_argument("--players", type=int, default=2, help="Number of players")
    distributed.add_argument("--tiers", type=str, default="D1-D10", help="Tiers to include (e.g., D1-D5 or D1,D3,D5)")
    distributed.add_argument("--games", type=int, default=50, help="Games per matchup")
    distributed.add_argument("--workers", type=int, default=4, help="Parallel workers")
    distributed.add_argument("--resume", type=str, help="Resume from checkpoint")

    # SSH mode (from run_ssh_distributed_tournament.py)
    ssh = subparsers.add_parser("ssh", help="SSH-distributed tournament")
    ssh.add_argument("--board", type=str, default="square8", help="Board type")
    ssh.add_argument("--players", type=int, default=2, help="Number of players")
    ssh.add_argument("--tiers", type=str, default="D1-D10", help="Tiers to include")
    ssh.add_argument("--games", type=int, default=50, help="Games per matchup")
    ssh.add_argument("--hosts", type=str, help="Hosts config file")

    # Eval mode (from run_eval_tournaments.py)
    eval_parser = subparsers.add_parser("eval", help="Evaluation pool tournament")
    eval_parser.add_argument("--pool", type=str, required=True, help="Evaluation pool name")
    eval_parser.add_argument("--ai-specs", type=str, nargs="+", help="AI specs to evaluate")
    eval_parser.add_argument("--games", type=int, default=20, help="Games per scenario")
    eval_parser.add_argument("--demo", action="store_true", help="Demo mode (fewer games)")

    # Diverse mode (from run_diverse_tournaments.py)
    diverse = subparsers.add_parser("diverse", help="All board/player configurations")
    diverse.add_argument("--orchestrator", type=str, choices=["local", "distributed"], default="local")
    diverse.add_argument("--games", type=int, default=50, help="Games per matchup")
    diverse.add_argument("--continuous", action="store_true", help="Run continuously")
    diverse.add_argument("--interval", type=int, default=3600, help="Interval between runs (seconds)")

    # Weights mode (from run_axis_aligned_tournament.py)
    weights = subparsers.add_parser("weights", help="Heuristic weight profile tournament")
    weights.add_argument("--board", type=str, default="square8", help="Board type")
    weights.add_argument("--profiles-dir", type=str, required=True, help="Profiles directory")
    weights.add_argument("--games", type=int, default=20, help="Games per matchup")
    weights.add_argument("--include-baselines", action="store_true", help="Include baseline AIs")

    # Crossboard mode (from run_crossboard_difficulty_tournament.py)
    crossboard = subparsers.add_parser("crossboard", help="Cross-board difficulty analysis")
    crossboard.add_argument("--boards", type=str, default="square8,square19,hexagonal", help="Boards to compare")
    crossboard.add_argument("--players", type=int, default=2, help="Number of players")
    crossboard.add_argument("--tiers", type=str, default="D1-D10", help="Tiers to include")
    crossboard.add_argument("--games", type=int, default=30, help="Games per matchup")
    crossboard.add_argument("--demo", action="store_true", help="Demo mode")

    return parser


def run_basic_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run basic AI vs AI tournament."""
    # Delegate to original script logic
    from scripts.run_ai_tournament import main as ai_tournament_main

    # Convert args to original format
    sys.argv = [
        "run_ai_tournament.py",
        "--p1", args.p1,
        "--p1-diff", str(args.p1_diff),
        "--p2", args.p2,
        "--p2-diff", str(args.p2_diff),
        "--board", args.board,
        "--games", str(args.games),
        "--max-moves", str(args.max_moves),
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
    ]
    return ai_tournament_main()


def run_models_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run neural network model evaluation."""
    # Delegate to original script logic
    from scripts.run_model_elo_tournament import main as model_tournament_main

    # Convert args to original format
    sys_argv = [
        "run_model_elo_tournament.py",
        "--board-type", args.board,
        "--num-players", str(args.players),
    ]
    if args.games:
        sys_argv.extend(["--games-per-matchup", str(args.games)])
    if args.models_dir:
        sys_argv.extend(["--models-dir", args.models_dir])
    if args.include_baselines:
        sys_argv.append("--include-baselines")
    if args.continuous:
        sys_argv.append("--continuous")
        sys_argv.extend(["--continuous-interval", str(args.interval)])
    if args.all_configs:
        sys_argv.append("--all-configs")
    if args.elo_matchmaking:
        sys_argv.append("--elo-matchmaking")

    sys.argv = sys_argv
    return model_tournament_main()


def run_distributed_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run distributed difficulty ladder tournament."""
    from scripts.run_distributed_tournament import main as distributed_main

    sys_argv = [
        "run_distributed_tournament.py",
        "--board-type", args.board,
        "--num-players", str(args.players),
        "--tiers", args.tiers,
        "--games-per-matchup", str(args.games),
        "--max-workers", str(args.workers),
        "--output-dir", args.output_dir,
    ]
    if args.resume:
        sys_argv.extend(["--resume", args.resume])

    sys.argv = sys_argv
    return distributed_main()


def run_ssh_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run SSH-distributed tournament."""
    from scripts.run_ssh_distributed_tournament import main as ssh_main

    sys_argv = [
        "run_ssh_distributed_tournament.py",
        "--board-type", args.board,
        "--num-players", str(args.players),
        "--tiers", args.tiers,
        "--games-per-matchup", str(args.games),
    ]
    if args.hosts:
        sys_argv.extend(["--hosts-config", args.hosts])

    sys.argv = sys_argv
    return ssh_main()


def run_eval_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run evaluation pool tournament."""
    from scripts.run_eval_tournaments import main as eval_main

    sys_argv = [
        "run_eval_tournaments.py",
        "--pool", args.pool,
        "--games", str(args.games),
    ]
    if args.ai_specs:
        sys_argv.extend(["--ai-specs"] + args.ai_specs)
    if args.demo:
        sys_argv.append("--demo")

    sys.argv = sys_argv
    return eval_main()


def run_diverse_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run all board/player configuration tournaments."""
    from scripts.run_diverse_tournaments import main as diverse_main

    sys_argv = [
        "run_diverse_tournaments.py",
        f"--{args.orchestrator}",
        "--games-per-matchup", str(args.games),
    ]
    if args.continuous:
        sys_argv.append("--continuous")
        sys_argv.extend(["--interval-hours", str(args.interval // 3600)])

    sys.argv = sys_argv
    return diverse_main()


def run_weights_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run heuristic weight profile tournament."""
    from scripts.run_axis_aligned_tournament import main as weights_main

    sys_argv = [
        "run_axis_aligned_tournament.py",
        "--board-type", args.board,
        "--profiles-dir", args.profiles_dir,
        "--games-per-matchup", str(args.games),
    ]
    if args.include_baselines:
        sys_argv.append("--include-baselines")

    sys.argv = sys_argv
    return weights_main()


def run_crossboard_tournament(args: argparse.Namespace, config: Any) -> int:
    """Run cross-board difficulty analysis."""
    from scripts.run_crossboard_difficulty_tournament import main as crossboard_main

    sys_argv = [
        "run_crossboard_difficulty_tournament.py",
        "--boards", args.boards,
        "--num-players", str(args.players),
        "--tiers", args.tiers,
        "--games-per-matchup", str(args.games),
    ]
    if args.demo:
        sys_argv.append("--demo")

    sys.argv = sys_argv
    return crossboard_main()


# Mode dispatch table
MODE_HANDLERS = {
    "basic": run_basic_tournament,
    "models": run_models_tournament,
    "distributed": run_distributed_tournament,
    "ssh": run_ssh_tournament,
    "eval": run_eval_tournament,
    "diverse": run_diverse_tournament,
    "weights": run_weights_tournament,
    "crossboard": run_crossboard_tournament,
}


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        print("\nError: Please specify a tournament mode.")
        return 1

    setup_logging(args.verbose)

    # Load config
    config_path = args.config or str(AI_SERVICE_ROOT / "config" / "unified_loop.yaml")
    config = get_config(config_path)

    # Apply config defaults
    if hasattr(args, "games") and args.games is None:
        args.games = config.tournament.default_games_per_matchup

    logger.info(f"Starting tournament mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dispatch to mode handler
    handler = MODE_HANDLERS.get(args.mode)
    if handler:
        try:
            return handler(args, config)
        except ImportError as e:
            logger.error(f"Failed to import tournament module: {e}")
            logger.error("Make sure the original script exists and has a main() function.")
            return 1
        except Exception as e:
            logger.exception(f"Tournament failed: {e}")
            return 1
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

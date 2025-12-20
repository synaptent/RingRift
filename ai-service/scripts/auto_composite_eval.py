#!/usr/bin/env python3
"""Automated Composite ELO Evaluation Runner.

This script runs periodic composite ELO evaluations to:
1. Evaluate new models against baselines
2. Increase games for participants with low reliability
3. Run consistency checks and report issues

Usage:
    # Single run
    python scripts/auto_composite_eval.py

    # Continuous mode (runs every N minutes)
    python scripts/auto_composite_eval.py --continuous --interval 30

    # Focus on specific algorithms
    python scripts/auto_composite_eval.py --algorithms mcts ebmo

    # Increase reliability (more games for low-game participants)
    python scripts/auto_composite_eval.py --increase-reliability --target-games 50
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.elo_service import get_elo_service
from app.tournament.consistency_monitor import run_consistency_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_low_reliability_participants(
    board_type: str,
    num_players: int,
    target_games: int = 30,
) -> list[dict]:
    """Get participants with fewer than target games."""
    elo_service = get_elo_service()
    leaderboard = elo_service.get_composite_leaderboard(
        board_type=board_type,
        num_players=num_players,
        min_games=0,
        limit=1000,
    )

    low_reliability = [
        p for p in leaderboard
        if p.get("games_played", 0) < target_games
        and p.get("games_played", 0) > 0  # Has some games
    ]

    return sorted(low_reliability, key=lambda x: x.get("games_played", 0))


def get_new_models(
    board_type: str,
    num_players: int,
    model_dir: Path,
    max_age_hours: int = 24,
) -> list[Path]:
    """Find new model files that haven't been evaluated."""
    import time

    cutoff = time.time() - (max_age_hours * 3600)

    # Board-specific patterns to avoid hex/hex8 confusion
    # hex8 = 9x9 hexagonal, hexagonal = 25x25 hexagonal
    board_patterns = {
        "square8": [f"*sq8*{num_players}p*.pth", f"*square8*{num_players}p*.pth"],
        "square19": [f"*sq19*{num_players}p*.pth", f"*square19*{num_players}p*.pth"],
        "hexagonal": [f"hex_{num_players}p*.pth", f"hex*{num_players}p*.pth"],  # hex_2p, not hex8_2p
        "hex8": [f"*hex8*{num_players}p*.pth"],  # Only hex8
    }

    # Exclusion patterns (to filter out from results)
    exclusions = {
        "hexagonal": ["hex8", "hex_8"],  # Exclude hex8 models from hexagonal
    }

    patterns = board_patterns.get(board_type, [f"*{board_type[:3]}*{num_players}p*.pth"])
    exclude_list = exclusions.get(board_type, [])

    recent_models = []

    for pattern in patterns:
        for model_path in model_dir.glob(pattern):
            # Check if model should be excluded
            name_lower = model_path.name.lower()
            if any(excl in name_lower for excl in exclude_list):
                continue
            if model_path.stat().st_mtime > cutoff:
                recent_models.append(model_path)

    # Deduplicate (in case multiple patterns match same file)
    recent_models = list(set(recent_models))

    return sorted(recent_models, key=lambda x: x.stat().st_mtime, reverse=True)


async def run_evaluation_round(
    board_type: str,
    num_players: int,
    algorithms: list[str],
    games_per_matchup: int = 10,
    max_models: int = 5,
) -> dict:
    """Run one round of evaluation."""
    from app.tournament.composite_gauntlet import (
        CompositeGauntlet,
        CompositeGauntletConfig,
        GauntletPhaseConfig,
    )

    results = {
        "games_played": 0,
        "models_evaluated": 0,
        "errors": [],
    }

    # Find models to evaluate
    model_dir = Path("models")
    new_models = get_new_models(board_type, num_players, model_dir)[:max_models]

    if not new_models:
        logger.info("No new models to evaluate")
        return results

    logger.info(f"Evaluating {len(new_models)} models with algorithms: {algorithms}")

    config = CompositeGauntletConfig(
        phase1=GauntletPhaseConfig(
            games_per_matchup=games_per_matchup,
            pass_threshold=0.8,  # Keep most for Phase 2
        ),
        phase2=GauntletPhaseConfig(
            games_per_matchup=games_per_matchup,
        ),
        phase2_algorithms=algorithms,
    )

    gauntlet = CompositeGauntlet(
        board_type=board_type,
        num_players=num_players,
        config=config,
    )

    try:
        result = await gauntlet.run_two_phase_gauntlet(new_models)
        results["games_played"] = result.total_games
        results["models_evaluated"] = len(new_models)
        logger.info(f"Evaluation complete: {result.total_games} games played")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        results["errors"].append(str(e))

    return results


async def increase_reliability(
    board_type: str,
    num_players: int,
    target_games: int = 50,
    games_to_add: int = 20,
) -> dict:
    """Run additional games for low-reliability participants."""
    from app.tournament.composite_gauntlet import CompositeGauntlet

    results = {
        "participants_boosted": 0,
        "games_added": 0,
    }

    low_rel = get_low_reliability_participants(board_type, num_players, target_games)

    if not low_rel:
        logger.info("All participants have reliable ratings")
        return results

    logger.info(f"Found {len(low_rel)} participants needing more games")

    # Group by algorithm to run efficient matchups
    # For now, just report - actual games would need model paths
    for p in low_rel[:10]:
        nn_id = p.get("nn_model_id", "unknown")
        algo = p.get("ai_algorithm", "unknown")
        games = p.get("games_played", 0)
        needed = target_games - games
        logger.info(f"  {nn_id[:40]}:{algo} - {games} games (needs {needed} more)")
        results["participants_boosted"] += 1

    return results


def run_consistency_report(board_type: str, num_players: int) -> bool:
    """Run consistency checks and report issues."""
    logger.info(f"Running consistency checks for {board_type}/{num_players}p")

    report = run_consistency_checks(board_type, num_players)

    if report.overall_healthy:
        logger.info("System is HEALTHY")
    else:
        logger.warning(f"System has issues: {len(report.errors)} errors, {len(report.warnings)} warnings")

    for check in report.checks:
        status = "PASS" if check.passed else f"FAIL ({check.severity})"
        logger.info(f"  {check.name}: {status}")
        if not check.passed:
            logger.info(f"    -> {check.message}")

    return report.overall_healthy


async def continuous_mode(args):
    """Run evaluations continuously."""
    logger.info(f"Starting continuous mode, interval: {args.interval} minutes")

    while True:
        logger.info("=" * 60)
        logger.info(f"Starting evaluation round at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run consistency check
        healthy = run_consistency_report(args.board, args.players)

        # Run evaluation if healthy
        if healthy or args.force:
            results = await run_evaluation_round(
                board_type=args.board,
                num_players=args.players,
                algorithms=args.algorithms,
                games_per_matchup=args.games,
                max_models=args.max_models,
            )
            logger.info(f"Round complete: {results}")

        # Increase reliability if requested
        if args.increase_reliability:
            rel_results = await increase_reliability(
                board_type=args.board,
                num_players=args.players,
                target_games=args.target_games,
            )
            logger.info(f"Reliability boost: {rel_results}")

        logger.info(f"Sleeping for {args.interval} minutes...")
        await asyncio.sleep(args.interval * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Automated Composite ELO Evaluation"
    )

    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
    )
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["mcts", "ebmo"],
        help="Algorithms to evaluate",
    )
    parser.add_argument("--games", type=int, default=10, help="Games per matchup")
    parser.add_argument("--max-models", type=int, default=5)

    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between runs")
    parser.add_argument("--force", action="store_true", help="Run even if unhealthy")

    parser.add_argument("--increase-reliability", action="store_true")
    parser.add_argument("--target-games", type=int, default=50)

    parser.add_argument("--check-only", action="store_true", help="Only run consistency checks")

    args = parser.parse_args()

    if args.check_only:
        run_consistency_report(args.board, args.players)
        return

    if args.continuous:
        await continuous_mode(args)
    else:
        # Single run
        run_consistency_report(args.board, args.players)

        results = await run_evaluation_round(
            board_type=args.board,
            num_players=args.players,
            algorithms=args.algorithms,
            games_per_matchup=args.games,
            max_models=args.max_models,
        )

        if args.increase_reliability:
            await increase_reliability(
                board_type=args.board,
                num_players=args.players,
                target_games=args.target_games,
            )

        print(f"\nResults: {results}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""CLI for running AI agent tournaments with Elo ratings.

Usage:
    # Run round-robin tournament with built-in agents
    python scripts/run_tournament.py --board-type square8 --num-players 2

    # Run with specific agents
    python scripts/run_tournament.py --agents baseline_v1 aggressive_v1 defensive_v1

    # Run 4-player tournament
    python scripts/run_tournament.py --board-type square8 --num-players 4 --games-per-pairing 4

    # Save results to file
    python scripts/run_tournament.py --output results/tournament_2024.json
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import BoardType
from app.tournament import (
    AIAgentRegistry,
    EloCalculator,
    RoundRobinScheduler,
    SwissScheduler,
    TournamentRunner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AI agent tournament with Elo ratings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Agent IDs to include (default: all built-in agents)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type for matches (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players per match (default: 2)",
    )
    parser.add_argument(
        "--games-per-pairing",
        type=int,
        default=2,
        help="Number of games per agent pairing (default: 2)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game (default: 500)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="round-robin",
        choices=["round-robin", "swiss"],
        help="Tournament scheduler type (default: round-robin)",
    )
    parser.add_argument(
        "--swiss-rounds",
        type=int,
        default=5,
        help="Number of Swiss rounds (only for swiss scheduler)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agents and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def progress_callback(completed: int, total: int) -> None:
    """Display progress bar."""
    pct = completed / total * 100 if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * completed / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {completed}/{total} ({pct:.1f}%)", end="", flush=True)
    if completed == total:
        print()  # Newline when done


def print_leaderboard(runner: TournamentRunner) -> None:
    """Print formatted leaderboard."""
    print("\n" + "=" * 60)
    print("FINAL LEADERBOARD")
    print("=" * 60)
    print(f"{'Rank':<6} {'Agent':<20} {'Rating':<10} {'W/L/D':<12} {'Win%':<8}")
    print("-" * 60)

    leaderboard = runner.get_leaderboard()
    for rank, (agent_id, rating, stats) in enumerate(leaderboard, 1):
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        draws = stats.get("draws", 0)
        win_rate = stats.get("win_rate", 0) * 100

        print(
            f"{rank:<6} {agent_id:<20} {rating:<10.1f} "
            f"{wins}/{losses}/{draws:<8} {win_rate:<8.1f}%"
        )

    print("=" * 60)


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize registry
    registry = AIAgentRegistry()

    # List agents if requested
    if args.list_agents:
        print("Available agents:")
        for agent in registry.list_agents():
            print(f"  {agent.agent_id:<20} - {agent.description}")
        return 0

    # Get agent IDs
    if args.agents:
        agent_ids = args.agents
        # Validate all agents exist
        for agent_id in agent_ids:
            if registry.get(agent_id) is None:
                logger.error(f"Agent not found: {agent_id}")
                return 1
    else:
        agent_ids = registry.list_agent_ids()

    # Validate we have enough agents
    if len(agent_ids) < args.num_players:
        logger.error(
            f"Need at least {args.num_players} agents for {args.num_players}-player matches"
        )
        return 1

    # Create scheduler
    if args.scheduler == "round-robin":
        scheduler = RoundRobinScheduler(
            games_per_pairing=args.games_per_pairing,
            shuffle_order=True,
            seed=args.seed,
        )
    else:
        if args.num_players != 2:
            logger.error("Swiss scheduler only supports 2-player matches")
            return 1
        scheduler = SwissScheduler(
            rounds=args.swiss_rounds,
            seed=args.seed,
        )

    # Create Elo calculator
    elo = EloCalculator()

    # Create runner
    runner = TournamentRunner(
        agent_registry=registry,
        scheduler=scheduler,
        elo_calculator=elo,
        max_workers=args.max_workers,
        max_moves=args.max_moves,
        seed=args.seed,
    )

    # Run tournament
    board_type = BoardType(args.board_type)

    print(f"\nStarting tournament:")
    print(f"  Board type: {args.board_type}")
    print(f"  Players per match: {args.num_players}")
    print(f"  Agents: {', '.join(agent_ids)}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Games per pairing: {args.games_per_pairing}")
    print(f"  Max moves: {args.max_moves}")
    print(f"  Workers: {args.max_workers}")
    print()

    try:
        results = runner.run_tournament(
            agent_ids=agent_ids,
            board_type=board_type,
            num_players=args.num_players,
            games_per_pairing=args.games_per_pairing,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.error(f"Tournament failed: {e}")
        raise

    # Print results
    print_leaderboard(runner)

    # Print summary
    stats = scheduler.get_stats()
    print(f"\nTournament completed:")
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    if results.completed_at and results.started_at:
        duration = (results.completed_at - results.started_at).total_seconds()
        print(f"  Duration: {duration:.1f}s")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        runner.save_results(output_path)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

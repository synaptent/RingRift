#!/usr/bin/env python3
"""Quick gauntlet evaluation for a model.

Usage:
    python scripts/quick_gauntlet.py --model models/hex8_2p_iter1.pth
    python scripts/quick_gauntlet.py --model models/canonical_hex8_2p.pth --games 30
    python scripts/quick_gauntlet.py --model models/my_model.pth --board-type square8 --num-players 4
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick gauntlet evaluation for a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a specific model
    python scripts/quick_gauntlet.py --model models/hex8_2p_iter1.pth

    # Evaluate with more games
    python scripts/quick_gauntlet.py --model models/canonical_hex8_2p.pth --games 30

    # Evaluate a 4-player model
    python scripts/quick_gauntlet.py --model models/sq8_4p.pth --board-type square8 --num-players 4
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model to evaluate (REQUIRED)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="hex8",
        choices=["hex8", "square8", "square19", "hexagonal"],
        help="Board type (default: hex8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=15,
        help="Games per opponent (default: 15)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel opponent evaluation (may cause issues on some systems)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate model path exists - NO SILENT FALLBACK
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        print(f"Please specify a valid model path with --model", file=sys.stderr)
        sys.exit(1)

    print(f"Running gauntlet for {model_path.name}...")
    print(f"  Board: {args.board_type}, Players: {args.num_players}, Games/opponent: {args.games}")
    print("=" * 60)

    result = run_baseline_gauntlet(
        model_path=str(model_path),
        board_type=args.board_type,
        num_players=args.num_players,
        games_per_opponent=args.games,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        verbose=True,
        early_stopping=True,
        parallel_opponents=args.parallel,
    )

    print("\n" + "=" * 60)
    print("GAUNTLET RESULTS")
    print("=" * 60)
    print(f"Model: {model_path.name}")
    print(f"Passed baseline gating: {result.passes_baseline_gating}")
    print(f"Total wins: {result.total_wins}/{result.total_games}")
    print(f"Win rate: {result.win_rate*100:.1f}%")

    for opp_name, stats in result.opponent_results.items():
        win_rate = stats.get('win_rate', 0)
        wins = stats.get('wins', 0)
        games = stats.get('games', 0)
        print(f"  vs {opp_name}: {wins}/{games} ({win_rate*100:.0f}%)")

    if hasattr(result, 'estimated_elo') and result.estimated_elo:
        print(f"\nEstimated Elo: {result.estimated_elo:.0f}")


if __name__ == "__main__":
    main()

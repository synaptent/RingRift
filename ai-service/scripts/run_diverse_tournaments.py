#!/usr/bin/env python3
"""Orchestrate diverse tournaments across all board/player combinations.

This script schedules and runs diverse selfplay tournaments for Elo calibration
across all supported configurations:
- Board types: square8, square19, hexagonal
- Player counts: 2, 3, 4

Usage:
    # Run a single round of tournaments (all combinations)
    python scripts/run_diverse_tournaments.py --games-per-config 100

    # Run continuously every 4 hours
    python scripts/run_diverse_tournaments.py --games-per-config 200 --interval-hours 4

    # Run specific configurations only
    python scripts/run_diverse_tournaments.py --board-types square8 square19 --player-counts 2

    # Dry run to see what would be scheduled
    python scripts/run_diverse_tournaments.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TournamentConfig:
    """Configuration for a single tournament."""
    board_type: str
    num_players: int
    num_games: int
    output_dir: str
    seed: Optional[int] = None


@dataclass
class TournamentResult:
    """Result of running a tournament."""
    config: TournamentConfig
    success: bool
    games_completed: int
    samples_generated: int
    duration_sec: float
    error: Optional[str] = None


# All supported configurations
ALL_BOARD_TYPES = ["square8", "square19", "hexagonal"]
ALL_PLAYER_COUNTS = [2, 3, 4]

# Recommended games per config for meaningful Elo estimates
DEFAULT_GAMES_PER_CONFIG = {
    ("square8", 2): 200,      # Fast, run more
    ("square8", 3): 100,      # Slower with 3 players
    ("square8", 4): 100,      # Slower with 4 players
    ("square19", 2): 50,      # Larger board, slower
    ("square19", 3): 30,
    ("square19", 4): 30,
    ("hexagonal", 2): 100,    # Medium speed
    ("hexagonal", 3): 50,
    ("hexagonal", 4): 50,
}


def get_output_path(base_dir: str, board_type: str, num_players: int) -> str:
    """Generate output path for a tournament."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"diverse_{board_type}_{num_players}p_{timestamp}.jsonl"
    return os.path.join(base_dir, filename)


def run_tournament(config: TournamentConfig) -> TournamentResult:
    """Run a single diverse tournament."""
    start_time = time.time()

    logger.info(
        f"Starting tournament: {config.board_type} {config.num_players}p, "
        f"{config.num_games} games"
    )

    # Build command
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_distributed_selfplay.py"),
        "--board-type", config.board_type,
        "--num-players", str(config.num_players),
        "--num-games", str(config.num_games),
        "--engine-mode", "diverse",
        "--output", f"file://{config.output_dir}",
    ]

    if config.seed is not None:
        cmd.extend(["--seed", str(config.seed)])

    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per tournament
        )

        duration = time.time() - start_time

        # Parse stats from output
        games_completed = 0
        samples_generated = 0

        for line in result.stdout.split("\n"):
            if "Games completed:" in line:
                try:
                    games_completed = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "Samples generated:" in line:
                try:
                    samples_generated = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        success = result.returncode == 0
        error = result.stderr if not success else None

        logger.info(
            f"Tournament complete: {config.board_type} {config.num_players}p, "
            f"{games_completed} games, {samples_generated} samples, "
            f"{duration:.1f}s"
        )

        return TournamentResult(
            config=config,
            success=success,
            games_completed=games_completed,
            samples_generated=samples_generated,
            duration_sec=duration,
            error=error,
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Tournament timed out: {config.board_type} {config.num_players}p")
        return TournamentResult(
            config=config,
            success=False,
            games_completed=0,
            samples_generated=0,
            duration_sec=duration,
            error="Timeout after 1 hour",
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Tournament failed: {config.board_type} {config.num_players}p: {e}")
        return TournamentResult(
            config=config,
            success=False,
            games_completed=0,
            samples_generated=0,
            duration_sec=duration,
            error=str(e),
        )


def run_tournament_round(
    board_types: List[str],
    player_counts: List[int],
    games_per_config: Optional[int],
    output_base: str,
    seed: Optional[int] = None,
) -> List[TournamentResult]:
    """Run one round of tournaments for all configured combinations."""

    results: List[TournamentResult] = []

    # Ensure output directory exists
    os.makedirs(output_base, exist_ok=True)

    for board_type in board_types:
        for num_players in player_counts:
            # Determine number of games
            if games_per_config:
                num_games = games_per_config
            else:
                num_games = DEFAULT_GAMES_PER_CONFIG.get(
                    (board_type, num_players), 50
                )

            config = TournamentConfig(
                board_type=board_type,
                num_players=num_players,
                num_games=num_games,
                output_dir=get_output_path(output_base, board_type, num_players),
                seed=seed,
            )

            result = run_tournament(config)
            results.append(result)

    return results


def print_summary(results: List[TournamentResult]) -> None:
    """Print summary of tournament results."""
    print("\n" + "=" * 70)
    print("TOURNAMENT ROUND SUMMARY")
    print("=" * 70)

    total_games = sum(r.games_completed for r in results)
    total_samples = sum(r.samples_generated for r in results)
    total_duration = sum(r.duration_sec for r in results)
    successful = sum(1 for r in results if r.success)

    print(f"\nConfigurations: {len(results)} ({successful} successful)")
    print(f"Total games:    {total_games}")
    print(f"Total samples:  {total_samples}")
    print(f"Total time:     {total_duration:.1f}s ({total_duration/60:.1f} min)")

    print("\nPer-configuration breakdown:")
    print("-" * 70)
    print(f"{'Config':<25} {'Games':>8} {'Samples':>10} {'Time':>10} {'Status':>10}")
    print("-" * 70)

    for r in results:
        config_name = f"{r.config.board_type} {r.config.num_players}p"
        status = "OK" if r.success else "FAILED"
        print(
            f"{config_name:<25} {r.games_completed:>8} "
            f"{r.samples_generated:>10} {r.duration_sec:>9.1f}s {status:>10}"
        )

    print("-" * 70)
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate diverse tournaments across board/player combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--board-types",
        nargs="+",
        choices=ALL_BOARD_TYPES,
        default=ALL_BOARD_TYPES,
        help=f"Board types to include (default: all)",
    )
    parser.add_argument(
        "--player-counts",
        nargs="+",
        type=int,
        choices=ALL_PLAYER_COUNTS,
        default=ALL_PLAYER_COUNTS,
        help=f"Player counts to include (default: all)",
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=None,
        help="Games per configuration (default: varies by config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "data" / "tournaments"),
        help="Output directory for tournament results",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Run continuously with this interval (hours). Omit for single run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Show configuration
    configs_to_run = [
        (bt, np) for bt in args.board_types for np in args.player_counts
    ]

    logger.info(f"Tournament configurations: {len(configs_to_run)}")
    for bt, np in configs_to_run:
        games = args.games_per_config or DEFAULT_GAMES_PER_CONFIG.get((bt, np), 50)
        logger.info(f"  {bt} {np}p: {games} games")

    if args.dry_run:
        logger.info("Dry run - not executing")
        return 0

    # Run tournaments
    round_num = 0
    while True:
        round_num += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"TOURNAMENT ROUND {round_num}")
        logger.info(f"{'='*60}\n")

        results = run_tournament_round(
            board_types=args.board_types,
            player_counts=args.player_counts,
            games_per_config=args.games_per_config,
            output_base=args.output_dir,
            seed=args.seed,
        )

        print_summary(results)

        if args.interval_hours is None:
            # Single run mode
            break

        # Wait for next round
        wait_sec = args.interval_hours * 3600
        logger.info(f"Waiting {args.interval_hours} hours until next round...")
        time.sleep(wait_sec)

    return 0


if __name__ == "__main__":
    sys.exit(main())

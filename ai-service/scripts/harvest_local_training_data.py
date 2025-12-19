#!/usr/bin/env python3
"""
Local Training Data Harvester

Harvests high-quality training data from the local node's selfplay data.
Run this on each cluster node, then aggregate the results.

Features:
- Quality scoring based on multiple criteria
- Deduplication
- Progress reporting
- JSON output with metadata

Usage:
    # Basic harvest
    python scripts/harvest_local_training_data.py \\
        --board-type square8 --num-players 2 \\
        --output data/harvested/local.jsonl

    # With quality threshold
    python scripts/harvest_local_training_data.py \\
        --board-type square8 --num-players 2 \\
        --min-quality 0.75 --max-games 50000 \\
        --output data/harvested/quality_games.jsonl

    # Verbose mode with JSON logs
    python scripts/harvest_local_training_data.py \\
        --board-type square8 --num-players 2 \\
        --output data/harvested/local.jsonl \\
        --verbose --json-logs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.logging_config import setup_script_logging, get_logger, get_metrics_logger
from scripts.lib.data_quality import (
    GameQualityScorer,
    QualityFilter,
    GameQuality,
    QualityStats,
    VictoryType,
    compute_quality_stats,
)

logger = get_logger(__name__)


def find_jsonl_files(data_dir: Path, board_type: str, num_players: int) -> List[Path]:
    """Find all JSONL files that might contain matching games.

    Args:
        data_dir: Root data directory to search
        board_type: Board type to filter for
        num_players: Number of players to filter for

    Returns:
        List of JSONL file paths
    """
    files = []
    config_pattern = f"{board_type}_{num_players}p"

    # Search patterns
    patterns = [
        f"**/*{config_pattern}*/*.jsonl",
        f"**/*{board_type}*{num_players}p*/*.jsonl",
        f"**/selfplay/**/*.jsonl",
    ]

    seen = set()
    for pattern in patterns:
        for path in data_dir.glob(pattern):
            if path not in seen:
                seen.add(path)
                files.append(path)

    logger.info(f"Found {len(files)} JSONL files to process")
    return files


def stream_games_from_file(
    file_path: Path,
    board_type: str,
    num_players: int,
    sample_rate: float = 1.0,
) -> Generator[Dict[str, Any], None, None]:
    """Stream games from a JSONL file with optional sampling.

    Args:
        file_path: Path to JSONL file
        board_type: Filter for this board type
        num_players: Filter for this number of players
        sample_rate: Fraction of games to sample (1.0 = all)

    Yields:
        Game dictionaries that match the filter
    """
    try:
        with open(file_path, "r") as f:
            for line in f:
                # Random sampling
                if sample_rate < 1.0 and random.random() > sample_rate:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    game = json.loads(line)

                    # Filter by config
                    game_board = game.get("board_type", "")
                    game_players = game.get("num_players", 0)

                    if game_board == board_type and game_players == num_players:
                        yield game

                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping invalid JSON in {file_path}: {e}")

    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")


def harvest_local_data(
    data_dir: Path,
    board_type: str,
    num_players: int,
    min_quality: float,
    max_games: int,
    output_file: Path,
    sample_rate: float = 1.0,
    shuffle_files: bool = True,
) -> Tuple[int, QualityStats]:
    """Harvest high-quality games from local data.

    Args:
        data_dir: Directory containing selfplay data
        board_type: Board type to harvest
        num_players: Number of players
        min_quality: Minimum quality score (0-1)
        max_games: Maximum games to harvest
        output_file: Output JSONL file path
        sample_rate: Fraction of games to sample from each file
        shuffle_files: Randomize file processing order

    Returns:
        Tuple of (games_harvested, quality_stats)
    """
    config_key = f"{board_type}_{num_players}p"
    hostname = os.uname().nodename
    metrics = get_metrics_logger("harvest")

    logger.info(f"Starting harvest for {config_key} on {hostname}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Min quality: {min_quality}, Max games: {max_games}")

    # Initialize scorer and filter
    scorer = GameQualityScorer(config_key=config_key)
    quality_filter = QualityFilter(
        min_quality=min_quality,
        max_games=max_games,
        deduplicate=True,
    )

    # Find files
    files = find_jsonl_files(data_dir, board_type, num_players)
    if shuffle_files:
        random.shuffle(files)

    # Process games
    quality_games: List[Tuple[GameQuality, Dict[str, Any]]] = []
    seen_ids: set = set()
    games_checked = 0
    quality_distribution: Counter = Counter()

    start_time = time.time()

    for file_idx, file_path in enumerate(files):
        if len(quality_games) >= max_games:
            logger.info(f"Reached max games limit ({max_games})")
            break

        file_games = 0
        file_quality = 0

        for game in stream_games_from_file(file_path, board_type, num_players, sample_rate):
            games_checked += 1

            # Score quality
            quality = scorer.score(
                game,
                source_file=str(file_path),
                source_node=hostname,
            )

            # Track distribution
            bucket = quality.quality_bucket
            quality_distribution[bucket] += 1

            # Filter
            if quality.total_score >= min_quality:
                if quality.game_id not in seen_ids:
                    seen_ids.add(quality.game_id)
                    quality_games.append((quality, game))
                    file_games += 1
                    file_quality += quality.total_score

                    if len(quality_games) >= max_games:
                        break

            # Progress logging
            if games_checked % 10000 == 0:
                elapsed = time.time() - start_time
                rate = games_checked / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {games_checked:,} checked, {len(quality_games):,} harvested "
                    f"({rate:.0f} games/sec)"
                )
                metrics.set("games_checked", games_checked)
                metrics.set("games_harvested", len(quality_games))

        if file_games > 0:
            avg_quality = file_quality / file_games
            logger.debug(f"File {file_path.name}: {file_games} games (avg quality: {avg_quality:.2f})")

    # Sort by quality
    quality_games.sort(key=lambda x: x[0].total_score, reverse=True)
    quality_games = quality_games[:max_games]

    elapsed = time.time() - start_time
    logger.info(
        f"Harvest complete: {games_checked:,} checked, {len(quality_games):,} harvested "
        f"in {elapsed:.1f}s"
    )

    # Compute stats
    qualities = [q for q, _ in quality_games]
    stats = compute_quality_stats(qualities)
    stats.total_games = games_checked

    # Log quality distribution
    logger.info("Quality distribution:")
    for bucket in sorted(quality_distribution.keys(), reverse=True):
        count = quality_distribution[bucket]
        bar = "█" * min(50, count // 100)
        logger.info(f"  {bucket}: {count:>6} {bar}")

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for quality, game in quality_games:
            # Add quality metadata
            game["_quality_score"] = quality.total_score
            game["_quality_scores"] = quality.scores.to_dict()
            game["_source_node"] = hostname
            game["_source_file"] = quality.source_file
            game["_harvested_at"] = datetime.now().isoformat()

            f.write(json.dumps(game) + "\n")

    logger.info(f"Saved {len(quality_games):,} games to {output_file}")

    # Save stats
    stats_file = output_file.with_suffix(".stats.json")
    stats_data = {
        "hostname": hostname,
        "config": config_key,
        "data_dir": str(data_dir),
        "min_quality": min_quality,
        "max_games": max_games,
        "games_checked": games_checked,
        "games_harvested": len(quality_games),
        "elapsed_seconds": elapsed,
        "quality_distribution": dict(quality_distribution),
        "stats": stats.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    with open(stats_file, "w") as f:
        json.dump(stats_data, f, indent=2)

    logger.info(f"Saved stats to {stats_file}")

    return len(quality_games), stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Harvest high-quality training data from local selfplay games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --board-type square8 --num-players 2 --output local.jsonl
  %(prog)s --board-type hex8 --num-players 3 --min-quality 0.75 --output quality.jsonl
  %(prog)s --board-type square8 --num-players 2 --max-games 100000 --output large.jsonl
        """,
    )

    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type to harvest (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum quality score 0-1 (default: 0.7)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=50000,
        help="Maximum games to harvest (default: 50000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/selfplay",
        help="Directory containing selfplay data (default: data/selfplay)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of games to sample per file (default: 1.0)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle file processing order",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON format for log files",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_script_logging(
        script_name="harvest_local",
        level=log_level,
        json_logs=args.json_logs,
    )

    # Validate arguments
    if not 0 <= args.min_quality <= 1:
        logger.error("min-quality must be between 0 and 1")
        return 1

    if args.max_games <= 0:
        logger.error("max-games must be positive")
        return 1

    if not 0 < args.sample_rate <= 1:
        logger.error("sample-rate must be between 0 and 1")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1

    output_file = Path(args.output)

    try:
        games_harvested, stats = harvest_local_data(
            data_dir=data_dir,
            board_type=args.board_type,
            num_players=args.num_players,
            min_quality=args.min_quality,
            max_games=args.max_games,
            output_file=output_file,
            sample_rate=args.sample_rate,
            shuffle_files=not args.no_shuffle,
        )

        if games_harvested == 0:
            logger.warning("No games harvested - check data directory and filters")
            return 1

        logger.info(f"Successfully harvested {games_harvested:,} games")
        logger.info(f"Average quality: {stats.avg_quality:.3f}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Harvest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

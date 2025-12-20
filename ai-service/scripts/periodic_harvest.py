#!/usr/bin/env python3
"""
Periodic Training Data Harvest

Automated periodic harvesting of high-quality training data.
Replaces periodic_harvest.sh with improved error handling and logging.

Usage:
    # Run as standalone
    python scripts/periodic_harvest.py

    # Via CLI
    python scripts/cli.py periodic --task harvest

Cron example (every 6 hours):
    0 */6 * * * cd ~/ringrift/ai-service && PYTHONPATH=. python scripts/periodic_harvest.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.logging_config import (
    setup_script_logging,
    get_logger,
    get_metrics_logger,
)

logger = get_logger(__name__)


@dataclass
class HarvestConfig:
    """Configuration for periodic harvest."""
    board_type: str = "square8"
    num_players: int = 2
    min_quality: float = 0.7
    max_games_per_harvest: int = 25000
    min_new_games_for_training: int = 10000
    data_dir: Path = Path("data/selfplay")
    harvest_dir: Path = Path("data/harvested")
    training_dir: Path = Path("data/training")
    log_dir: Path = Path("logs")

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"

    @property
    def accumulated_file(self) -> Path:
        return self.harvest_dir / f"accumulated_{self.config_key}.jsonl"


@dataclass
class HarvestResult:
    """Result of a periodic harvest run."""
    hostname: str
    timestamp: str
    config_key: str
    existing_games: int
    harvested_games: int
    new_unique_games: int
    total_accumulated: int
    training_triggered: bool
    training_pid: int | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PeriodicHarvester:
    """Manages periodic training data harvesting."""

    def __init__(self, config: HarvestConfig):
        """Initialize harvester.

        Args:
            config: Harvest configuration
        """
        self.config = config
        self.hostname = os.uname().nodename
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = get_metrics_logger("periodic_harvest", log_interval=120)

    def run(self, trigger_training: bool = True) -> HarvestResult:
        """Run the periodic harvest.

        Args:
            trigger_training: Whether to trigger training if enough new data

        Returns:
            HarvestResult with harvest details
        """
        logger.info(f"Starting periodic harvest on {self.hostname}")
        logger.info(f"Config: {self.config.config_key}")

        # Setup directories
        self._setup_directories()

        # Count existing games
        existing_games = self._count_accumulated_games()
        logger.info(f"Existing accumulated games: {existing_games:,}")

        # Harvest new games
        harvest_output = (
            self.config.harvest_dir /
            f"harvest_{self.hostname}_{self.timestamp}.jsonl"
        )

        try:
            harvested_count = self._run_harvest(harvest_output)
        except Exception as e:
            logger.error(f"Harvest failed: {e}")
            return HarvestResult(
                hostname=self.hostname,
                timestamp=self.timestamp,
                config_key=self.config.config_key,
                existing_games=existing_games,
                harvested_games=0,
                new_unique_games=0,
                total_accumulated=existing_games,
                training_triggered=False,
                training_pid=None,
                error=str(e),
            )

        logger.info(f"Harvested {harvested_count:,} games")

        # Merge with accumulated data (deduplicated)
        new_unique, total_accumulated = self._merge_accumulated(harvest_output)
        logger.info(
            f"Total accumulated: {total_accumulated:,} "
            f"(+{new_unique:,} new unique)"
        )

        # Cleanup harvest file
        if harvest_output.exists():
            harvest_output.unlink()

        # Check if we should trigger training
        training_pid = None
        training_triggered = False

        if trigger_training and new_unique >= self.config.min_new_games_for_training:
            logger.info(
                f"Sufficient new games ({new_unique:,} >= "
                f"{self.config.min_new_games_for_training:,}), triggering training"
            )
            training_pid = self._trigger_training()
            training_triggered = training_pid is not None
        else:
            logger.info(
                f"Not enough new games for training "
                f"({new_unique:,} < {self.config.min_new_games_for_training:,})"
            )

        self.metrics.set("total_accumulated", total_accumulated)
        self.metrics.set("new_unique_games", new_unique)
        self.metrics.flush()

        return HarvestResult(
            hostname=self.hostname,
            timestamp=self.timestamp,
            config_key=self.config.config_key,
            existing_games=existing_games,
            harvested_games=harvested_count,
            new_unique_games=new_unique,
            total_accumulated=total_accumulated,
            training_triggered=training_triggered,
            training_pid=training_pid,
            error=None,
        )

    def _setup_directories(self) -> None:
        """Create required directories."""
        self.config.harvest_dir.mkdir(parents=True, exist_ok=True)
        self.config.training_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def _count_accumulated_games(self) -> int:
        """Count games in accumulated file."""
        if not self.config.accumulated_file.exists():
            return 0

        try:
            with open(self.config.accumulated_file) as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.warning(f"Error counting accumulated games: {e}")
            return 0

    def _run_harvest(self, output_file: Path) -> int:
        """Run the harvest script.

        Args:
            output_file: Output JSONL file

        Returns:
            Number of games harvested
        """
        from scripts.harvest_local_training_data import harvest_local_data

        games, stats = harvest_local_data(
            data_dir=self.config.data_dir,
            board_type=self.config.board_type,
            num_players=self.config.num_players,
            min_quality=self.config.min_quality,
            max_games=self.config.max_games_per_harvest,
            output_file=output_file,
            sample_rate=1.0,
            shuffle_files=True,
        )

        self.metrics.set("harvest_quality_avg", stats.avg_quality)
        return games

    def _merge_accumulated(self, harvest_file: Path) -> tuple[int, int]:
        """Merge harvest with accumulated data, deduplicating.

        Args:
            harvest_file: Path to new harvest file

        Returns:
            Tuple of (new_unique_count, total_count)
        """
        seen_ids: set[str] = set()
        accumulated_file = self.config.accumulated_file
        temp_file = accumulated_file.with_suffix(".tmp")

        # Load existing game IDs
        if accumulated_file.exists():
            with open(accumulated_file) as f:
                for line in f:
                    try:
                        game = json.loads(line.strip())
                        game_id = game.get("game_id", str(hash(line))[:20])
                        seen_ids.add(game_id)
                    except json.JSONDecodeError:
                        pass

        existing_count = len(seen_ids)
        new_count = 0

        # Write deduplicated output
        with open(temp_file, "w") as out:
            # First, copy existing accumulated data
            if accumulated_file.exists():
                with open(accumulated_file) as f:
                    for line in f:
                        out.write(line if line.endswith("\n") else line + "\n")

            # Then add new unique games
            with open(harvest_file) as f:
                for line in f:
                    try:
                        game = json.loads(line.strip())
                        game_id = game.get("game_id", str(hash(line))[:20])

                        if game_id not in seen_ids:
                            seen_ids.add(game_id)
                            new_count += 1
                            out.write(line if line.endswith("\n") else line + "\n")

                    except json.JSONDecodeError:
                        pass

        # Replace accumulated file
        temp_file.rename(accumulated_file)

        total_count = existing_count + new_count
        return new_count, total_count

    def _trigger_training(self) -> int | None:
        """Trigger training with accumulated data.

        Returns:
            PID of training process if started, None otherwise
        """
        # First convert to NPZ
        npz_output = (
            self.config.training_dir /
            f"accumulated_{self.config.config_key}_{self.timestamp}.npz"
        )

        logger.info(f"Converting to NPZ: {npz_output}")

        try:
            # Run NPZ conversion
            convert_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "jsonl_to_npz.py"),
                    "--input", str(self.config.accumulated_file),
                    "--output", str(npz_output),
                    "--board-type", self.config.board_type,
                    "--num-players", str(self.config.num_players),
                ],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if convert_result.returncode != 0:
                logger.error(f"NPZ conversion failed: {convert_result.stderr}")
                return None

            if not npz_output.exists():
                logger.error("NPZ file not created")
                return None

            logger.info(f"NPZ conversion complete: {npz_output}")

        except subprocess.TimeoutExpired:
            logger.error("NPZ conversion timed out")
            return None
        except Exception as e:
            logger.error(f"NPZ conversion error: {e}")
            return None

        # Start training in background
        log_file = self.config.log_dir / f"training_{self.timestamp}.log"

        try:
            with open(log_file, "w") as log_out:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m", "app.training.train",
                        "--data", str(npz_output),
                        "--board-type", self.config.board_type,
                        "--num-players", str(self.config.num_players),
                    ],
                    stdout=log_out,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                )

            logger.info(f"Training started (PID: {process.pid})")
            logger.info(f"Training log: {log_file}")
            return process.pid

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return None


def run_periodic_harvest(
    board_type: str = "square8",
    num_players: int = 2,
    min_quality: float = 0.7,
    max_games: int = 25000,
    min_new_games_for_training: int = 10000,
    trigger_training: bool = True,
) -> int:
    """Run periodic harvest (callable from CLI).

    Args:
        board_type: Board type to harvest
        num_players: Number of players
        min_quality: Minimum quality score
        max_games: Max games per harvest
        min_new_games_for_training: Min new games to trigger training
        trigger_training: Whether to trigger training

    Returns:
        Exit code (0 = success)
    """
    config = HarvestConfig(
        board_type=board_type,
        num_players=num_players,
        min_quality=min_quality,
        max_games_per_harvest=max_games,
        min_new_games_for_training=min_new_games_for_training,
    )

    harvester = PeriodicHarvester(config)
    result = harvester.run(trigger_training=trigger_training)

    if result.error:
        logger.error(f"Harvest failed: {result.error}")
        return 1

    # Save result
    result_file = config.harvest_dir / f"result_{result.timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Periodic harvest complete. Result saved to {result_file}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run periodic training data harvest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default harvest
  %(prog)s --board-type hex8 --num-players 3  # Hex board, 3 players
  %(prog)s --no-training                      # Harvest only, no training
        """,
    )

    parser.add_argument(
        "--board-type",
        default="square8",
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum quality score (default: 0.7)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=25000,
        help="Maximum games per harvest (default: 25000)",
    )
    parser.add_argument(
        "--min-new-for-training",
        type=int,
        default=10000,
        help="Minimum new games to trigger training (default: 10000)",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Don't trigger training even if enough data",
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
        script_name="periodic_harvest",
        level=log_level,
        json_logs=args.json_logs,
    )

    return run_periodic_harvest(
        board_type=args.board_type,
        num_players=args.num_players,
        min_quality=args.min_quality,
        max_games=args.max_games,
        min_new_games_for_training=args.min_new_for_training,
        trigger_training=not args.no_training,
    )


if __name__ == "__main__":
    sys.exit(main())

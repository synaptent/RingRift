#!/usr/bin/env python3
"""Automated NPZ conversion for gumbel selfplay data.

Monitors JSONL files for each config and triggers NPZ conversion when
they reach the target game count (default: 200 games).

Usage:
    # Run once to check and convert ready configs
    python scripts/auto_convert_gumbel.py --check-once

    # Run continuously, checking every 5 minutes
    python scripts/auto_convert_gumbel.py --interval 300
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# All 12 configurations
CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hex8", 2),
    ("hex8", 3),
    ("hex8", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]

def count_games_in_jsonl(path: Path) -> int:
    """Count games in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def find_gumbel_files(base_dir: Path, board: str, players: int) -> list[Path]:
    """Find all gumbel JSONL files for a config."""
    patterns = [
        f"gumbel_{board}_{players}p*.jsonl",
        f"gumbel_{board[:3]}*{players}p*.jsonl",
        f"*gumbel*{board}*{players}p*.jsonl",
    ]
    files = []
    for pattern in patterns:
        files.extend(base_dir.glob(f"**/{pattern}"))
    return list(set(files))


def convert_to_npz(
    input_files: list[Path],
    output_path: Path,
    board: str,
    players: int,
) -> bool:
    """Run NPZ conversion."""
    if len(input_files) == 1:
        input_arg = str(input_files[0])
    else:
        # Create a temp combined file
        combined = output_path.parent / f"_combined_{board}_{players}p.jsonl"
        with open(combined, "w") as out:
            for f in input_files:
                with open(f) as inp:
                    out.write(inp.read())
        input_arg = str(combined)

    cmd = [
        sys.executable,
        "scripts/jsonl_to_npz.py",
        "--input", input_arg,
        "--output", str(output_path),
        "--board-type", board,
        "--num-players", str(players),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        return False

    logger.info(f"Conversion complete: {output_path}")
    return True


def check_and_convert(
    selfplay_dir: Path,
    output_dir: Path,
    target_games: int = 200,
    force: bool = False,
) -> dict:
    """Check all configs and convert those with enough games."""
    results = {}

    for board, players in CONFIGS:
        config_name = f"{board}_{players}p"
        output_path = output_dir / f"gumbel_{config_name}.npz"

        # Skip if already converted (unless force)
        if output_path.exists() and not force:
            logger.info(f"{config_name}: Already converted, skipping")
            results[config_name] = {"status": "exists", "games": 0}
            continue

        # Find JSONL files
        files = find_gumbel_files(selfplay_dir, board, players)
        total_games = sum(count_games_in_jsonl(f) for f in files)

        if total_games < target_games:
            logger.info(f"{config_name}: {total_games}/{target_games} games (waiting)")
            results[config_name] = {"status": "waiting", "games": total_games}
            continue

        logger.info(f"{config_name}: {total_games} games (converting...)")
        success = convert_to_npz(files, output_path, board, players)
        results[config_name] = {
            "status": "converted" if success else "failed",
            "games": total_games,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-convert gumbel selfplay to NPZ")
    parser.add_argument(
        "--selfplay-dir",
        type=Path,
        default=Path("data/selfplay"),
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training"),
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--target-games",
        type=int,
        default=200,
        help="Target games per config",
    )
    parser.add_argument(
        "--check-once",
        action="store_true",
        help="Check once and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 5 min)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if NPZ exists",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.check_once:
        results = check_and_convert(
            args.selfplay_dir,
            args.output_dir,
            args.target_games,
            args.force,
        )
        print("\n=== CONVERSION STATUS ===")
        for config, info in sorted(results.items()):
            print(f"{config}: {info['status']} ({info['games']} games)")
        return

    # Continuous monitoring
    logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
    while True:
        check_and_convert(
            args.selfplay_dir,
            args.output_dir,
            args.target_games,
            args.force,
        )
        logger.info(f"Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Filter and organize high-quality selfplay games by engine mode.

This script:
1. Filters games that have explicit engine_mode field (higher quality)
2. Organizes them into subdirectories by board_type/num_players/engine_mode
3. Provides statistics on the filtered data
4. Optionally removes low-quality GPU-generated JSONLs

Usage:
    # Dry run - show what would be done
    python scripts/filter_quality_games.py --input-dir /tmp/cluster_jsonl_analysis --dry-run

    # Filter to output directory
    python scripts/filter_quality_games.py --input-dir /tmp/cluster_jsonl_analysis --output-dir /tmp/quality_games

    # Also remove low-quality files
    python scripts/filter_quality_games.py --input-dir data/selfplay --output-dir data/quality --remove-low-quality
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def is_high_quality_game(game: dict[str, Any]) -> bool:
    """Check if a game is high quality (has explicit engine_mode)."""
    # Must have engine_mode
    if "engine_mode" not in game:
        return False

    # Must be a completed game (not eval pool)
    if game.get("game_status") == "active":
        return False
    if "move_history" in game and "moves" not in game:
        return False

    # Must have moves
    return not ("moves" not in game or len(game.get("moves", [])) == 0)


def get_board_type(game: dict[str, Any]) -> str:
    """Extract board type from game."""
    if "board_type" in game:
        return game["board_type"]
    if "config" in game and "board_type" in game["config"]:
        return game["config"]["board_type"]
    # Infer from board_size
    board_size = game.get("board_size", 8)
    if board_size <= 10:
        return "square8"
    elif board_size <= 20:
        return "square19"
    else:
        return f"square{board_size}"


def get_num_players(game: dict[str, Any]) -> int:
    """Extract number of players from game."""
    if "num_players" in game:
        return game["num_players"]
    if "config" in game and "num_players" in game["config"]:
        return game["config"]["num_players"]
    return 2


def get_termination_reason(game: dict[str, Any]) -> str:
    """Extract termination reason from game."""
    # New format
    if "termination_reason" in game:
        tr = game["termination_reason"]
        if tr.startswith("status:completed:"):
            return tr.split(":")[-1]
        return tr

    # Old format
    if "victory_type" in game:
        vt = game["victory_type"]
        # Normalize
        if vt == "ring_elimination":
            return "elimination"
        return vt

    return "unknown"


def process_files(
    input_dir: Path,
    output_dir: Path | None,
    remove_low_quality: bool,
    dry_run: bool
) -> dict[str, Any]:
    """Process all JSONL files and filter high-quality games."""

    stats = {
        "files_processed": 0,
        "files_with_quality": 0,
        "files_low_quality_only": 0,
        "total_games": 0,
        "high_quality_games": 0,
        "low_quality_games": 0,
        "by_engine_mode": Counter(),
        "by_board_type": Counter(),
        "by_config": Counter(),  # (board, players, engine)
        "termination_by_config": defaultdict(Counter),  # config -> termination -> count
    }

    # Collect games by output path
    output_games: dict[Path, list[dict]] = defaultdict(list)
    low_quality_files: list[Path] = []

    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        stats["files_processed"] += 1
        has_quality = False
        is_all_low_quality = True

        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        game = json.loads(line)
                        stats["total_games"] += 1

                        if is_high_quality_game(game):
                            has_quality = True
                            is_all_low_quality = False
                            stats["high_quality_games"] += 1

                            # Extract metadata
                            engine = game.get("engine_mode", "unknown")
                            board = get_board_type(game)
                            players = get_num_players(game)
                            termination = get_termination_reason(game)

                            stats["by_engine_mode"][engine] += 1
                            stats["by_board_type"][board] += 1

                            config_key = f"{board}_{players}p_{engine}"
                            stats["by_config"][config_key] += 1
                            stats["termination_by_config"][config_key][termination] += 1

                            # Determine output path
                            if output_dir:
                                out_path = output_dir / board / f"{players}p" / engine / "games.jsonl"
                                output_games[out_path].append(game)
                        else:
                            stats["low_quality_games"] += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}", file=sys.stderr)
            continue

        if has_quality:
            stats["files_with_quality"] += 1
        if is_all_low_quality:
            stats["files_low_quality_only"] += 1
            low_quality_files.append(jsonl_path)

    # Write output files
    if output_dir and not dry_run:
        for out_path, games in output_games.items():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                for game in games:
                    f.write(json.dumps(game) + '\n')
            print(f"  Wrote {len(games)} games to {out_path}")

    # Remove low-quality files
    if remove_low_quality and not dry_run:
        for lq_path in low_quality_files:
            print(f"  Removing low-quality file: {lq_path}")
            lq_path.unlink()
        stats["files_removed"] = len(low_quality_files)
    elif remove_low_quality and dry_run:
        stats["files_would_remove"] = len(low_quality_files)

    return stats


def print_report(stats: dict[str, Any]) -> None:
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("HIGH-QUALITY GAME FILTER REPORT")
    print("=" * 60)

    print("\n### File Summary")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files with quality games: {stats['files_with_quality']}")
    print(f"  Files with only low-quality: {stats['files_low_quality_only']}")

    print("\n### Game Summary")
    print(f"  Total games: {stats['total_games']:,}")
    print(f"  High-quality games: {stats['high_quality_games']:,} ({100*stats['high_quality_games']/stats['total_games']:.1f}%)")
    print(f"  Low-quality games: {stats['low_quality_games']:,} ({100*stats['low_quality_games']/stats['total_games']:.1f}%)")

    print("\n### By Engine Mode")
    for engine, count in stats["by_engine_mode"].most_common():
        pct = 100 * count / stats["high_quality_games"] if stats["high_quality_games"] else 0
        print(f"  {count:>8} ({pct:>5.1f}%)  {engine}")

    print("\n### By Board Type")
    for board, count in stats["by_board_type"].most_common():
        pct = 100 * count / stats["high_quality_games"] if stats["high_quality_games"] else 0
        print(f"  {count:>8} ({pct:>5.1f}%)  {board}")

    print("\n### By Configuration (Board/Players/Engine)")
    for config, count in stats["by_config"].most_common(20):
        print(f"  {count:>8}  {config}")

    print("\n### Termination Reason by Configuration")
    print("-" * 70)
    print(f"{'Config':<35} {'Territory':>10} {'Elimination':>12} {'LPS':>8} {'Other':>8}")
    print("-" * 70)

    for config in sorted(stats["termination_by_config"].keys()):
        term_counts = stats["termination_by_config"][config]
        total = sum(term_counts.values())
        territory = term_counts.get("territory", 0)
        elimination = term_counts.get("elimination", 0)
        lps = term_counts.get("lps", 0)
        other = total - territory - elimination - lps

        def pct(v, *, _total: int = total):
            return f"{100*v/_total:.0f}%" if _total else "-"

        print(f"{config:<35} {pct(territory):>10} {pct(elimination):>12} {pct(lps):>8} {pct(other):>8}")

    if "files_removed" in stats:
        print("\n### Cleanup")
        print(f"  Files removed: {stats['files_removed']}")
    elif "files_would_remove" in stats:
        print("\n### Cleanup (dry-run)")
        print(f"  Files would remove: {stats['files_would_remove']}")


def main():
    parser = argparse.ArgumentParser(description="Filter high-quality selfplay games")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Input directory")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory for filtered games")
    parser.add_argument("--remove-low-quality", action="store_true", help="Remove files with only low-quality games")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Processing {args.input_dir}...")
    if args.dry_run:
        print("(dry-run mode - no files will be written or removed)")

    stats = process_files(
        args.input_dir,
        args.output_dir,
        args.remove_low_quality,
        args.dry_run
    )

    print_report(stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())

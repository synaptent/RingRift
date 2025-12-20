#!/usr/bin/env python3
"""Quick CLI wrapper to convert local JSONL selfplay data to SQLite.

This is a convenience script that wraps chunked_jsonl_converter.py with
sensible defaults for common use cases. Designed for:
- Node startup (clear backlog)
- Manual batch processing
- Cron jobs

Usage:
    # Convert all local selfplay data (default)
    python scripts/convert_local_jsonl.py

    # Convert only square8 2-player games
    python scripts/convert_local_jsonl.py --board square8 --players 2

    # Fast parallel conversion with 4 workers
    python scripts/convert_local_jsonl.py --fast

    # Dry run to see what would be converted
    python scripts/convert_local_jsonl.py --dry-run

    # Force full reconversion
    python scripts/convert_local_jsonl.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Get paths
SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
CHUNKED_CONVERTER = SCRIPT_DIR / "chunked_jsonl_converter.py"


def main():
    parser = argparse.ArgumentParser(
        description="Convert local JSONL selfplay data to SQLite databases"
    )
    parser.add_argument(
        "--board", "-b",
        choices=["square8", "square19", "hexagonal"],
        help="Filter to specific board type",
    )
    parser.add_argument(
        "--players", "-p",
        type=int,
        choices=[2, 3, 4],
        help="Filter to specific player count",
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Use 4 workers for faster parallel processing",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=500,
        help="Records per batch (default: 500)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be converted without actually converting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconversion of all files (ignore marker)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(AI_SERVICE_ROOT / "data" / "selfplay"),
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(AI_SERVICE_ROOT / "data" / "games"),
        help="Output directory for SQLite databases",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Check converter exists
    if not CHUNKED_CONVERTER.exists():
        print(f"Error: Chunked converter not found at {CHUNKED_CONVERTER}")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable,
        str(CHUNKED_CONVERTER),
        "--input-dir", args.input_dir,
        "--output-dir", args.output_dir,
        "--chunk-size", str(args.chunk_size),
    ]

    # Worker count
    if args.fast:
        cmd.extend(["--workers", "4"])
    else:
        cmd.extend(["--workers", str(args.workers)])

    # Filters
    if args.board:
        cmd.extend(["--board-type", args.board])
    if args.players:
        cmd.extend(["--num-players", str(args.players)])

    # Flags
    if args.dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    if args.verbose:
        cmd.append("--verbose")

    # Run
    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SERVICE_ROOT),
            check=False,
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()

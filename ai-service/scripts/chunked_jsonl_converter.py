#!/usr/bin/env python3
"""Chunked JSONL to SQLite converter with parallel processing support.

This utility processes JSONL selfplay files in manageable chunks to avoid
memory issues with large files. It supports:
- Chunked reading (configurable lines per batch)
- Parallel file processing with configurable workers
- Progress tracking and resumption
- Graceful handling of corrupted/incomplete files
- Integration with P2P orchestrator marker files

Usage:
    # Process all selfplay JSONL files with defaults
    python scripts/chunked_jsonl_converter.py

    # Process with custom settings
    python scripts/chunked_jsonl_converter.py \
        --input-dir data/selfplay \
        --output-dir data/games \
        --chunk-size 1000 \
        --workers 4 \
        --board-type square8 \
        --num-players 2

    # Dry run to see what would be processed
    python scripts/chunked_jsonl_converter.py --dry-run

    # Force reprocess all files (ignore marker)
    python scripts/chunked_jsonl_converter.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections.abc import Iterator

# Ensure ai-service root on path for scripts/lib imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Unified logging and file format utilities
from scripts.lib.logging_config import setup_script_logging
from scripts.lib.file_formats import open_jsonl_file

logger = setup_script_logging("chunked_jsonl_converter")


@dataclass
class ConversionStats:
    """Track conversion statistics."""
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    games_added: int = 0
    games_skipped_duplicate: int = 0
    games_skipped_invalid: int = 0
    bytes_processed: int = 0
    start_time: float = field(default_factory=time.time)

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.games_added / elapsed if elapsed > 0 else 0
        return (
            f"Files: {self.files_processed} processed, {self.files_skipped} skipped, "
            f"{self.files_failed} failed | "
            f"Games: {self.games_added:,} added, {self.games_skipped_duplicate:,} duplicates | "
            f"Rate: {rate:.1f} games/sec | "
            f"Time: {elapsed:.1f}s"
        )


def detect_board_type(path: Path) -> tuple[str, int]:
    """Detect board type and player count from file path.

    Returns (board_type, num_players) tuple.
    """
    path_str = str(path).lower()

    # Detect board type
    if "hex" in path_str:
        board_type = "hexagonal"
    elif "square19" in path_str or "sq19" in path_str:
        board_type = "square19"
    else:
        board_type = "square8"

    # Detect player count
    if "4p" in path_str:
        num_players = 4
    elif "3p" in path_str:
        num_players = 3
    else:
        num_players = 2

    return board_type, num_players


def read_jsonl_chunked(
    filepath: Path,
    chunk_size: int = 1000,
) -> Iterator[list[dict[str, Any]]]:
    """Read JSONL file in chunks to avoid memory issues.

    Yields lists of parsed records, each list up to chunk_size items.
    Automatically detects and handles gzip-compressed files.
    """
    chunk: list[dict[str, Any]] = []

    try:
        with open_jsonl_file(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    record["_source_line"] = line_num
                    chunk.append(record)

                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []

                except json.JSONDecodeError:
                    # Skip malformed lines silently
                    continue

        # Yield remaining records
        if chunk:
            yield chunk

    except Exception as e:
        logger.warning(f"Error reading {filepath}: {e}")
        if chunk:
            yield chunk


def get_db_connection(
    db_path: Path,
    create_if_missing: bool = True,
) -> sqlite3.Connection:
    """Get SQLite connection with proper schema."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    if create_if_missing:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                winner INTEGER,
                move_count INTEGER,
                game_status TEXT,
                victory_type TEXT,
                created_at TEXT,
                source TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_board_type ON games(board_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_status ON games(game_status)")
        conn.commit()

    return conn


def process_chunk(
    records: list[dict[str, Any]],
    conn: sqlite3.Connection,
    source_file: str,
    file_stem: str,
) -> tuple[int, int, int]:
    """Process a chunk of records and insert into database.

    Returns (added, duplicates, invalid) counts.
    """
    added = 0
    duplicates = 0
    invalid = 0

    for record in records:
        try:
            # Skip incomplete games
            status = record.get("status", record.get("game_status", ""))
            if status != "completed":
                invalid += 1
                continue

            # Generate unique game ID
            orig_id = record.get("game_id", record.get("_source_line", 0))
            game_id = f"{file_stem}_{orig_id}"

            # Extract fields
            board_type = record.get("board_type", "square8")
            num_players = record.get("num_players", 2)
            winner = record.get("winner", 0)
            move_count = record.get("move_count", len(record.get("moves", [])))
            victory_type = record.get("victory_type", "unknown")
            timestamp = record.get("timestamp", record.get("created_at", ""))

            # Insert with conflict handling
            cursor = conn.execute("""
                INSERT OR IGNORE INTO games
                (game_id, board_type, num_players, winner, move_count,
                 game_status, victory_type, created_at, source, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                board_type,
                num_players,
                winner,
                move_count,
                "completed",
                victory_type,
                timestamp,
                f"jsonl:{source_file}",
                json.dumps(record),
            ))

            if cursor.rowcount > 0:
                added += 1
            else:
                duplicates += 1

        except Exception:
            invalid += 1

    conn.commit()
    return added, duplicates, invalid


def process_file(
    filepath: Path,
    output_dir: Path,
    chunk_size: int,
    board_type_filter: str | None = None,
    num_players_filter: int | None = None,
) -> tuple[str, int, int, int]:
    """Process a single JSONL file.

    Returns (filepath, added, duplicates, invalid).
    """
    # Detect board type from path
    detected_board, detected_players = detect_board_type(filepath)

    # Apply filters
    if board_type_filter and detected_board != board_type_filter:
        return str(filepath), 0, 0, 0
    if num_players_filter and detected_players != num_players_filter:
        return str(filepath), 0, 0, 0

    # Determine output database
    db_name = f"jsonl_converted_{detected_board}_{detected_players}p.db"
    db_path = output_dir / db_name

    # Process file in chunks
    total_added = 0
    total_duplicates = 0
    total_invalid = 0

    conn = get_db_connection(db_path)
    file_stem = filepath.stem
    source_file = filepath.name

    try:
        for chunk in read_jsonl_chunked(filepath, chunk_size):
            added, dups, invalid = process_chunk(chunk, conn, source_file, file_stem)
            total_added += added
            total_duplicates += dups
            total_invalid += invalid
    finally:
        conn.close()

    return str(filepath), total_added, total_duplicates, total_invalid


def scan_jsonl_files(
    input_dir: Path,
    converted_files: set[str],
    min_size_bytes: int = 100,
) -> list[Path]:
    """Scan for unconverted JSONL files."""
    files = []

    for jsonl_file in input_dir.rglob("*.jsonl"):
        try:
            # Skip small/empty files
            if jsonl_file.stat().st_size < min_size_bytes:
                continue

            # Skip already converted
            rel_path = str(jsonl_file.relative_to(input_dir.parent))
            if rel_path in converted_files:
                continue

            files.append(jsonl_file)

        except Exception:
            continue

    return sorted(files, key=lambda f: f.stat().st_size, reverse=True)


def load_marker_file(marker_path: Path) -> set[str]:
    """Load set of already-converted file paths."""
    if not marker_path.exists():
        return set()

    try:
        content = marker_path.read_text().strip()
        return set(content.split("\n")) if content else set()
    except Exception:
        return set()


def save_marker_file(marker_path: Path, converted: set[str]) -> None:
    """Save converted file paths to marker."""
    try:
        marker_path.write_text("\n".join(sorted(converted)))
    except Exception as e:
        logger.warning(f"Failed to save marker file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Chunked JSONL to SQLite converter with parallel processing"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/selfplay",
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/games",
        help="Output directory for SQLite databases",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of records to process per batch (default: 500)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel file processors (default: 2)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal"],
        help="Filter to specific board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter to specific player count",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all files (ignore marker)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    # Handle both relative and absolute paths
    if args.input_dir.startswith("/"):
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(__file__).parent.parent / args.input_dir

    if args.output_dir.startswith("/"):
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / args.output_dir

    marker_path = input_dir.parent / ".jsonl_converted"

    logger.info("=" * 60)
    logger.info("Chunked JSONL Converter")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Workers: {args.workers}")
    if args.board_type:
        logger.info(f"Board type filter: {args.board_type}")
    if args.num_players:
        logger.info(f"Player count filter: {args.num_players}")
    logger.info("")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load marker file
    converted_files = set() if args.force else load_marker_file(marker_path)
    logger.info(f"Already converted: {len(converted_files)} files")

    # Scan for files to process
    files_to_process = scan_jsonl_files(input_dir, converted_files)
    total_size = sum(f.stat().st_size for f in files_to_process)

    logger.info(f"Files to process: {len(files_to_process)}")
    logger.info(f"Total size: {total_size / (1024*1024):.1f} MB")
    logger.info("")

    if not files_to_process:
        logger.info("No files to process")
        return 0

    if args.dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for f in files_to_process[:20]:
            board, players = detect_board_type(f)
            logger.info(f"  {f.name} ({f.stat().st_size/1024:.1f}KB) -> {board}_{players}p")
        if len(files_to_process) > 20:
            logger.info(f"  ... and {len(files_to_process) - 20} more files")
        return 0

    # Process files
    stats = ConversionStats()
    newly_converted: set[str] = set()

    logger.info("Starting conversion...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_file,
                f,
                output_dir,
                args.chunk_size,
                args.board_type,
                args.num_players,
            ): f
            for f in files_to_process
        }

        for future in as_completed(futures):
            filepath = futures[future]
            try:
                path_str, added, dups, invalid = future.result()

                stats.files_processed += 1
                stats.games_added += added
                stats.games_skipped_duplicate += dups
                stats.games_skipped_invalid += invalid
                stats.bytes_processed += filepath.stat().st_size

                # Track converted file
                rel_path = str(filepath.relative_to(input_dir.parent))
                newly_converted.add(rel_path)

                # Progress logging
                if stats.files_processed % 50 == 0:
                    logger.info(f"Progress: {stats.summary()}")

            except Exception as e:
                stats.files_failed += 1
                logger.warning(f"Failed to process {filepath}: {e}")

    # Save marker file
    all_converted = converted_files | newly_converted
    save_marker_file(marker_path, all_converted)

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(stats.summary())
    logger.info(f"Marker file updated: {marker_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

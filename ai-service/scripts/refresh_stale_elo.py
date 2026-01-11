#!/usr/bin/env python3
"""
Refresh stale Elo entries using GPU Gumbel MCTS on cluster.

This script identifies Elo entries that haven't been updated within a threshold
period and queues them for re-evaluation using GPU Gumbel MCTS with 800+ simulations.

Usage:
    # Dry run - see what would be queued
    python scripts/refresh_stale_elo.py --age-days 30 --dry-run

    # Execute - actually queue for re-evaluation
    python scripts/refresh_stale_elo.py --age-days 30 --execute

    # Queue all entries older than 7 days (default threshold)
    python scripts/refresh_stale_elo.py --age-days 7 --execute

Environment:
    Run this script on the cluster (mac-studio or GH200 node) where:
    - The unified_elo.db database exists
    - The evaluation queue can be accessed
    - Model files are available for validation
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional


def get_db_path() -> Path:
    """Get path to unified Elo database."""
    # Try multiple locations (ordered by preference)
    candidates = [
        # Standard locations
        Path("data/elo/unified_elo.db"),
        Path("data/games/unified_elo.db"),
        Path("~/RingRift/ai-service/data/elo/unified_elo.db").expanduser(),
        Path("/home/ubuntu/RingRift/ai-service/data/elo/unified_elo.db"),
        # Mac Studio external storage
        Path("/Volumes/RingRift-Data/ai-service/data/games/unified_elo.db"),
        Path("/Volumes/RingRift-Data/ai-service/data/elo/unified_elo.db"),
        # Nebius backup (fallback if main is empty)
        Path("/Volumes/RingRift-Data/nebius_backup_20260108/ringrift-h100-3/data/unified_elo.db"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]  # Return default for error message


def find_stale_entries(conn: sqlite3.Connection, age_days: int, max_entries: int) -> list:
    """Find Elo entries older than the specified threshold."""
    c = conn.cursor()
    cutoff = time.time() - (age_days * 86400)

    c.execute("""
        SELECT participant_id, board_type, num_players, rating, last_update,
               (? - last_update) / 86400 as age_days
        FROM elo_ratings
        WHERE archived_at IS NULL
          AND last_update IS NOT NULL
          AND last_update < ?
        ORDER BY last_update ASC
        LIMIT ?
    """, (time.time(), cutoff, max_entries))

    return c.fetchall()


def count_harness_metadata(conn: sqlite3.Connection) -> dict:
    """Count match entries by harness_type presence."""
    c = conn.cursor()
    c.execute("""
        SELECT
            CASE WHEN harness_type IS NULL THEN 'missing' ELSE 'present' END as status,
            COUNT(*) as count
        FROM match_history
        GROUP BY status
    """)
    return dict(c.fetchall())


def get_model_path_for_participant(participant_id: str, board_type: str, num_players: int) -> Optional[str]:
    """Extract or construct model path from participant_id."""
    # Format 1: Composite ID "nn:gumbel_mcts:hex8_2p:v5"
    if participant_id.startswith("nn:"):
        parts = participant_id.split(":")
        if len(parts) >= 3:
            config = parts[2]  # e.g., "hex8_2p"
            return f"models/canonical_{config}.pth"

    # Format 2: Direct model name "hex8_2p_v5.pth" or "canonical_hex8_2p.pth"
    if participant_id.endswith(".pth"):
        return f"models/{participant_id}"

    # Format 3: Config-based reconstruction
    config_key = f"{board_type}_{num_players}p"
    return f"models/canonical_{config_key}.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Refresh stale Elo entries using GPU Gumbel MCTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview stale entries older than 30 days
    python scripts/refresh_stale_elo.py --age-days 30 --dry-run

    # Queue all entries older than 7 days for re-evaluation
    python scripts/refresh_stale_elo.py --age-days 7 --execute

    # Show harness metadata statistics
    python scripts/refresh_stale_elo.py --stats-only
        """
    )
    parser.add_argument("--age-days", type=int, default=7,
                        help="Re-evaluate entries older than N days (default: 7)")
    parser.add_argument("--max-entries", type=int, default=50,
                        help="Maximum entries to queue per run (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be queued without executing")
    parser.add_argument("--execute", action="store_true",
                        help="Actually queue the re-evaluations")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only show database statistics, don't queue anything")
    parser.add_argument("--priority-boost", type=int, default=0,
                        help="Add priority boost to all queued entries (0-50)")
    args = parser.parse_args()

    # Validate arguments
    if not args.stats_only and not args.dry_run and not args.execute:
        print("Error: Must specify --dry-run, --execute, or --stats-only")
        print("Run with --help for usage information")
        sys.exit(1)

    # Find database
    db_path = get_db_path()
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Make sure you're running this on the cluster where the database exists")
        sys.exit(1)

    print(f"Using database: {db_path}")
    conn = sqlite3.connect(db_path)

    # Show statistics
    harness_stats = count_harness_metadata(conn)
    print(f"\nHarness metadata status:")
    for status, count in harness_stats.items():
        print(f"  harness_type {status}: {count} matches")

    if args.stats_only:
        # Also show stale entry preview
        stale_entries = find_stale_entries(conn, args.age_days, 10)
        print(f"\nOldest entries (would be re-evaluated with --age-days {args.age_days}):")
        for entry in stale_entries:
            print(f"  {entry[0]}: {entry[3]:.0f} Elo, {entry[5]:.1f} days old "
                  f"({entry[1]}/{entry[2]}p)")
        conn.close()
        return

    # Find stale entries
    stale_entries = find_stale_entries(conn, args.age_days, args.max_entries)

    if not stale_entries:
        print(f"\nNo entries older than {args.age_days} days found")
        conn.close()
        return

    print(f"\nFound {len(stale_entries)} entries older than {args.age_days} days:")
    for entry in stale_entries:
        print(f"  {entry[0]}: {entry[3]:.0f} Elo, {entry[5]:.1f} days old "
              f"({entry[1]}/{entry[2]}p)")

    if args.dry_run:
        print("\n[DRY RUN] Would queue these for re-evaluation with GPU Gumbel MCTS")
        print("Run with --execute to actually queue them")
        conn.close()
        return

    if args.execute:
        try:
            from app.coordination.evaluation_queue import get_evaluation_queue
        except ImportError as e:
            print(f"\nError: Could not import evaluation_queue: {e}")
            print("Make sure you're running from the ai-service directory")
            conn.close()
            sys.exit(1)

        queue = get_evaluation_queue()
        queued = 0
        skipped = 0

        for entry in stale_entries:
            participant_id, board_type, num_players, rating, _, age_days = entry

            # Get model path
            model_path = get_model_path_for_participant(participant_id, board_type, num_players)

            if model_path and Path(model_path).exists():
                # Priority based on age (older = higher priority), capped at 100
                priority = min(100, int(age_days) + args.priority_boost)

                try:
                    request_id = queue.add_request(
                        model_path=model_path,
                        board_type=board_type,
                        num_players=num_players,
                        priority=priority,
                        source="stale_refresh_script",
                        harness_type="gumbel_mcts"
                    )
                    if request_id:
                        queued += 1
                        print(f"Queued: {participant_id} -> {model_path} (priority={priority})")
                    else:
                        skipped += 1
                        print(f"Skipped (already queued or duplicate): {participant_id}")
                except Exception as e:
                    skipped += 1
                    print(f"Error queueing {participant_id}: {e}")
            else:
                skipped += 1
                print(f"Skipped (model not found): {participant_id} -> {model_path}")

        print(f"\n{'='*50}")
        print(f"Queued {queued} models for GPU Gumbel MCTS re-evaluation")
        print(f"Skipped {skipped} entries")

        # Show queue status
        try:
            status = queue.get_queue_status()
            print(f"\nEvaluation queue status:")
            print(f"  Pending: {status.get('pending', 'N/A')}")
            print(f"  Running: {status.get('running', 'N/A')}")
            print(f"  Completed: {status.get('completed', 'N/A')}")
        except Exception as e:
            print(f"\nCould not get queue status: {e}")

    conn.close()


if __name__ == "__main__":
    main()

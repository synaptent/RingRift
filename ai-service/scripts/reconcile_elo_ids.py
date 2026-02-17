#!/usr/bin/env python3
"""One-time migration: reconcile canonical_* participant IDs to ringrift_best_*.

Migrates all `canonical_*` participant IDs in unified_elo.db to use the stable
`ringrift_best_*` prefix. This ensures Elo ratings accumulate under a single
participant ID across model promotions.

Operations:
1. Scan elo_ratings for participant_id LIKE 'canonical_%'
2. Compute normalized ID via normalize_nn_id()
3. If target ringrift_best_* entry doesn't exist: RENAME (UPDATE participant_id)
4. If target exists for same (board_type, num_players): MERGE
   - Combine games/wins/losses/draws
   - Weighted-average rating by games_played
   - MAX peak_rating
   - DELETE old entry
5. Update match_history.participant_ids JSON field
6. Add bidirectional participant_aliases

Usage:
    python3 scripts/reconcile_elo_ids.py --dry-run    # Preview changes
    python3 scripts/reconcile_elo_ids.py               # Execute migration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.composite_participant import normalize_nn_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Get path to unified_elo.db."""
    return Path(__file__).resolve().parent.parent / "data" / "unified_elo.db"


def get_column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    """Get column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def get_id_column(columns: set[str]) -> str:
    """Determine the participant ID column name."""
    return "model_id" if "model_id" in columns else "participant_id"


def reconcile_elo_ratings(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Reconcile canonical_* entries in elo_ratings table.

    Returns number of entries reconciled.
    """
    columns = get_column_names(conn, "elo_ratings")
    id_col = get_id_column(columns)

    # Find all canonical_* entries
    cursor = conn.execute(f"""
        SELECT {id_col}, board_type, num_players, rating, games_played,
               wins, losses, draws,
               {'peak_rating' if 'peak_rating' in columns else 'rating'} as peak_rating
        FROM elo_ratings
        WHERE {id_col} LIKE 'canonical_%'
        ORDER BY {id_col}
    """)
    canonical_entries = cursor.fetchall()

    if not canonical_entries:
        logger.info("No canonical_* entries found in elo_ratings - nothing to reconcile")
        return 0

    logger.info(f"Found {len(canonical_entries)} canonical_* entries to reconcile")
    reconciled = 0

    for entry in canonical_entries:
        old_id = entry[0]
        board_type = entry[1]
        num_players = entry[2]
        rating = entry[3]
        games = entry[4]
        wins = entry[5]
        losses = entry[6]
        draws = entry[7]
        peak = entry[8]

        # Compute normalized ID
        # For composite IDs like "canonical_hex8_2p:gumbel_mcts:d2",
        # normalize only the nn_id part
        if ":" in old_id:
            parts = old_id.split(":")
            normalized_nn = normalize_nn_id(parts[0])
            new_id = ":".join([normalized_nn] + parts[1:])
        else:
            new_id = normalize_nn_id(old_id)

        if new_id == old_id:
            continue  # No normalization needed

        # Check if target already exists
        cursor = conn.execute(f"""
            SELECT {id_col}, rating, games_played, wins, losses, draws,
                   {'peak_rating' if 'peak_rating' in columns else 'rating'} as peak_rating
            FROM elo_ratings
            WHERE {id_col} = ? AND board_type = ? AND num_players = ?
        """, (new_id, board_type, num_players))
        existing = cursor.fetchone()

        if existing is None:
            # Simple rename
            logger.info(f"  RENAME: {old_id} -> {new_id} "
                        f"(board={board_type}, players={num_players}, "
                        f"rating={rating:.1f}, games={games})")
            if not dry_run:
                conn.execute(f"""
                    UPDATE elo_ratings SET {id_col} = ?
                    WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                """, (new_id, old_id, board_type, num_players))
        else:
            # Merge: combine stats
            ex_rating = existing[1]
            ex_games = existing[2]
            ex_wins = existing[3]
            ex_losses = existing[4]
            ex_draws = existing[5]
            ex_peak = existing[6]

            total_games = games + ex_games
            # Weighted average rating
            if total_games > 0:
                merged_rating = (rating * games + ex_rating * ex_games) / total_games
            else:
                merged_rating = max(rating, ex_rating)
            merged_wins = wins + ex_wins
            merged_losses = losses + ex_losses
            merged_draws = draws + ex_draws
            merged_peak = max(peak, ex_peak)

            logger.info(
                f"  MERGE: {old_id} -> {new_id} "
                f"(board={board_type}, players={num_players}, "
                f"old_rating={rating:.1f}@{games}g + existing={ex_rating:.1f}@{ex_games}g "
                f"-> merged={merged_rating:.1f}@{total_games}g)"
            )
            if not dry_run:
                # Update the target entry with merged stats
                update_parts = [
                    f"rating = ?",
                    f"games_played = ?",
                    f"wins = ?",
                    f"losses = ?",
                    f"draws = ?",
                ]
                update_values = [merged_rating, total_games, merged_wins, merged_losses, merged_draws]

                if "peak_rating" in columns:
                    update_parts.append("peak_rating = ?")
                    update_values.append(merged_peak)

                update_values.extend([new_id, board_type, num_players])
                conn.execute(f"""
                    UPDATE elo_ratings SET {', '.join(update_parts)}
                    WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                """, update_values)

                # Delete the old canonical entry
                conn.execute(f"""
                    DELETE FROM elo_ratings
                    WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                """, (old_id, board_type, num_players))

        reconciled += 1

    return reconciled


def reconcile_match_history(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Update match_history participant_ids JSON to use ringrift_best_* prefix.

    Returns number of match records updated.
    """
    # Check if match_history table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='match_history'"
    )
    if not cursor.fetchone():
        logger.info("No match_history table found - skipping")
        return 0

    columns = get_column_names(conn, "match_history")

    # Check for participant_ids column (JSON)
    if "participant_ids" not in columns:
        logger.info("match_history has no participant_ids column - skipping")
        return 0

    cursor = conn.execute("""
        SELECT rowid, participant_ids FROM match_history
        WHERE participant_ids LIKE '%canonical_%'
    """)
    matches = cursor.fetchall()

    if not matches:
        logger.info("No match_history entries with canonical_* IDs")
        return 0

    logger.info(f"Found {len(matches)} match_history entries with canonical_* IDs")
    updated = 0

    for rowid, pid_json in matches:
        try:
            pids = json.loads(pid_json)
        except (json.JSONDecodeError, TypeError):
            continue

        new_pids = []
        changed = False
        for pid in pids:
            if isinstance(pid, str) and "canonical_" in pid:
                if ":" in pid:
                    parts = pid.split(":")
                    normalized_nn = normalize_nn_id(parts[0])
                    new_pid = ":".join([normalized_nn] + parts[1:])
                else:
                    new_pid = normalize_nn_id(pid)
                new_pids.append(new_pid)
                if new_pid != pid:
                    changed = True
            else:
                new_pids.append(pid)

        if changed:
            logger.info(f"  UPDATE match_history rowid={rowid}: {pids} -> {new_pids}")
            if not dry_run:
                conn.execute(
                    "UPDATE match_history SET participant_ids = ? WHERE rowid = ?",
                    (json.dumps(new_pids), rowid),
                )
            updated += 1

    return updated


def add_participant_aliases(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Add bidirectional aliases for canonical_* <-> ringrift_best_* IDs.

    Returns number of aliases added.
    """
    # Check if participant_aliases table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='participant_aliases'"
    )
    if not cursor.fetchone():
        # Create the table
        if not dry_run:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS participant_aliases (
                    alias_from TEXT NOT NULL,
                    alias_to TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (alias_from, alias_to)
                )
            """)
        logger.info("Created participant_aliases table")

    columns = get_column_names(conn, "elo_ratings")
    id_col = get_id_column(columns)

    # Get all ringrift_best_* entries to create aliases
    cursor = conn.execute(f"""
        SELECT DISTINCT {id_col} FROM elo_ratings
        WHERE {id_col} LIKE 'ringrift_best_%'
    """)
    entries = cursor.fetchall()

    added = 0
    now = time.time()
    for (pid,) in entries:
        # Compute the canonical_ equivalent
        if ":" in pid:
            parts = pid.split(":")
            nn_part = parts[0]
            if nn_part.startswith("ringrift_best_"):
                canonical_nn = "canonical_" + nn_part[len("ringrift_best_"):]
                canonical_id = ":".join([canonical_nn] + parts[1:])
            else:
                continue
        else:
            if pid.startswith("ringrift_best_"):
                canonical_id = "canonical_" + pid[len("ringrift_best_"):]
            else:
                continue

        if not dry_run:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO participant_aliases (alias_from, alias_to, created_at) "
                    "VALUES (?, ?, ?)",
                    (canonical_id, pid, now),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO participant_aliases (alias_from, alias_to, created_at) "
                    "VALUES (?, ?, ?)",
                    (pid, canonical_id, now),
                )
            except sqlite3.OperationalError:
                pass  # Table might not exist in dry-run
        added += 1

    if added:
        logger.info(f"Added {added} bidirectional alias pairs")
    return added


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile canonical_* participant IDs to ringrift_best_* in unified_elo.db"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying database"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to unified_elo.db (default: data/unified_elo.db)"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_db_path()

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "EXECUTE"
    logger.info(f"{'=' * 60}")
    logger.info(f"ELO PARTICIPANT ID RECONCILIATION [{mode}]")
    logger.info(f"{'=' * 60}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Database size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("")

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        # Step 1: Reconcile elo_ratings
        logger.info("--- Step 1: Reconcile elo_ratings ---")
        ratings_count = reconcile_elo_ratings(conn, args.dry_run)

        # Step 2: Update match_history
        logger.info("")
        logger.info("--- Step 2: Update match_history ---")
        history_count = reconcile_match_history(conn, args.dry_run)

        # Step 3: Add participant aliases
        logger.info("")
        logger.info("--- Step 3: Add participant aliases ---")
        alias_count = add_participant_aliases(conn, args.dry_run)

        if not args.dry_run:
            conn.commit()
            logger.info("")
            logger.info("Changes committed to database.")

        # Summary
        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"SUMMARY [{mode}]")
        logger.info(f"{'=' * 60}")
        logger.info(f"  elo_ratings reconciled: {ratings_count}")
        logger.info(f"  match_history updated:  {history_count}")
        logger.info(f"  aliases added:          {alias_count}")

        # Verify no canonical_ left
        columns = get_column_names(conn, "elo_ratings")
        id_col = get_id_column(columns)
        cursor = conn.execute(f"""
            SELECT COUNT(*) FROM elo_ratings WHERE {id_col} LIKE 'canonical_%'
        """)
        remaining = cursor.fetchone()[0]
        if remaining > 0 and not args.dry_run:
            logger.warning(f"  WARNING: {remaining} canonical_* entries still remain!")
        elif args.dry_run:
            logger.info(f"  canonical_* entries remaining (pre-migration): {remaining}")
        else:
            logger.info(f"  canonical_* entries remaining: 0 (all migrated)")

    except Exception as e:
        logger.error(f"Error during reconciliation: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()

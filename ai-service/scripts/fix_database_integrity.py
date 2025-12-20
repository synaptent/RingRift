#!/usr/bin/env python3
"""Fix database integrity issues identified in 2025-12-16 audit.

This script addresses:
1. Duplicate matches in ELO database (match_history table)
2. Missing unique indices for deduplication
3. Games with inconsistent status (active but completed)
4. Recalculates games_played/wins/losses from actual match history

Usage:
    python scripts/fix_database_integrity.py --check    # Check for issues
    python scripts/fix_database_integrity.py --fix      # Fix all issues
    python scripts/fix_database_integrity.py --fix-elo  # Fix ELO database only
    python scripts/fix_database_integrity.py --fix-games # Fix game databases only
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Pinned baselines (anchors for ELO calibration)
PINNED_BASELINES = {
    "baseline_random": 400.0,  # Random player pinned at 400 ELO
}

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

DATA_DIR = AI_SERVICE_ROOT / "data"
UNIFIED_ELO_DB = DATA_DIR / "unified_elo.db"

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("fix_database_integrity")

def check_elo_duplicates(db_path: Path) -> dict:
    """Check for duplicate matches in ELO database."""
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    results = {}

    # Check duplicate game_ids (excluding NULL)
    cur.execute("""
        SELECT game_id, COUNT(*) as count
        FROM match_history
        WHERE game_id IS NOT NULL
        GROUP BY game_id
        HAVING COUNT(*) > 1
    """)
    dup_game_ids = cur.fetchall()
    results["duplicate_game_ids"] = len(dup_game_ids)

    # Check duplicate composite keys
    cur.execute("""
        SELECT participant_a, participant_b, timestamp, COUNT(*) as count
        FROM match_history
        GROUP BY participant_a, participant_b, timestamp
        HAVING COUNT(*) > 1
    """)
    dup_composite = cur.fetchall()
    results["duplicate_composite_keys"] = len(dup_composite)

    # Count total duplicates
    cur.execute("""
        SELECT SUM(count - 1) FROM (
            SELECT COUNT(*) as count
            FROM match_history
            GROUP BY participant_a, participant_b, timestamp
            HAVING COUNT(*) > 1
        )
    """)
    row = cur.fetchone()
    results["total_duplicate_matches"] = row[0] if row[0] else 0

    # Check game_id coverage
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN game_id IS NOT NULL THEN 1 ELSE 0 END) as with_id
        FROM match_history
    """)
    row = cur.fetchone()
    results["total_matches"] = row[0]
    results["matches_with_game_id"] = row[1]
    results["game_id_coverage_pct"] = round(100.0 * row[1] / row[0], 2) if row[0] > 0 else 0

    # Check win/loss conservation
    cur.execute("""
        SELECT
            board_type, num_players,
            SUM(wins) as total_wins,
            SUM(losses) as total_losses,
            ABS(SUM(wins) - SUM(losses)) as imbalance
        FROM elo_ratings
        GROUP BY board_type, num_players
    """)
    imbalances = cur.fetchall()
    results["win_loss_imbalances"] = [
        {"config": f"{r[0]}_{r[1]}p", "wins": r[2], "losses": r[3], "imbalance": r[4]}
        for r in imbalances if r[4] > 10  # Only significant imbalances
    ]

    conn.close()
    return results


def fix_elo_duplicates(db_path: Path, dry_run: bool = False) -> dict:
    """Remove duplicate matches from ELO database."""
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    results = {"removed": 0, "indices_added": 0}

    if dry_run:
        logger.info("DRY RUN - No changes will be made")

    # Step 1: Count duplicates before
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT 1 FROM match_history
            GROUP BY participant_a, participant_b, timestamp
            HAVING COUNT(*) > 1
        )
    """)
    before_dup_groups = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM match_history")
    before_total = cur.fetchone()[0]

    logger.info(f"Before: {before_total} total matches, {before_dup_groups} duplicate groups")

    if not dry_run:
        # Step 2: Remove duplicates, keeping the one with lowest id (oldest)
        logger.info("Removing duplicate matches...")
        cur.execute("""
            DELETE FROM match_history
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM match_history
                GROUP BY participant_a, participant_b, board_type, num_players, timestamp
            )
        """)
        removed = cur.rowcount
        results["removed"] = removed
        logger.info(f"Removed {removed} duplicate matches")

        # Step 3: Add unique index on game_id (if not exists)
        try:
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_match_game_id
                ON match_history(game_id) WHERE game_id IS NOT NULL
            """)
            results["indices_added"] += 1
            logger.info("Added unique index on game_id")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add game_id index: {e}")

        # Step 4: Add deduplication index
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_match_dedupe_key
                ON match_history(participant_a, participant_b, board_type, num_players, timestamp)
            """)
            results["indices_added"] += 1
            logger.info("Added deduplication index")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add dedupe index: {e}")

        conn.commit()

        # Verify
        cur.execute("SELECT COUNT(*) FROM match_history")
        after_total = cur.fetchone()[0]
        logger.info(f"After: {after_total} total matches")

    conn.close()
    return results


def recalculate_stats_from_history(db_path: Path, dry_run: bool = False) -> dict:
    """Recalculate games_played, wins, losses from actual match history."""
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    results = {"updated": 0, "discrepancies": []}

    # Get all participants and their actual match counts
    logger.info("Calculating actual stats from match history...")
    cur.execute("""
        WITH match_stats AS (
            SELECT
                participant_a as participant_id,
                board_type,
                num_players,
                COUNT(*) as games,
                SUM(CASE WHEN winner = participant_a THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN winner = participant_b THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN winner IS NULL OR (winner != participant_a AND winner != participant_b) THEN 1 ELSE 0 END) as draws
            FROM match_history
            GROUP BY participant_a, board_type, num_players

            UNION ALL

            SELECT
                participant_b as participant_id,
                board_type,
                num_players,
                COUNT(*) as games,
                SUM(CASE WHEN winner = participant_b THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN winner = participant_a THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN winner IS NULL OR (winner != participant_a AND winner != participant_b) THEN 1 ELSE 0 END) as draws
            FROM match_history
            GROUP BY participant_b, board_type, num_players
        )
        SELECT
            participant_id, board_type, num_players,
            SUM(games) as total_games,
            SUM(wins) as total_wins,
            SUM(losses) as total_losses,
            SUM(draws) as total_draws
        FROM match_stats
        GROUP BY participant_id, board_type, num_players
    """)
    actual_stats = cur.fetchall()

    logger.info(f"Found stats for {len(actual_stats)} participant/config combinations")

    # Compare with current elo_ratings and find discrepancies
    discrepancies = []
    for row in actual_stats:
        pid, bt, np, games, wins, losses, draws = row

        cur.execute("""
            SELECT games_played, wins, losses, draws
            FROM elo_ratings
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (pid, bt, np))
        current = cur.fetchone()

        if current and (current[0] != games or current[1] != wins or
            current[2] != losses or current[3] != draws):
            discrepancies.append({
                "participant_id": pid,
                "config": f"{bt}_{np}p",
                "current": {"games": current[0], "wins": current[1], "losses": current[2], "draws": current[3]},
                "actual": {"games": games, "wins": wins, "losses": losses, "draws": draws},
            })

    results["discrepancies"] = discrepancies[:50]  # Limit output
    results["total_discrepancies"] = len(discrepancies)

    if discrepancies and not dry_run:
        logger.info(f"Fixing {len(discrepancies)} stat discrepancies...")
        for row in actual_stats:
            pid, bt, np, games, wins, losses, draws = row
            cur.execute("""
                UPDATE elo_ratings
                SET games_played = ?, wins = ?, losses = ?, draws = ?
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (games, wins, losses, draws, pid, bt, np))

        # Re-pin baseline ratings after recalculation
        for prefix, pinned_rating in PINNED_BASELINES.items():
            cur.execute("""
                UPDATE elo_ratings
                SET rating = ?
                WHERE participant_id LIKE ?
            """, (pinned_rating, f"{prefix}%"))
            if cur.rowcount > 0:
                logger.info(f"Re-pinned {cur.rowcount} {prefix} ratings to {pinned_rating}")

        conn.commit()
        results["updated"] = len(actual_stats)
        logger.info(f"Updated {results['updated']} rating records")

    conn.close()
    return results


def check_game_databases(data_dir: Path) -> dict:
    """Check game databases for integrity issues."""
    results = {"databases": [], "total_inconsistent": 0}

    # Find all game databases
    db_files = list(data_dir.glob("*.db")) + list(data_dir.glob("games/**/*.db"))

    for db_path in db_files:
        if "elo" in db_path.name.lower() or "coordination" in str(db_path):
            continue  # Skip ELO and coordination databases

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # Check if this is a game database
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            if not cur.fetchone():
                conn.close()
                continue

            # Check for games with inconsistent status
            cur.execute("""
                SELECT COUNT(*) FROM games
                WHERE game_status = 'active'
                AND (termination_reason IS NOT NULL AND termination_reason != '')
            """)
            inconsistent = cur.fetchone()[0]

            if inconsistent > 0:
                results["databases"].append({
                    "path": str(db_path),
                    "inconsistent_games": inconsistent,
                })
                results["total_inconsistent"] += inconsistent

            conn.close()
        except sqlite3.Error:
            continue  # Skip non-SQLite files

    return results


def fix_game_status(data_dir: Path, dry_run: bool = False) -> dict:
    """Fix games with inconsistent status."""
    results = {"databases_fixed": 0, "games_fixed": 0}

    # Find all game databases
    db_files = list(data_dir.glob("*.db")) + list(data_dir.glob("games/**/*.db"))

    for db_path in db_files:
        if "elo" in db_path.name.lower() or "coordination" in str(db_path):
            continue

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # Check if this is a game database
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            if not cur.fetchone():
                conn.close()
                continue

            if not dry_run:
                # Fix games that have termination_reason but status='active'
                cur.execute("""
                    UPDATE games
                    SET game_status = 'completed'
                    WHERE game_status = 'active'
                    AND termination_reason IS NOT NULL
                    AND termination_reason != ''
                """)
                fixed = cur.rowcount

                if fixed > 0:
                    conn.commit()
                    results["databases_fixed"] += 1
                    results["games_fixed"] += fixed
                    logger.info(f"Fixed {fixed} games in {db_path.name}")

            conn.close()
        except sqlite3.Error as e:
            logger.warning(f"Error processing {db_path}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fix database integrity issues from 2025-12-16 audit"
    )
    parser.add_argument("--check", action="store_true", help="Check for issues only")
    parser.add_argument("--fix", action="store_true", help="Fix all issues")
    parser.add_argument("--fix-elo", action="store_true", help="Fix ELO database only")
    parser.add_argument("--fix-games", action="store_true", help="Fix game databases only")
    parser.add_argument("--recalc-stats", action="store_true", help="Recalculate stats from match history")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--db", type=str, help="Path to specific database")

    args = parser.parse_args()

    if not any([args.check, args.fix, args.fix_elo, args.fix_games, args.recalc_stats]):
        parser.print_help()
        return

    db_path = Path(args.db) if args.db else UNIFIED_ELO_DB

    if args.check:
        print("\n" + "=" * 60)
        print("DATABASE INTEGRITY CHECK")
        print("=" * 60)

        # Check ELO database
        print("\n--- ELO Database ---")
        elo_results = check_elo_duplicates(db_path)
        print(f"Total matches: {elo_results.get('total_matches', 'N/A')}")
        print(f"Matches with game_id: {elo_results.get('matches_with_game_id', 'N/A')} ({elo_results.get('game_id_coverage_pct', 0)}%)")
        print(f"Duplicate composite keys: {elo_results.get('duplicate_composite_keys', 0)}")
        print(f"Total duplicate matches to remove: {elo_results.get('total_duplicate_matches', 0)}")

        if elo_results.get("win_loss_imbalances"):
            print("\nWin/Loss imbalances (>10):")
            for imb in elo_results["win_loss_imbalances"]:
                print(f"  {imb['config']}: wins={imb['wins']}, losses={imb['losses']}, imbalance={imb['imbalance']}")

        # Check game databases
        print("\n--- Game Databases ---")
        game_results = check_game_databases(DATA_DIR)
        print(f"Databases with inconsistent games: {len(game_results['databases'])}")
        print(f"Total inconsistent games: {game_results['total_inconsistent']}")

        for db in game_results["databases"][:10]:
            print(f"  {db['path']}: {db['inconsistent_games']} inconsistent")

    if args.fix or args.fix_elo:
        print("\n" + "=" * 60)
        print("FIXING ELO DATABASE")
        print("=" * 60)

        results = fix_elo_duplicates(db_path, dry_run=args.dry_run)
        print(f"Duplicates removed: {results.get('removed', 0)}")
        print(f"Indices added: {results.get('indices_added', 0)}")

    if args.fix or args.fix_games:
        print("\n" + "=" * 60)
        print("FIXING GAME DATABASES")
        print("=" * 60)

        results = fix_game_status(DATA_DIR, dry_run=args.dry_run)
        print(f"Databases fixed: {results['databases_fixed']}")
        print(f"Games fixed: {results['games_fixed']}")

    if args.recalc_stats:
        print("\n" + "=" * 60)
        print("RECALCULATING STATS FROM MATCH HISTORY")
        print("=" * 60)

        results = recalculate_stats_from_history(db_path, dry_run=args.dry_run)
        print(f"Total discrepancies found: {results.get('total_discrepancies', 0)}")
        print(f"Records updated: {results.get('updated', 0)}")

        if results.get("discrepancies"):
            print("\nSample discrepancies:")
            for d in results["discrepancies"][:10]:
                print(f"  {d['participant_id']} ({d['config']})")
                print(f"    Current: games={d['current']['games']}, wins={d['current']['wins']}, losses={d['current']['losses']}")
                print(f"    Actual:  games={d['actual']['games']}, wins={d['actual']['wins']}, losses={d['actual']['losses']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fix ELO database integrity issues.

This script addresses:
1. Duplicate matches (same game recorded multiple times)
2. Win/loss imbalance (ratings not recalculated after merge)
3. Missing game_ids

Usage:
    python scripts/fix_elo_database.py --analyze          # Show issues without fixing
    python scripts/fix_elo_database.py --deduplicate      # Remove duplicate matches
    python scripts/fix_elo_database.py --recalculate      # Recalculate all ratings
    python scripts/fix_elo_database.py --fix-all          # Do everything
"""

import argparse
import hashlib
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

DEFAULT_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# ELO calculation constants
INITIAL_RATING = 1500.0
K_FACTOR = 32.0

# Pinned baselines (anchors)
PINNED_BASELINES = {
    "baseline_random": 400.0,
}


def get_pinned_rating(participant_id: str) -> Optional[float]:
    """Check if participant is a pinned baseline."""
    for prefix, rating in PINNED_BASELINES.items():
        if participant_id.startswith(prefix):
            return rating
    return None


def analyze_database(db_path: Path) -> Dict:
    """Analyze database for integrity issues."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    issues = {
        "total_matches": 0,
        "null_game_ids": 0,
        "duplicate_matches": 0,
        "duplicate_groups": [],
        "win_loss_imbalance": 0,
        "zero_loss_high_win_models": [],
    }

    # Count total matches
    cur.execute("SELECT COUNT(*) FROM match_history")
    issues["total_matches"] = cur.fetchone()[0]

    # Count null game_ids
    cur.execute("SELECT COUNT(*) FROM match_history WHERE game_id IS NULL OR game_id = ''")
    issues["null_game_ids"] = cur.fetchone()[0]

    # Find duplicate matches (same participants, winner, within 1 second)
    cur.execute("""
        SELECT participant_a, participant_b, winner, board_type, num_players,
               COUNT(*) as cnt, MIN(timestamp), MAX(timestamp)
        FROM match_history
        GROUP BY participant_a, participant_b, winner, board_type, num_players
        HAVING cnt > 10
        ORDER BY cnt DESC
        LIMIT 20
    """)

    for row in cur.fetchall():
        issues["duplicate_groups"].append({
            "participant_a": row[0],
            "participant_b": row[1],
            "winner": row[2],
            "board_type": row[3],
            "num_players": row[4],
            "count": row[5],
        })
        issues["duplicate_matches"] += row[5] - 1  # All but one are duplicates

    # Check win/loss balance for 2-player games
    cur.execute("""
        SELECT SUM(wins), SUM(losses)
        FROM elo_ratings
        WHERE num_players = 2
    """)
    row = cur.fetchone()
    if row[0] and row[1]:
        issues["win_loss_imbalance"] = abs(row[0] - row[1])

    # Find models with 0 losses but many wins
    cur.execute("""
        SELECT participant_id, wins, losses, games_played
        FROM elo_ratings
        WHERE losses = 0 AND wins > 100 AND archived_at IS NULL
        ORDER BY wins DESC
        LIMIT 10
    """)
    issues["zero_loss_high_win_models"] = [
        {"id": r[0], "wins": r[1], "losses": r[2], "games": r[3]}
        for r in cur.fetchall()
    ]

    conn.close()
    return issues


def deduplicate_matches(db_path: Path, dry_run: bool = False) -> int:
    """Remove duplicate matches, keeping one representative per unique game.

    Uses a signature of (participant_a, participant_b, winner, board_type, num_players,
    rounded_timestamp) to identify duplicates.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Analyzing matches for deduplication...")

    # Get all matches
    cur.execute("""
        SELECT id, participant_a, participant_b, winner, board_type, num_players,
               timestamp, game_id
        FROM match_history
        ORDER BY timestamp
    """)
    matches = cur.fetchall()

    # Group by signature
    signatures = defaultdict(list)
    for match in matches:
        match_id, p_a, p_b, winner, board, players, ts, game_id = match
        # Round timestamp to nearest second for grouping
        rounded_ts = int(ts) if ts else 0
        sig = f"{p_a}|{p_b}|{winner}|{board}|{players}|{rounded_ts}"
        signatures[sig].append(match_id)

    # Find IDs to delete (keep first of each group)
    ids_to_delete = []
    for sig, match_ids in signatures.items():
        if len(match_ids) > 1:
            # Keep the first one, delete the rest
            ids_to_delete.extend(match_ids[1:])

    print(f"Found {len(ids_to_delete)} duplicate matches to remove")

    if dry_run:
        print("DRY RUN - no changes made")
        conn.close()
        return len(ids_to_delete)

    # Delete duplicates in batches
    if ids_to_delete:
        batch_size = 1000
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i+batch_size]
            placeholders = ",".join("?" * len(batch))
            cur.execute(f"DELETE FROM match_history WHERE id IN ({placeholders})", batch)
            print(f"Deleted batch {i//batch_size + 1}: {len(batch)} matches")

        conn.commit()

    conn.close()
    return len(ids_to_delete)


def add_game_ids(db_path: Path, dry_run: bool = False) -> int:
    """Add unique game_ids to matches that don't have one."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find matches without game_id
    cur.execute("""
        SELECT id, participant_a, participant_b, timestamp
        FROM match_history
        WHERE game_id IS NULL OR game_id = ''
    """)
    matches = cur.fetchall()

    print(f"Found {len(matches)} matches without game_id")

    if dry_run:
        print("DRY RUN - no changes made")
        conn.close()
        return len(matches)

    # Generate unique game_ids based on content hash
    updated = 0
    for match_id, p_a, p_b, ts in matches:
        # Create deterministic game_id from match content
        content = f"{p_a}|{p_b}|{ts}|{match_id}"
        game_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        cur.execute(
            "UPDATE match_history SET game_id = ? WHERE id = ?",
            (game_id, match_id)
        )
        updated += 1

        if updated % 10000 == 0:
            conn.commit()
            print(f"Updated {updated} matches...")

    conn.commit()
    conn.close()
    return updated


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def recalculate_ratings(db_path: Path, dry_run: bool = False) -> Dict:
    """Recalculate all ELO ratings from match history.

    This replays all matches chronologically to rebuild accurate ratings.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Loading match history...")

    # Get all matches ordered by timestamp
    cur.execute("""
        SELECT participant_a, participant_b, winner, board_type, num_players, timestamp
        FROM match_history
        WHERE winner IS NOT NULL
        ORDER BY timestamp
    """)
    matches = cur.fetchall()

    print(f"Replaying {len(matches)} matches to recalculate ratings...")

    # Initialize ratings storage: (board_type, num_players, participant_id) -> rating_data
    ratings = defaultdict(lambda: {
        "rating": INITIAL_RATING,
        "games_played": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    })

    # Replay all matches
    for i, (p_a, p_b, winner, board_type, num_players, ts) in enumerate(matches):
        key_a = (board_type, num_players, p_a)
        key_b = (board_type, num_players, p_b)

        r_a = ratings[key_a]
        r_b = ratings[key_b]

        # Calculate expected scores
        exp_a = expected_score(r_a["rating"], r_b["rating"])
        exp_b = 1.0 - exp_a

        # Determine actual scores
        if winner == p_a:
            score_a, score_b = 1.0, 0.0
            r_a["wins"] += 1
            r_b["losses"] += 1
        elif winner == p_b:
            score_a, score_b = 0.0, 1.0
            r_a["losses"] += 1
            r_b["wins"] += 1
        elif winner == "draw":
            score_a, score_b = 0.5, 0.5
            r_a["draws"] += 1
            r_b["draws"] += 1
        else:
            # Unknown winner format, skip
            continue

        # Update ratings (unless pinned)
        pinned_a = get_pinned_rating(p_a)
        pinned_b = get_pinned_rating(p_b)

        if pinned_a is None:
            r_a["rating"] += K_FACTOR * (score_a - exp_a)
        else:
            r_a["rating"] = pinned_a

        if pinned_b is None:
            r_b["rating"] += K_FACTOR * (score_b - exp_b)
        else:
            r_b["rating"] = pinned_b

        r_a["games_played"] += 1
        r_b["games_played"] += 1

        if (i + 1) % 50000 == 0:
            print(f"Processed {i + 1} matches...")

    print(f"Recalculated ratings for {len(ratings)} participant/config combinations")

    # Verify win/loss conservation for 2-player games
    total_wins_2p = sum(r["wins"] for (bt, np, _), r in ratings.items() if np == 2)
    total_losses_2p = sum(r["losses"] for (bt, np, _), r in ratings.items() if np == 2)

    print(f"2-player games: {total_wins_2p} wins, {total_losses_2p} losses")
    if total_wins_2p != total_losses_2p:
        print(f"WARNING: Win/loss imbalance of {abs(total_wins_2p - total_losses_2p)}")
    else:
        print("Win/loss conservation VERIFIED")

    if dry_run:
        print("DRY RUN - no changes made")
        conn.close()
        return {"participants": len(ratings), "matches": len(matches)}

    # Clear existing ratings
    print("Clearing existing ratings...")
    cur.execute("DELETE FROM elo_ratings")

    # Insert recalculated ratings
    print("Inserting recalculated ratings...")
    now = time.time()

    for (board_type, num_players, participant_id), data in ratings.items():
        cur.execute("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played,
             wins, losses, draws, rating_deviation, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            participant_id, board_type, num_players,
            data["rating"], data["games_played"],
            data["wins"], data["losses"], data["draws"],
            350.0,  # Initial rating deviation
            now,
        ))

    conn.commit()
    conn.close()

    return {"participants": len(ratings), "matches": len(matches)}


def verify_integrity(db_path: Path) -> bool:
    """Verify database integrity after fixes."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("\n=== INTEGRITY CHECK ===")

    # Check 1: Win/loss conservation for 2-player games
    cur.execute("""
        SELECT SUM(wins), SUM(losses)
        FROM elo_ratings
        WHERE num_players = 2
    """)
    row = cur.fetchone()
    wins_2p = row[0] or 0
    losses_2p = row[1] or 0

    print(f"2-player wins: {wins_2p}, losses: {losses_2p}")
    if wins_2p != losses_2p:
        print(f"FAIL: Win/loss imbalance of {abs(wins_2p - losses_2p)}")
        return False
    print("PASS: Win/loss conservation")

    # Check 2: No duplicate game_ids
    cur.execute("""
        SELECT game_id, COUNT(*) as cnt
        FROM match_history
        WHERE game_id IS NOT NULL AND game_id != ''
        GROUP BY game_id
        HAVING cnt > 1
        LIMIT 5
    """)
    dups = cur.fetchall()
    if dups:
        print(f"FAIL: Found {len(dups)} duplicate game_ids")
        return False
    print("PASS: No duplicate game_ids")

    # Check 3: All matches have game_ids
    cur.execute("""
        SELECT COUNT(*) FROM match_history
        WHERE game_id IS NULL OR game_id = ''
    """)
    null_ids = cur.fetchone()[0]
    if null_ids > 0:
        print(f"WARN: {null_ids} matches still without game_id")
    else:
        print("PASS: All matches have game_ids")

    # Check 4: Random baselines pinned at 400
    cur.execute("""
        SELECT participant_id, rating
        FROM elo_ratings
        WHERE participant_id LIKE 'baseline_random%'
        AND rating != 400
    """)
    unpinned = cur.fetchall()
    if unpinned:
        print(f"WARN: {len(unpinned)} random baselines not at 400 ELO")
    else:
        print("PASS: Random baselines pinned at 400")

    conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix ELO database integrity issues")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="Path to ELO database")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze database without making changes")
    parser.add_argument("--deduplicate", action="store_true",
                        help="Remove duplicate matches")
    parser.add_argument("--add-game-ids", action="store_true",
                        help="Add game_ids to matches without them")
    parser.add_argument("--recalculate", action="store_true",
                        help="Recalculate all ratings from match history")
    parser.add_argument("--verify", action="store_true",
                        help="Verify database integrity")
    parser.add_argument("--fix-all", action="store_true",
                        help="Apply all fixes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")

    args = parser.parse_args()
    db_path = Path(args.db)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    # Backup before any changes
    if (args.deduplicate or args.recalculate or args.fix_all) and not args.dry_run:
        backup_path = db_path.with_suffix(f".db.backup_{int(time.time())}")
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy(db_path, backup_path)

    if args.analyze or args.fix_all:
        print("\n=== DATABASE ANALYSIS ===")
        issues = analyze_database(db_path)
        print(f"Total matches: {issues['total_matches']}")
        print(f"Null game_ids: {issues['null_game_ids']}")
        print(f"Estimated duplicates: {issues['duplicate_matches']}")
        print(f"Win/loss imbalance (2p): {issues['win_loss_imbalance']}")
        print(f"Zero-loss high-win models: {len(issues['zero_loss_high_win_models'])}")

        if issues['duplicate_groups']:
            print("\nTop duplicate groups:")
            for g in issues['duplicate_groups'][:5]:
                print(f"  {g['participant_a']} vs {g['participant_b']}: {g['count']} matches")

    if args.deduplicate or args.fix_all:
        print("\n=== DEDUPLICATION ===")
        removed = deduplicate_matches(db_path, dry_run=args.dry_run)
        print(f"Removed {removed} duplicate matches")

    if args.add_game_ids or args.fix_all:
        print("\n=== ADDING GAME IDS ===")
        added = add_game_ids(db_path, dry_run=args.dry_run)
        print(f"Added game_ids to {added} matches")

    if args.recalculate or args.fix_all:
        print("\n=== RECALCULATING RATINGS ===")
        result = recalculate_ratings(db_path, dry_run=args.dry_run)
        print(f"Recalculated {result['participants']} ratings from {result['matches']} matches")

    if args.verify or args.fix_all:
        success = verify_integrity(db_path)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

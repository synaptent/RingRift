#!/usr/bin/env python3
"""Filter out timeout games from selfplay databases to improve training data quality.

Timeout games (games that hit move limit without decisive victory) contaminate
training data because they don't represent proper game outcomes.

This script:
1. Identifies timeout games in selfplay databases
2. Marks them as excluded from training OR deletes them
3. Reports statistics on filtered games
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime


def get_timeout_games(conn: sqlite3.Connection) -> list[tuple]:
    """Get all timeout game IDs."""
    cursor = conn.cursor()
    
    # Games with explicit timeout status
    cursor.execute("""
        SELECT game_id, board_type, num_players, num_moves, termination_reason
        FROM games 
        WHERE termination_reason LIKE '%timeout%'
           OR termination_reason LIKE '%move_limit%'
           OR termination_reason LIKE '%max_moves%'
    """)
    explicit_timeouts = cursor.fetchall()
    
    # Games with NULL termination but high move counts (likely timeouts)
    # Default move limits: square8=200, square19=400, hexagonal=300
    cursor.execute("""
        SELECT game_id, board_type, num_players, num_moves, termination_reason
        FROM games 
        WHERE termination_reason IS NULL
          AND (
            (board_type = 'square8' AND num_moves >= 195)
            OR (board_type = 'square19' AND num_moves >= 395)
            OR (board_type = 'hexagonal' AND num_moves >= 295)
          )
    """)
    implicit_timeouts = cursor.fetchall()
    
    return explicit_timeouts + implicit_timeouts


def filter_games(db_path: Path, dry_run: bool = True, delete: bool = False) -> dict:
    """Filter timeout games from a database.
    
    Args:
        db_path: Path to the SQLite database
        dry_run: If True, only report what would be done
        delete: If True, delete games; otherwise mark as excluded
        
    Returns:
        Statistics about filtered games
    """
    stats = {
        'db': str(db_path),
        'total_games': 0,
        'timeout_games': 0,
        'filtered': 0,
        'by_config': {},
    }
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return stats
    
    conn = sqlite3.connect(db_path)
    
    # Get total games
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM games")
    stats['total_games'] = cursor.fetchone()[0]
    
    # Get timeout games
    timeout_games = get_timeout_games(conn)
    stats['timeout_games'] = len(timeout_games)
    
    # Group by config
    for game_id, board_type, num_players, num_moves, reason in timeout_games:
        config = f"{board_type}_{num_players}p"
        if config not in stats['by_config']:
            stats['by_config'][config] = 0
        stats['by_config'][config] += 1
    
    if dry_run:
        print(f"\n[DRY RUN] Would filter {stats['timeout_games']} timeout games from {db_path.name}")
        print(f"  Total games: {stats['total_games']}")
        print(f"  Timeout games: {stats['timeout_games']} ({100*stats['timeout_games']/max(1,stats['total_games']):.1f}%)")
        for config, count in sorted(stats['by_config'].items(), key=lambda x: -x[1]):
            print(f"    {config}: {count}")
    else:
        game_ids = [g[0] for g in timeout_games]
        
        if delete:
            # Delete timeout games and their moves
            for i in range(0, len(game_ids), 500):
                batch = game_ids[i:i+500]
                placeholders = ','.join(['?'] * len(batch))
                cursor.execute(f"DELETE FROM moves WHERE game_id IN ({placeholders})", batch)
                cursor.execute(f"DELETE FROM games WHERE game_id IN ({placeholders})", batch)
            conn.commit()
            stats['filtered'] = len(game_ids)
            print(f"Deleted {len(game_ids)} timeout games from {db_path.name}")
        else:
            # Check if excluded column exists, add if not
            cursor.execute("PRAGMA table_info(games)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'excluded_from_training' not in columns:
                cursor.execute("ALTER TABLE games ADD COLUMN excluded_from_training INTEGER DEFAULT 0")
            
            # Mark as excluded
            for i in range(0, len(game_ids), 500):
                batch = game_ids[i:i+500]
                placeholders = ','.join(['?'] * len(batch))
                cursor.execute(f"UPDATE games SET excluded_from_training = 1 WHERE game_id IN ({placeholders})", batch)
            conn.commit()
            stats['filtered'] = len(game_ids)
            print(f"Marked {len(game_ids)} timeout games as excluded in {db_path.name}")
    
    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description='Filter timeout games from training data')
    parser.add_argument('--db', type=str, help='Specific database to filter')
    parser.add_argument('--data-dir', type=str, default='data/games', help='Directory containing databases')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Only report, do not modify')
    parser.add_argument('--apply', action='store_true', help='Actually apply the filter')
    parser.add_argument('--delete', action='store_true', help='Delete games instead of marking excluded')
    args = parser.parse_args()
    
    dry_run = not args.apply
    
    if args.db:
        db_paths = [Path(args.db)]
    else:
        data_dir = Path(args.data_dir)
        db_paths = list(data_dir.glob('*.db'))
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Filtering timeout games from {len(db_paths)} databases")
    print(f"Mode: {'DELETE' if args.delete else 'MARK EXCLUDED'}")
    print("=" * 60)
    
    total_stats = {
        'total_games': 0,
        'timeout_games': 0,
        'filtered': 0,
    }
    
    for db_path in sorted(db_paths):
        if 'holdout' in str(db_path) or 'tournament' in str(db_path):
            continue  # Skip holdout/tournament DBs
        stats = filter_games(db_path, dry_run=dry_run, delete=args.delete)
        total_stats['total_games'] += stats['total_games']
        total_stats['timeout_games'] += stats['timeout_games']
        total_stats['filtered'] += stats['filtered']
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_stats['timeout_games']} timeout games out of {total_stats['total_games']}")
    print(f"  Percentage: {100*total_stats['timeout_games']/max(1,total_stats['total_games']):.1f}%")
    if not dry_run:
        print(f"  Filtered: {total_stats['filtered']}")


if __name__ == '__main__':
    main()

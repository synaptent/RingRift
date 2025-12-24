#!/usr/bin/env python3
"""
Audit and cleanup game databases across the cluster.

Checks for games without recorded moves and either:
1. Filters them out (if <90% are empty)
2. Deletes the database (if >=90% are empty)

Usage:
    python scripts/audit_and_cleanup_databases.py --scan  # Just scan, no changes
    python scripts/audit_and_cleanup_databases.py --cleanup  # Actually cleanup
    python scripts/audit_and_cleanup_databases.py --cleanup --dry-run  # Show what would be done
"""

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class DatabaseAudit:
    """Result of auditing a single database."""
    path: str
    total_games: int
    games_with_moves: int
    total_moves: int
    games_without_moves: int
    percent_empty: float
    sources: dict[str, int]
    size_bytes: int
    recommendation: str  # 'keep', 'filter', 'delete'
    error: Optional[str] = None


def audit_database(db_path: str) -> DatabaseAudit:
    """Audit a single database for games without moves."""
    try:
        if not os.path.exists(db_path):
            return DatabaseAudit(
                path=db_path,
                total_games=0,
                games_with_moves=0,
                total_moves=0,
                games_without_moves=0,
                percent_empty=100.0,
                sources={},
                size_bytes=0,
                recommendation='delete',
                error='File not found'
            )

        size_bytes = os.path.getsize(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not cursor.fetchone():
            conn.close()
            return DatabaseAudit(
                path=db_path,
                total_games=0,
                games_with_moves=0,
                total_moves=0,
                games_without_moves=0,
                percent_empty=100.0,
                sources={},
                size_bytes=size_bytes,
                recommendation='delete',
                error='No games table'
            )

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
        has_moves_table = cursor.fetchone() is not None

        # Count total games
        cursor.execute("SELECT COUNT(*) FROM games")
        total_games = cursor.fetchone()[0]

        if total_games == 0:
            conn.close()
            return DatabaseAudit(
                path=db_path,
                total_games=0,
                games_with_moves=0,
                total_moves=0,
                games_without_moves=0,
                percent_empty=100.0,
                sources={},
                size_bytes=size_bytes,
                recommendation='delete',
                error='Empty database'
            )

        # Count moves
        if has_moves_table:
            cursor.execute("SELECT COUNT(*) FROM game_moves")
            total_moves = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM game_moves")
            games_with_moves = cursor.fetchone()[0]
        else:
            total_moves = 0
            games_with_moves = 0

        games_without_moves = total_games - games_with_moves
        percent_empty = (games_without_moves / total_games * 100) if total_games > 0 else 100.0

        # Get sources breakdown
        sources = {}
        try:
            cursor.execute("SELECT source, COUNT(*) FROM games GROUP BY source")
            for row in cursor.fetchall():
                sources[row[0] or 'unknown'] = row[1]
        except sqlite3.OperationalError:
            sources = {'unknown': total_games}

        conn.close()

        # Determine recommendation
        if percent_empty >= 90:
            recommendation = 'delete'
        elif percent_empty > 0:
            recommendation = 'filter'
        else:
            recommendation = 'keep'

        return DatabaseAudit(
            path=db_path,
            total_games=total_games,
            games_with_moves=games_with_moves,
            total_moves=total_moves,
            games_without_moves=games_without_moves,
            percent_empty=percent_empty,
            sources=sources,
            size_bytes=size_bytes,
            recommendation=recommendation
        )

    except Exception as e:
        return DatabaseAudit(
            path=db_path,
            total_games=0,
            games_with_moves=0,
            total_moves=0,
            games_without_moves=0,
            percent_empty=100.0,
            sources={},
            size_bytes=0,
            recommendation='skip',
            error=str(e)
        )


def cleanup_database(db_path: str, dry_run: bool = True) -> tuple[str, int]:
    """
    Clean up a database by removing games without moves.
    Returns (action_taken, games_removed).
    """
    audit = audit_database(db_path)

    if audit.error:
        return f'error: {audit.error}', 0

    if audit.recommendation == 'keep':
        return 'keep', 0

    if audit.recommendation == 'delete':
        if dry_run:
            return f'would_delete ({audit.total_games} games, {audit.percent_empty:.1f}% empty)', 0
        else:
            # Create backup name
            backup_path = db_path + f'.deleted_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.rename(db_path, backup_path)
            return f'deleted (backup: {os.path.basename(backup_path)})', audit.total_games

    if audit.recommendation == 'filter':
        if dry_run:
            return f'would_filter ({audit.games_without_moves} games)', 0
        else:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Delete games without moves
            cursor.execute("""
                DELETE FROM games
                WHERE game_id NOT IN (SELECT DISTINCT game_id FROM game_moves)
            """)
            removed = cursor.rowcount

            # Also delete any orphaned data
            try:
                cursor.execute("""
                    DELETE FROM game_states
                    WHERE game_id NOT IN (SELECT game_id FROM games)
                """)
            except sqlite3.OperationalError:
                pass  # Table might not exist

            conn.commit()
            cursor.execute("VACUUM")
            conn.close()

            return f'filtered ({removed} games removed)', removed

    return 'unknown', 0


def find_databases(root_path: str) -> list[str]:
    """Find all .db files under root_path."""
    databases = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.db'):
                databases.append(os.path.join(dirpath, filename))
    return sorted(databases)


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def main():
    parser = argparse.ArgumentParser(description='Audit and cleanup game databases')
    parser.add_argument('--scan', action='store_true', help='Just scan databases, no changes')
    parser.add_argument('--cleanup', action='store_true', help='Actually perform cleanup')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--path', type=str, default='data/games', help='Path to scan for databases')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if not args.scan and not args.cleanup:
        args.scan = True  # Default to scan

    databases = find_databases(args.path)

    if not databases:
        print(f"No databases found in {args.path}")
        return 1

    print(f"Found {len(databases)} databases in {args.path}")
    print()

    results = []
    total_games = 0
    total_with_moves = 0
    total_without_moves = 0
    total_size = 0

    to_delete = []
    to_filter = []

    for db_path in databases:
        audit = audit_database(db_path)
        results.append(audit)

        total_games += audit.total_games
        total_with_moves += audit.games_with_moves
        total_without_moves += audit.games_without_moves
        total_size += audit.size_bytes

        if audit.recommendation == 'delete':
            to_delete.append(audit)
        elif audit.recommendation == 'filter':
            to_filter.append(audit)

        if args.verbose or audit.recommendation != 'keep':
            status = '🗑️' if audit.recommendation == 'delete' else ('🧹' if audit.recommendation == 'filter' else '✅')
            print(f"{status} {os.path.basename(audit.path)}")
            print(f"   Games: {audit.games_with_moves}/{audit.total_games} with moves ({audit.percent_empty:.1f}% empty)")
            print(f"   Size: {format_size(audit.size_bytes)}, Moves: {audit.total_moves}")
            if audit.sources:
                for source, count in sorted(audit.sources.items(), key=lambda x: -x[1])[:3]:
                    print(f"   Source: {source}: {count}")
            if audit.error:
                print(f"   Error: {audit.error}")
            print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total databases: {len(databases)}")
    print(f"Total games: {total_games:,}")
    print(f"Games with moves: {total_with_moves:,} ({total_with_moves/total_games*100:.1f}%)" if total_games > 0 else "Games with moves: 0")
    print(f"Games without moves: {total_without_moves:,}")
    print(f"Total size: {format_size(total_size)}")
    print()
    print(f"Databases to DELETE (>=90% empty): {len(to_delete)}")
    print(f"Databases to FILTER (<90% empty): {len(to_filter)}")
    print(f"Databases to KEEP: {len(databases) - len(to_delete) - len(to_filter)}")

    if to_delete:
        print()
        print("Databases to DELETE:")
        for audit in to_delete:
            print(f"  - {os.path.basename(audit.path)}: {audit.total_games} games, {audit.percent_empty:.1f}% empty")

    if to_filter:
        print()
        print("Databases to FILTER:")
        for audit in to_filter:
            print(f"  - {os.path.basename(audit.path)}: {audit.games_without_moves}/{audit.total_games} games to remove")

    if args.cleanup:
        print()
        print("=" * 60)
        print("CLEANUP" + (" (DRY RUN)" if args.dry_run else ""))
        print("=" * 60)

        cleaned = 0
        for db_path in databases:
            action, removed = cleanup_database(db_path, dry_run=args.dry_run)
            if action not in ['keep', 'unknown']:
                print(f"{os.path.basename(db_path)}: {action}")
                cleaned += removed

        print()
        print(f"Total games {'would be ' if args.dry_run else ''}removed: {cleaned:,}")

    if args.json:
        output = {
            'summary': {
                'total_databases': len(databases),
                'total_games': total_games,
                'games_with_moves': total_with_moves,
                'games_without_moves': total_without_moves,
                'total_size_bytes': total_size,
                'to_delete': len(to_delete),
                'to_filter': len(to_filter),
            },
            'databases': [
                {
                    'path': a.path,
                    'total_games': a.total_games,
                    'games_with_moves': a.games_with_moves,
                    'percent_empty': a.percent_empty,
                    'recommendation': a.recommendation,
                    'size_bytes': a.size_bytes,
                    'error': a.error,
                }
                for a in results
            ]
        }
        print()
        print("JSON OUTPUT:")
        print(json.dumps(output, indent=2))

    return 0


if __name__ == '__main__':
    sys.exit(main())

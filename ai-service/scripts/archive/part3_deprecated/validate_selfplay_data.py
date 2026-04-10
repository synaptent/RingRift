#!/usr/bin/env python3
"""Validate selfplay data files for training usability.

This script checks DB and JSONL files to ensure they contain valid
game records that can be used for neural network training.

Usage:
    # Validate a single file
    python scripts/validate_selfplay_data.py path/to/file.db
    python scripts/validate_selfplay_data.py path/to/file.jsonl

    # Scan entire directory
    python scripts/validate_selfplay_data.py --scan data/selfplay

    # Output as JSON for further processing
    python scripts/validate_selfplay_data.py --scan data/selfplay --json

Required fields for training:
    - moves: Non-empty list of game moves
    - winner: Integer indicating the winning player (1-based)

Exit codes:
    0 - All files valid
    1 - Some files invalid (details in output)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


def validate_db(db_path: Path) -> dict[str, Any]:
    """Check if DB has usable game records.

    Supports two database formats:
    1. Simplified format: moves stored as JSON in 'moves' column of 'games' table
    2. GameReplayDB format: moves stored in separate 'game_moves' table

    Args:
        db_path: Path to SQLite database file

    Returns:
        Validation result dict with keys:
            - valid: bool
            - reason: str (if invalid)
            - total: int (total games)
            - usable: int (games with moves and winner)
            - empty_moves: int (games with empty/null moves)
            - no_winner: int (games without winner)
            - usable_pct: float (percentage usable, if valid)
            - schema: str ('simple' or 'replay_db')
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        if 'games' not in tables:
            conn.close()
            return {'valid': False, 'reason': 'No games table', 'total': 0, 'usable': 0}

        # Get column info for games table
        cursor.execute("PRAGMA table_info(games)")
        columns = [row[1] for row in cursor.fetchall()]

        # Determine schema type
        has_moves_column = 'moves' in columns
        has_game_moves_table = 'game_moves' in tables

        if not has_moves_column and not has_game_moves_table:
            conn.close()
            return {'valid': False, 'reason': 'No moves data (no moves column or game_moves table)', 'total': 0, 'usable': 0}

        schema_type = 'simple' if has_moves_column else 'replay_db'

        # Check for winner column
        if 'winner' not in columns:
            conn.close()
            return {'valid': False, 'reason': 'Missing winner column', 'total': 0, 'usable': 0}

        # Count total games
        cursor.execute("SELECT COUNT(*) FROM games")
        total = cursor.fetchone()[0]

        if total == 0:
            conn.close()
            return {'valid': False, 'reason': 'Empty database', 'total': 0, 'usable': 0}

        if schema_type == 'simple':
            # Simple format: moves in games table
            cursor.execute("""
                SELECT COUNT(*) FROM games
                WHERE moves IS NOT NULL
                AND moves != ''
                AND moves != '[]'
                AND winner IS NOT NULL
            """)
            usable = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM games
                WHERE moves IS NULL OR moves = '' OR moves = '[]'
            """)
            empty_moves = cursor.fetchone()[0]
        else:
            # GameReplayDB format: moves in game_moves table
            # Count games that have at least one move AND a winner
            cursor.execute("""
                SELECT COUNT(DISTINCT g.game_id) FROM games g
                JOIN game_moves gm ON g.game_id = gm.game_id
                WHERE g.winner IS NOT NULL
            """)
            usable = cursor.fetchone()[0]

            # Count games with no moves
            cursor.execute("""
                SELECT COUNT(*) FROM games g
                WHERE NOT EXISTS (SELECT 1 FROM game_moves gm WHERE gm.game_id = g.game_id)
            """)
            empty_moves = cursor.fetchone()[0]

        # Count games without winner
        cursor.execute("SELECT COUNT(*) FROM games WHERE winner IS NULL")
        no_winner = cursor.fetchone()[0]

        conn.close()

        if usable == 0:
            return {
                'valid': False,
                'reason': f'No usable games (empty_moves={empty_moves}, no_winner={no_winner})',
                'total': total,
                'usable': 0,
                'empty_moves': empty_moves,
                'no_winner': no_winner,
                'schema': schema_type
            }

        return {
            'valid': True,
            'total': total,
            'usable': usable,
            'empty_moves': empty_moves,
            'no_winner': no_winner,
            'usable_pct': usable / total * 100 if total > 0 else 0,
            'schema': schema_type
        }

    except Exception as e:
        return {'valid': False, 'reason': str(e), 'total': 0, 'usable': 0}


def validate_jsonl(jsonl_path: Path) -> dict[str, Any]:
    """Check if JSONL has usable game records.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        Validation result dict (same structure as validate_db)
    """
    try:
        total = 0
        usable = 0
        empty_moves = 0
        no_winner = 0

        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    game = json.loads(line)
                    moves = game.get('moves', [])
                    winner = game.get('winner')

                    if not moves or len(moves) == 0:
                        empty_moves += 1
                        continue
                    if winner is None:
                        no_winner += 1
                        continue
                    usable += 1
                except json.JSONDecodeError:
                    continue

        if total == 0:
            return {'valid': False, 'reason': 'Empty file', 'total': 0, 'usable': 0}

        if usable == 0:
            return {
                'valid': False,
                'reason': f'No usable games (empty_moves={empty_moves}, no_winner={no_winner})',
                'total': total,
                'usable': 0,
                'empty_moves': empty_moves,
                'no_winner': no_winner
            }

        return {
            'valid': True,
            'total': total,
            'usable': usable,
            'empty_moves': empty_moves,
            'no_winner': no_winner,
            'usable_pct': usable / total * 100 if total > 0 else 0
        }

    except Exception as e:
        return {'valid': False, 'reason': str(e), 'total': 0, 'usable': 0}


def validate_file(path: Path) -> dict[str, Any]:
    """Validate a single file based on its extension."""
    if path.suffix == '.db':
        return validate_db(path)
    elif path.suffix == '.jsonl':
        return validate_jsonl(path)
    else:
        return {'valid': False, 'reason': 'Unknown file type'}


def scan_directory(directory: Path) -> list[dict[str, Any]]:
    """Scan a directory for all DB and JSONL files and validate them."""
    results = []

    for pattern in ['**/*.db', '**/*.jsonl']:
        for path in directory.glob(pattern):
            result = validate_file(path)
            result['path'] = str(path)
            result['size'] = path.stat().st_size
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate selfplay data files for training usability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('path', nargs='?', type=Path, help='File or directory to validate')
    parser.add_argument('--scan', type=Path, help='Directory to scan recursively')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--invalid-only', action='store_true', help='Only show invalid files')
    args = parser.parse_args()

    if args.scan:
        results = scan_directory(args.scan)

        if args.invalid_only:
            results = [r for r in results if not r['valid']]

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            valid_count = sum(1 for r in results if r['valid'])
            invalid_count = len(results) - valid_count

            print(f"Scanned {len(results)} files: {valid_count} valid, {invalid_count} invalid\n")

            if invalid_count > 0:
                print("Invalid files:")
                for r in results:
                    if not r['valid']:
                        print(f"  {r['path']}: {r.get('reason', 'Unknown')}")

            sys.exit(0 if invalid_count == 0 else 1)

    elif args.path:
        result = validate_file(args.path)

        if args.json:
            print(json.dumps(result))
        else:
            if result['valid']:
                print(f"VALID: {result['usable']}/{result['total']} usable ({result['usable_pct']:.1f}%)")
            else:
                print(f"INVALID: {result.get('reason', 'Unknown')}")

            sys.exit(0 if result['valid'] else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

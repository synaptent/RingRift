#!/usr/bin/env python3
"""Detect and quarantine games generated with biased heuristic engine.

This script identifies games that were generated with the buggy heuristic-only
engine that had asymmetric evaluation (P2 bias). These games should be excluded
from training data because they teach the model incorrect patterns.

The bug was in:
1. material_evaluator.py - summed opponents instead of averaging
2. heuristic_ai.py - asymmetric swap bonus for P2
3. strategic_evaluator.py - asymmetric opponent_victory_threat

Known biased engine modes:
- 'heuristic-only' - CPU heuristic selfplay (P1=15%, P2=85%)

Known balanced engine modes:
- 'gpu_heuristic' - GPU heuristic (P1=50%, P2=50%)
- 'gumbel-mcts' - Gumbel MCTS
- 'descent-only' - NN descent
- 'nnue-guided' - NNUE guided

Usage:
    # Scan all databases and report biased periods
    python scripts/detect_biased_games.py --scan

    # Quarantine heuristic-only games (dry run)
    python scripts/detect_biased_games.py --quarantine --engine heuristic-only --dry-run

    # Apply quarantine
    python scripts/detect_biased_games.py --quarantine --engine heuristic-only --apply

    # Verify quarantine
    python scripts/detect_biased_games.py --verify
"""

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Known biased engine modes that should be quarantined
BIASED_ENGINES = {
    'heuristic-only',
    'heuristic_only',  # Alternative naming
}

# Known balanced engine modes that are safe for training
BALANCED_ENGINES = {
    'gpu_heuristic',
    'gumbel-mcts',
    'gumbel_mcts',
    'descent-only',
    'descent_only',
    'nnue-guided',
    'nnue_guided',
    'nn_vs_nn_tournament',
    'policy-only',
    'policy_only',
}

# Threshold for detecting bias (P1 win rate outside this range is biased)
BIAS_MIN_P1_RATE = 0.35
BIAS_MAX_P1_RATE = 0.65


def scan_database(db_path: str) -> dict:
    """Scan a database for biased game periods.

    Returns dict with statistics about games by engine mode.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if it has game data
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='games'
        """)
        if not cursor.fetchone():
            conn.close()
            return None

        # Get column info
        cursor.execute("PRAGMA table_info(games)")
        cols = {row[1] for row in cursor.fetchall()}

        if 'engine_mode' not in cols:
            conn.close()
            return {'error': 'no engine_mode column'}

        has_excluded = 'excluded_from_training' in cols

        # Get stats by engine mode
        query = """
            SELECT
                engine_mode,
                board_type,
                num_players,
                winner,
                COUNT(*) as count
            FROM games
            WHERE game_status = 'completed' AND winner IS NOT NULL
        """
        if has_excluded:
            query += " AND (excluded_from_training IS NULL OR excluded_from_training = 0)"
        query += " GROUP BY engine_mode, board_type, num_players, winner"

        cursor.execute(query)

        stats = defaultdict(lambda: defaultdict(lambda: {'p1': 0, 'p2': 0, 'other': 0, 'total': 0}))
        for row in cursor.fetchall():
            engine = row['engine_mode'] or 'unknown'
            config = f"{row['board_type']}_{row['num_players']}p"
            winner = row['winner']
            count = row['count']

            if winner == 1:
                stats[engine][config]['p1'] += count
            elif winner == 2:
                stats[engine][config]['p2'] += count
            else:
                stats[engine][config]['other'] += count
            stats[engine][config]['total'] += count

        conn.close()
        return dict(stats)

    except Exception as e:
        return {'error': str(e)}


def quarantine_by_engine(db_path: str, engine_mode: str, dry_run: bool = True) -> dict:
    """Quarantine games from a specific engine mode.

    Sets excluded_from_training = 1 for games with the specified engine_mode.

    Returns dict with count of affected games.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if it has game data
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='games'
    """)
    if not cursor.fetchone():
        conn.close()
        return {'error': 'no games table'}

    # Get column info
    cursor.execute("PRAGMA table_info(games)")
    cols = {row[1] for row in cursor.fetchall()}

    if 'engine_mode' not in cols:
        conn.close()
        return {'error': 'no engine_mode column'}

    # Add excluded_from_training column if needed
    if 'excluded_from_training' not in cols:
        cursor.execute("""
            ALTER TABLE games ADD COLUMN excluded_from_training INTEGER DEFAULT 0
        """)
        conn.commit()

    # Count games to quarantine
    cursor.execute("""
        SELECT COUNT(*) FROM games
        WHERE engine_mode = ?
        AND (excluded_from_training IS NULL OR excluded_from_training = 0)
    """, (engine_mode,))
    count = cursor.fetchone()[0]

    result = {
        'db_path': db_path,
        'engine_mode': engine_mode,
        'games_to_quarantine': count,
        'dry_run': dry_run,
    }

    if not dry_run and count > 0:
        cursor.execute("""
            UPDATE games
            SET excluded_from_training = 1
            WHERE engine_mode = ?
            AND (excluded_from_training IS NULL OR excluded_from_training = 0)
        """, (engine_mode,))
        conn.commit()
        result['games_quarantined'] = cursor.rowcount

    conn.close()
    return result


def verify_database(db_path: str) -> dict:
    """Verify quarantine status of a database.

    Returns dict with counts of quarantined vs active games by engine.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if it has game data
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='games'
    """)
    if not cursor.fetchone():
        conn.close()
        return {'error': 'no games table'}

    # Get column info
    cursor.execute("PRAGMA table_info(games)")
    cols = {row[1] for row in cursor.fetchall()}

    if 'excluded_from_training' not in cols:
        conn.close()
        return {'error': 'no excluded_from_training column'}

    # Get stats
    cursor.execute("""
        SELECT
            engine_mode,
            excluded_from_training,
            COUNT(*) as count
        FROM games
        GROUP BY engine_mode, excluded_from_training
    """)

    stats = defaultdict(lambda: {'active': 0, 'quarantined': 0})
    for row in cursor.fetchall():
        engine = row['engine_mode'] or 'unknown'
        excluded = row['excluded_from_training'] or 0
        count = row['count']

        if excluded:
            stats[engine]['quarantined'] += count
        else:
            stats[engine]['active'] += count

    conn.close()
    return dict(stats)


def find_game_databases() -> list[Path]:
    """Find all game databases in data/games and subdirectories."""
    base_dir = Path("data/games")
    if not base_dir.exists():
        base_dir = Path(".")

    game_dbs = []
    for db_path in base_dir.rglob("*.db"):
        game_dbs.append(db_path)

    return sorted(game_dbs)


def main():
    parser = argparse.ArgumentParser(
        description='Detect and quarantine biased heuristic games'
    )
    parser.add_argument(
        '--scan', action='store_true',
        help='Scan all databases for biased periods'
    )
    parser.add_argument(
        '--quarantine', action='store_true',
        help='Quarantine games from biased engine'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify quarantine status of all databases'
    )
    parser.add_argument(
        '--engine', type=str, default='heuristic-only',
        help='Engine mode to quarantine (default: heuristic-only)'
    )
    parser.add_argument(
        '--db', type=str, default=None,
        help='Specific database to process (default: all in data/games)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Apply changes (required for --quarantine)'
    )
    args = parser.parse_args()

    if args.db:
        db_paths = [Path(args.db)]
    else:
        db_paths = find_game_databases()

    print(f"Found {len(db_paths)} database(s)")
    print()

    if args.scan:
        print("=" * 80)
        print("SCANNING FOR BIASED ENGINE PERIODS")
        print("=" * 80)

        biased_found = []

        for db_path in db_paths:
            stats = scan_database(str(db_path))
            if not stats or 'error' in stats:
                continue

            db_has_bias = False
            for engine, configs in stats.items():
                for config, data in configs.items():
                    total = data['total']
                    if total < 20:
                        continue

                    if data['p2'] > 0 or data['p1'] > 0:
                        # Only check 2-player games
                        if '2p' in config:
                            p1_rate = data['p1'] / total
                            is_biased = p1_rate < BIAS_MIN_P1_RATE or p1_rate > BIAS_MAX_P1_RATE

                            if is_biased:
                                db_has_bias = True
                                biased_found.append({
                                    'db': str(db_path),
                                    'engine': engine,
                                    'config': config,
                                    'p1_rate': p1_rate,
                                    'total': total,
                                })

            if db_has_bias:
                print(f"\n{db_path}:")
                for engine, configs in stats.items():
                    for config, data in configs.items():
                        total = data['total']
                        if total < 20:
                            continue
                        if '2p' in config:
                            p1_rate = data['p1'] / total if total > 0 else 0
                            flag = "⚠️ BIASED" if (p1_rate < BIAS_MIN_P1_RATE or p1_rate > BIAS_MAX_P1_RATE) else "✓"
                            print(f"  {engine} / {config}: P1={data['p1']} ({p1_rate*100:.1f}%) P2={data['p2']} ({(1-p1_rate)*100:.1f}%) {flag}")

        print()
        print("=" * 80)
        print(f"SUMMARY: Found {len(biased_found)} biased engine/config combinations")
        print("=" * 80)

        if biased_found:
            engines = set(b['engine'] for b in biased_found)
            print(f"Biased engines: {', '.join(sorted(engines))}")
            print()
            print("Run with --quarantine --engine <engine> --apply to quarantine")

    if args.quarantine:
        if not args.apply and not args.dry_run:
            print("ERROR: Must specify either --dry-run or --apply")
            return

        print("=" * 80)
        print(f"QUARANTINING ENGINE: {args.engine}")
        print("=" * 80)

        total_quarantined = 0
        for db_path in db_paths:
            result = quarantine_by_engine(
                str(db_path),
                args.engine,
                dry_run=not args.apply
            )
            if 'error' in result:
                continue

            count = result.get('games_to_quarantine', 0)
            if count > 0:
                status = "WOULD QUARANTINE" if not args.apply else "QUARANTINED"
                print(f"{db_path}: {status} {count} games")
                total_quarantined += count

        print()
        print(f"Total: {total_quarantined} games {'would be' if not args.apply else ''} quarantined")
        if not args.apply:
            print()
            print("Run with --apply to execute quarantine")

    if args.verify:
        print("=" * 80)
        print("VERIFYING QUARANTINE STATUS")
        print("=" * 80)

        for db_path in db_paths:
            stats = verify_database(str(db_path))
            if 'error' in stats:
                continue

            has_data = False
            for engine, data in stats.items():
                if data['active'] > 0 or data['quarantined'] > 0:
                    has_data = True

            if has_data:
                print(f"\n{db_path}:")
                for engine, data in sorted(stats.items()):
                    if data['active'] > 0 or data['quarantined'] > 0:
                        status = "⚠️" if engine in BIASED_ENGINES and data['active'] > 0 else "✓"
                        print(f"  {engine}: active={data['active']} quarantined={data['quarantined']} {status}")


if __name__ == "__main__":
    main()

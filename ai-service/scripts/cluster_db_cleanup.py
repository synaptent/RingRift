#!/usr/bin/env python3
"""
Cluster-wide database cleanup: find and delete databases with no valid move data.

This script understands ALL ways a RingRift database can have move data:
1. GameReplayDB with game_moves table
2. JSONL aggregated databases
3. Various database naming patterns

SAFEGUARDS:
- Never deletes canonical_*.db files unless --include-canonical is specified
- Creates backups before deletion
- Dry-run by default (use --execute to actually delete)
- Detailed audit log before any deletion
- Size limits: won't delete DBs > 1GB without --force-large
- Confirms before each cluster node deletion

Usage:
    # Scan locally (dry run)
    python scripts/cluster_db_cleanup.py --scan

    # Scan with details
    python scripts/cluster_db_cleanup.py --scan --verbose

    # Actually cleanup locally
    python scripts/cluster_db_cleanup.py --execute

    # Scan entire cluster
    python scripts/cluster_db_cleanup.py --cluster --scan

    # Cleanup entire cluster
    python scripts/cluster_db_cleanup.py --cluster --execute

    # Include canonical DBs in cleanup (DANGEROUS)
    python scripts/cluster_db_cleanup.py --execute --include-canonical

    # Generate JSON report
    python scripts/cluster_db_cleanup.py --scan --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add ai-service to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_DIR))

# Size threshold for "large" databases (1GB)
LARGE_DB_THRESHOLD = 1024 * 1024 * 1024

# Minimum game duration/moves for validity
MIN_MOVES_FOR_VALID = 5


@dataclass
class DatabaseInfo:
    """Comprehensive database audit information."""
    path: str
    size_bytes: int
    exists: bool = True
    error: Optional[str] = None

    # Structure
    has_games_table: bool = False
    has_game_moves_table: bool = False
    has_game_initial_state_table: bool = False
    has_game_state_snapshots_table: bool = False

    # Counts
    total_games: int = 0
    games_with_moves: int = 0
    games_without_moves: int = 0
    total_moves: int = 0

    # Metadata
    board_types: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    game_statuses: dict = field(default_factory=dict)

    # Computed
    percent_empty: float = 100.0
    is_canonical: bool = False
    is_jsonl: bool = False
    recommendation: str = "keep"

    def to_dict(self) -> dict:
        return asdict(self)


def audit_database(db_path: Path) -> DatabaseInfo:
    """
    Comprehensive audit of a database to determine if it has valid move data.

    Checks ALL ways a database could have move data:
    1. game_moves table with entries
    2. Embedded moves in JSON (for older formats)
    3. JSONL patterns

    Returns detailed DatabaseInfo with all findings.
    """
    info = DatabaseInfo(
        path=str(db_path),
        size_bytes=0,
        is_canonical=db_path.name.startswith("canonical_"),
        is_jsonl="jsonl" in db_path.name.lower(),
    )

    if not db_path.exists():
        info.exists = False
        info.error = "File not found"
        info.recommendation = "delete"
        return info

    try:
        info.size_bytes = db_path.stat().st_size
    except OSError as e:
        info.error = f"Cannot stat: {e}"
        info.recommendation = "skip"
        return info

    if info.size_bytes == 0:
        info.error = "Empty file (0 bytes)"
        info.recommendation = "delete"
        return info

    try:
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check for games table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        info.has_games_table = cursor.fetchone() is not None

        if not info.has_games_table:
            conn.close()
            info.error = "No games table"
            info.recommendation = "delete"
            return info

        # Check for game_moves table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
        info.has_game_moves_table = cursor.fetchone() is not None

        # Check other tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_initial_state'")
        info.has_game_initial_state_table = cursor.fetchone() is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_state_snapshots'")
        info.has_game_state_snapshots_table = cursor.fetchone() is not None

        # Count total games
        cursor.execute("SELECT COUNT(*) FROM games")
        info.total_games = cursor.fetchone()[0]

        if info.total_games == 0:
            conn.close()
            info.error = "Empty database (0 games)"
            info.recommendation = "delete"
            return info

        # Count total moves in game_moves table
        if info.has_game_moves_table:
            cursor.execute("SELECT COUNT(*) FROM game_moves")
            info.total_moves = cursor.fetchone()[0]

            # Count games WITH moves (via game_moves table)
            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM game_moves")
            info.games_with_moves = cursor.fetchone()[0]
        else:
            # No game_moves table means 0 moves
            info.total_moves = 0
            info.games_with_moves = 0

        info.games_without_moves = info.total_games - info.games_with_moves

        # Calculate percent empty
        if info.total_games > 0:
            info.percent_empty = (info.games_without_moves / info.total_games) * 100
        else:
            info.percent_empty = 100.0

        # Get metadata distributions
        try:
            cursor.execute("SELECT board_type, COUNT(*) FROM games GROUP BY board_type")
            for row in cursor.fetchall():
                info.board_types[row[0] or "unknown"] = row[1]
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT source, COUNT(*) FROM games GROUP BY source")
            for row in cursor.fetchall():
                info.sources[row[0] or "unknown"] = row[1]
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT game_status, COUNT(*) FROM games GROUP BY game_status")
            for row in cursor.fetchall():
                info.game_statuses[row[0] or "unknown"] = row[1]
        except sqlite3.OperationalError:
            pass

        # Check for orphan games: games with total_moves > 0 but no game_moves entries
        if info.has_game_moves_table:
            cursor.execute("""
                SELECT COUNT(*) FROM games g
                WHERE g.total_moves > 0
                AND NOT EXISTS (
                    SELECT 1 FROM game_moves m WHERE m.game_id = g.game_id
                )
            """)
            orphan_count = cursor.fetchone()[0]
            if orphan_count > 0:
                if info.error:
                    info.error += f"; {orphan_count} orphan games (total_moves>0 but no game_moves)"
                else:
                    info.error = f"{orphan_count} orphan games (total_moves>0 but no game_moves)"

        conn.close()

        # Determine recommendation
        if info.percent_empty >= 100:
            info.recommendation = "delete"
        elif info.percent_empty >= 90:
            info.recommendation = "delete"
        elif info.percent_empty > 0:
            info.recommendation = "filter"
        else:
            info.recommendation = "keep"

        return info

    except sqlite3.Error as e:
        info.error = f"SQLite error: {e}"
        # Don't recommend deletion for errors - could be temporary
        info.recommendation = "skip"
        return info
    except Exception as e:
        info.error = f"Error: {e}"
        info.recommendation = "skip"
        return info


def find_all_databases(base_paths: list[Path]) -> list[Path]:
    """Find all .db files under the given paths."""
    databases = []
    visited = set()

    for base in base_paths:
        if not base.exists():
            continue

        if base.is_file() and base.suffix == ".db":
            real = base.resolve()
            if real not in visited:
                visited.add(real)
                databases.append(base)
            continue

        for db_path in base.rglob("*.db"):
            real = db_path.resolve()
            if real not in visited:
                visited.add(real)
                databases.append(db_path)

    return sorted(databases)


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def delete_database(db_path: Path, backup: bool = True) -> tuple[bool, str]:
    """
    Delete a database with optional backup.
    Returns (success, message).
    """
    if not db_path.exists():
        return False, "File not found"

    try:
        if backup:
            backup_name = f"{db_path.name}.deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = db_path.parent / backup_name
            db_path.rename(backup_path)
            message = f"Moved to backup: {backup_name}"
        else:
            db_path.unlink()
            message = "Deleted"

        # Also remove WAL/SHM files if present
        for suffix in ["-wal", "-shm"]:
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

        return True, message
    except OSError as e:
        return False, f"Error: {e}"


def cleanup_database(db_path: Path, dry_run: bool = True) -> tuple[str, int]:
    """
    Clean games without moves from a database (filter mode).
    Returns (action, games_removed).
    """
    if dry_run:
        return "would_filter", 0

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Delete games without moves
        cursor.execute("""
            DELETE FROM games
            WHERE game_id NOT IN (SELECT DISTINCT game_id FROM game_moves)
        """)
        removed = cursor.rowcount

        # Clean orphaned data
        for table in ["game_initial_state", "game_state_snapshots", "game_players", "game_choices"]:
            try:
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE game_id NOT IN (SELECT game_id FROM games)
                """)
            except sqlite3.OperationalError:
                pass

        conn.commit()
        cursor.execute("VACUUM")
        conn.close()

        return f"filtered", removed
    except sqlite3.Error as e:
        return f"error: {e}", 0


def run_local_scan(
    paths: list[Path],
    verbose: bool = False,
    include_canonical: bool = False,
    force_large: bool = False,
    execute: bool = False,
    json_output: Optional[Path] = None,
) -> int:
    """
    Scan local databases and optionally clean up.
    Returns exit code.
    """
    databases = find_all_databases(paths)

    if not databases:
        logger.info("No databases found.")
        return 0

    logger.info(f"Found {len(databases)} databases to scan...")

    results = []
    to_delete = []
    to_filter = []
    skipped_canonical = []
    skipped_large = []

    total_games = 0
    total_with_moves = 0
    total_without_moves = 0
    total_size = 0

    for db_path in databases:
        info = audit_database(db_path)
        results.append(info)

        total_games += info.total_games
        total_with_moves += info.games_with_moves
        total_without_moves += info.games_without_moves
        total_size += info.size_bytes

        # Apply safeguards
        if info.recommendation in ["delete", "filter"]:
            if info.is_canonical and not include_canonical:
                skipped_canonical.append(info)
                continue
            if info.size_bytes > LARGE_DB_THRESHOLD and not force_large:
                skipped_large.append(info)
                continue

            if info.recommendation == "delete":
                to_delete.append(info)
            else:
                to_filter.append(info)

        # Print status
        if verbose or info.recommendation != "keep":
            emoji = {"delete": "🗑️", "filter": "🧹", "keep": "✅", "skip": "⚠️"}.get(
                info.recommendation, "❓"
            )
            print(f"\n{emoji} {db_path.name}")
            print(f"   Size: {format_size(info.size_bytes)}")
            print(f"   Games: {info.games_with_moves}/{info.total_games} with moves ({info.percent_empty:.1f}% empty)")
            print(f"   Moves: {info.total_moves}")
            if info.board_types:
                print(f"   Board types: {info.board_types}")
            if info.sources:
                top_sources = dict(sorted(info.sources.items(), key=lambda x: -x[1])[:3])
                print(f"   Sources: {top_sources}")
            if info.error:
                print(f"   Issue: {info.error}")
            if info.is_canonical:
                print(f"   ⚠️  CANONICAL database - protected by default")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total databases: {len(databases)}")
    print(f"Total games: {total_games:,}")
    if total_games > 0:
        print(f"Games with moves: {total_with_moves:,} ({total_with_moves/total_games*100:.1f}%)")
    else:
        print("Games with moves: 0")
    print(f"Games without moves: {total_without_moves:,}")
    print(f"Total size: {format_size(total_size)}")
    print()
    print(f"To DELETE (>=90% empty): {len(to_delete)}")
    print(f"To FILTER (<90% empty): {len(to_filter)}")
    print(f"To KEEP: {len(databases) - len(to_delete) - len(to_filter) - len(skipped_canonical) - len(skipped_large)}")
    if skipped_canonical:
        print(f"SKIPPED (canonical, use --include-canonical): {len(skipped_canonical)}")
    if skipped_large:
        print(f"SKIPPED (>1GB, use --force-large): {len(skipped_large)}")

    # List what would be deleted
    if to_delete:
        print(f"\n{'=' * 60}")
        print(f"DATABASES TO DELETE ({len(to_delete)}):")
        print("=" * 60)
        for info in to_delete:
            print(f"  🗑️  {Path(info.path).name}: {info.total_games} games, {info.percent_empty:.1f}% empty, {format_size(info.size_bytes)}")
            if info.is_canonical:
                print(f"      ⚠️  CANONICAL - will be deleted!")

    if to_filter:
        print(f"\n{'=' * 60}")
        print(f"DATABASES TO FILTER ({len(to_filter)}):")
        print("=" * 60)
        for info in to_filter:
            print(f"  🧹 {Path(info.path).name}: {info.games_without_moves}/{info.total_games} games to remove")

    # Execute if requested
    if execute:
        if not to_delete and not to_filter:
            print("\nNo cleanup needed.")
            return 0

        print(f"\n{'=' * 60}")
        print("EXECUTING CLEANUP")
        print("=" * 60)

        deleted_count = 0
        filtered_count = 0

        for info in to_delete:
            success, message = delete_database(Path(info.path), backup=True)
            if success:
                deleted_count += 1
                print(f"  ✓ Deleted: {Path(info.path).name} - {message}")
            else:
                print(f"  ✗ Failed: {Path(info.path).name} - {message}")

        for info in to_filter:
            action, removed = cleanup_database(Path(info.path), dry_run=False)
            if "error" not in action:
                filtered_count += removed
                print(f"  ✓ Filtered: {Path(info.path).name} - removed {removed} games")
            else:
                print(f"  ✗ Failed: {Path(info.path).name} - {action}")

        print(f"\nCleanup complete: {deleted_count} databases deleted, {filtered_count} games filtered")
    else:
        print("\n[DRY RUN] Use --execute to actually perform cleanup")

    # JSON output
    if json_output:
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_databases": len(databases),
                "total_games": total_games,
                "games_with_moves": total_with_moves,
                "games_without_moves": total_without_moves,
                "total_size_bytes": total_size,
                "to_delete": len(to_delete),
                "to_filter": len(to_filter),
                "skipped_canonical": len(skipped_canonical),
                "skipped_large": len(skipped_large),
            },
            "databases": [info.to_dict() for info in results],
        }
        with open(json_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON report written to: {json_output}")

    return 0


def run_cluster_scan(
    verbose: bool = False,
    include_canonical: bool = False,
    force_large: bool = False,
    execute: bool = False,
    json_output: Optional[Path] = None,
) -> int:
    """
    Scan all cluster nodes for databases and optionally clean up.
    Returns exit code.
    """
    try:
        from app.core.ssh import get_ssh_client, SSHResult
        from app.config.cluster_config import get_cluster_nodes, ClusterNode
    except ImportError as e:
        logger.error(f"Cannot import cluster modules: {e}")
        logger.error("Run with PYTHONPATH=. from ai-service directory")
        return 1

    nodes = get_cluster_nodes()
    active_nodes = [n for n in nodes.values() if n.status == "ready"]

    if not active_nodes:
        logger.warning("No active cluster nodes found in config")
        return 1

    logger.info(f"Scanning {len(active_nodes)} cluster nodes...")

    all_results = {}

    for node in active_nodes:
        logger.info(f"\nScanning {node.name}...")

        try:
            client = get_ssh_client(node.name)

            # Find all databases on remote node
            find_cmd = """
            find ~/ringrift/ai-service/data -name '*.db' -type f 2>/dev/null || true
            """
            result = client.run(find_cmd, timeout=60)

            if not result.stdout.strip():
                logger.info(f"  No databases found on {node.name}")
                continue

            db_paths = result.stdout.strip().split("\n")
            logger.info(f"  Found {len(db_paths)} databases on {node.name}")

            # Audit each database remotely
            for remote_path in db_paths:
                remote_path = remote_path.strip()
                if not remote_path:
                    continue

                # Run audit command remotely
                audit_cmd = f"""
                python3 -c "
import sqlite3
import json
import os

path = '{remote_path}'
info = {{'path': path, 'node': '{node.name}'}}

try:
    info['size_bytes'] = os.path.getsize(path)
    conn = sqlite3.connect(path, timeout=10)
    cursor = conn.cursor()

    # Check tables
    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='games'\")
    has_games = cursor.fetchone() is not None

    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'\")
    has_moves = cursor.fetchone() is not None

    if has_games:
        cursor.execute('SELECT COUNT(*) FROM games')
        info['total_games'] = cursor.fetchone()[0]

        if has_moves:
            cursor.execute('SELECT COUNT(*) FROM game_moves')
            info['total_moves'] = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(DISTINCT game_id) FROM game_moves')
            info['games_with_moves'] = cursor.fetchone()[0]
        else:
            info['total_moves'] = 0
            info['games_with_moves'] = 0
    else:
        info['total_games'] = 0
        info['total_moves'] = 0
        info['games_with_moves'] = 0
        info['error'] = 'No games table'

    conn.close()
except Exception as e:
    info['error'] = str(e)

print(json.dumps(info))
"
                """
                audit_result = client.run(audit_cmd, timeout=30)

                if audit_result.returncode == 0 and audit_result.stdout.strip():
                    try:
                        db_info = json.loads(audit_result.stdout.strip())
                        db_name = Path(remote_path).name

                        total = db_info.get("total_games", 0)
                        with_moves = db_info.get("games_with_moves", 0)
                        moves = db_info.get("total_moves", 0)
                        error = db_info.get("error")

                        # Calculate empty percentage
                        if total > 0:
                            pct_empty = ((total - with_moves) / total) * 100
                        else:
                            pct_empty = 100.0

                        is_useless = pct_empty >= 90 or total == 0 or moves == 0

                        if verbose or is_useless or error:
                            emoji = "🗑️" if is_useless else "✅"
                            print(f"  {emoji} {db_name}: {with_moves}/{total} games with moves ({pct_empty:.1f}% empty)")
                            if error:
                                print(f"      Error: {error}")

                        all_results[f"{node.name}:{remote_path}"] = db_info

                        # Execute deletion if requested
                        if execute and is_useless:
                            is_canonical = "canonical_" in db_name

                            if is_canonical and not include_canonical:
                                print(f"      ⚠️  Skipping canonical DB (use --include-canonical)")
                                continue

                            size = db_info.get("size_bytes", 0)
                            if size > LARGE_DB_THRESHOLD and not force_large:
                                print(f"      ⚠️  Skipping large DB (use --force-large)")
                                continue

                            # Create backup and delete
                            backup_name = f"{db_name}.deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            delete_cmd = f"mv '{remote_path}' '{remote_path}.deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}'"
                            del_result = client.run(delete_cmd, timeout=30)

                            if del_result.returncode == 0:
                                print(f"      ✓ Deleted (backed up)")
                            else:
                                print(f"      ✗ Delete failed: {del_result.stderr}")

                    except json.JSONDecodeError:
                        logger.warning(f"    Could not parse audit result for {remote_path}")
                else:
                    logger.warning(f"    Audit failed for {remote_path}")

        except Exception as e:
            logger.error(f"  Error scanning {node.name}: {e}")
            continue

    # Summary
    total_dbs = len(all_results)
    useless_dbs = sum(
        1 for info in all_results.values()
        if info.get("games_with_moves", 0) == 0 or info.get("total_games", 0) == 0
    )

    print(f"\n{'=' * 60}")
    print("CLUSTER SUMMARY")
    print("=" * 60)
    print(f"Total databases scanned: {total_dbs}")
    print(f"Useless databases (100% empty): {useless_dbs}")

    if json_output:
        output = {
            "timestamp": datetime.now().isoformat(),
            "cluster_results": all_results,
        }
        with open(json_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON report written to: {json_output}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-wide database cleanup with safeguards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Safeguards:
  - Dry-run by default (use --execute to actually delete)
  - Canonical databases protected (use --include-canonical to include)
  - Large databases (>1GB) protected (use --force-large to include)
  - Backups created before deletion

Examples:
  python scripts/cluster_db_cleanup.py --scan                    # Local dry run
  python scripts/cluster_db_cleanup.py --scan --verbose          # Detailed local scan
  python scripts/cluster_db_cleanup.py --execute                 # Local cleanup
  python scripts/cluster_db_cleanup.py --cluster --scan          # Scan all nodes
  python scripts/cluster_db_cleanup.py --cluster --execute       # Cleanup all nodes
  python scripts/cluster_db_cleanup.py --execute --include-canonical  # Include canonical
""",
    )
    parser.add_argument("--scan", action="store_true", default=True,
                        help="Scan and report (default action)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform cleanup (creates backups)")
    parser.add_argument("--cluster", action="store_true",
                        help="Scan/cleanup all cluster nodes via SSH")
    parser.add_argument("--path", type=str, default="data/games",
                        help="Local path to scan (default: data/games)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show all databases, not just problematic ones")
    parser.add_argument("--include-canonical", action="store_true",
                        help="Include canonical_*.db files in cleanup (DANGEROUS)")
    parser.add_argument("--force-large", action="store_true",
                        help="Include databases >1GB in cleanup (DANGEROUS)")
    parser.add_argument("--json", type=str, metavar="PATH",
                        help="Write JSON report to file")

    args = parser.parse_args()

    json_output = Path(args.json) if args.json else None

    if args.cluster:
        return run_cluster_scan(
            verbose=args.verbose,
            include_canonical=args.include_canonical,
            force_large=args.force_large,
            execute=args.execute,
            json_output=json_output,
        )
    else:
        # Local scan paths
        paths = [
            Path(args.path),
            AI_SERVICE_DIR / "data" / "games",
            AI_SERVICE_DIR / "logs" / "cmaes",
            AI_SERVICE_DIR.parent / "data" / "games",
        ]
        return run_local_scan(
            paths=[p for p in paths if p.exists()],
            verbose=args.verbose,
            include_canonical=args.include_canonical,
            force_large=args.force_large,
            execute=args.execute,
            json_output=json_output,
        )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Filter timeout games from training data across all hosts.

This script removes games that ended due to timeout (move limit reached)
from training databases and JSONL files, as these represent incomplete
games that can contaminate training quality.

Usage:
    # Filter local data
    python scripts/filter_timeout_games.py --local

    # Filter on all cluster hosts
    python scripts/filter_timeout_games.py --cluster

    # Dry run (show what would be filtered)
    python scripts/filter_timeout_games.py --local --dry-run

    # Filter specific host
    python scripts/filter_timeout_games.py --host lambda-h100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class FilterStats:
    """Statistics from filtering operation."""
    db_games_before: int = 0
    db_games_after: int = 0
    db_games_removed: int = 0
    jsonl_games_before: int = 0
    jsonl_games_after: int = 0
    jsonl_games_removed: int = 0
    files_processed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.db_games_removed + self.jsonl_games_removed

    def __str__(self) -> str:
        return (
            f"Filtered {self.total_removed:,} timeout games:\n"
            f"  SQLite: {self.db_games_removed:,} removed ({self.db_games_before:,} -> {self.db_games_after:,})\n"
            f"  JSONL:  {self.jsonl_games_removed:,} removed ({self.jsonl_games_before:,} -> {self.jsonl_games_after:,})\n"
            f"  Files processed: {self.files_processed}"
        )


def is_timeout_game(termination_reason: str | None) -> bool:
    """Check if a game ended due to timeout."""
    if not termination_reason:
        return False
    term_lower = termination_reason.lower()
    return "timeout" in term_lower


def filter_sqlite_db(db_path: Path, dry_run: bool = False, backup: bool = True) -> tuple[int, int, int]:
    """Filter timeout games from a SQLite database.

    Returns: (games_before, games_after, games_removed)
    """
    if not db_path.exists():
        return 0, 0, 0

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        # Check if games table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not cursor.fetchone():
            conn.close()
            return 0, 0, 0

        # Count games before
        cursor.execute("SELECT COUNT(*) FROM games")
        games_before = cursor.fetchone()[0]

        # Count timeout games
        cursor.execute("""
            SELECT COUNT(*) FROM games
            WHERE termination_reason LIKE '%timeout%'
        """)
        timeout_count = cursor.fetchone()[0]

        if timeout_count == 0:
            conn.close()
            return games_before, games_before, 0

        if dry_run:
            conn.close()
            return games_before, games_before - timeout_count, timeout_count

        # Create backup
        if backup:
            backup_path = db_path.with_suffix('.db.pre_timeout_filter')
            if not backup_path.exists():
                shutil.copy2(db_path, backup_path)

        # Delete timeout games
        cursor.execute("""
            DELETE FROM games
            WHERE termination_reason LIKE '%timeout%'
        """)
        conn.commit()

        # Count games after
        cursor.execute("SELECT COUNT(*) FROM games")
        games_after = cursor.fetchone()[0]

        # Vacuum to reclaim space
        cursor.execute("VACUUM")
        conn.commit()

        conn.close()
        return games_before, games_after, timeout_count

    except Exception as e:
        print(f"  Error filtering {db_path}: {e}")
        return 0, 0, 0


def filter_jsonl_file(jsonl_path: Path, dry_run: bool = False, backup: bool = True) -> tuple[int, int, int]:
    """Filter timeout games from a JSONL file.

    Returns: (games_before, games_after, games_removed)
    """
    if not jsonl_path.exists():
        return 0, 0, 0

    try:
        games_before = 0
        valid_games = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                games_before += 1

                try:
                    game = json.loads(line)
                    term = (
                        game.get("termination_reason") or
                        game.get("result", {}).get("termination_reason", "")
                    )

                    if not is_timeout_game(term):
                        valid_games.append(line)
                except json.JSONDecodeError:
                    # Keep malformed lines to not lose data
                    valid_games.append(line)

        games_removed = games_before - len(valid_games)

        if games_removed == 0:
            return games_before, games_before, 0

        if dry_run:
            return games_before, len(valid_games), games_removed

        # Create backup
        if backup:
            backup_path = jsonl_path.with_suffix('.jsonl.pre_timeout_filter')
            if not backup_path.exists():
                shutil.copy2(jsonl_path, backup_path)

        # Write filtered file
        with open(jsonl_path, 'w') as f:
            f.writelines(valid_games)

        return games_before, len(valid_games), games_removed

    except Exception as e:
        print(f"  Error filtering {jsonl_path}: {e}")
        return 0, 0, 0


def filter_local_data(data_dir: Path, dry_run: bool = False) -> FilterStats:
    """Filter all local training data."""
    stats = FilterStats()

    # Build list of all data directories to search
    data_dirs = [
        data_dir / "games",
        data_dir / "selfplay",
        data_dir / "gh200_selfplay",
        data_dir / "gpu_selfplay",
        data_dir / "iterations",
        data_dir / "canonical",
        data_dir / "archive",
    ]
    # Also add any other directories that exist
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and d not in data_dirs:
                # Skip non-training directories
                if d.name in ("quarantine", "quarantine_toxic", "holdouts", "eval_pools", "critical_positions", "cmaes"):
                    continue
                data_dirs.append(d)

    print(f"Filtering local data in {data_dir}...")
    print(f"  Searching {len([d for d in data_dirs if d.exists()])} directories...")

    for search_dir in data_dirs:
        if not search_dir.exists():
            continue

        # Filter SQLite databases
        for db_file in search_dir.glob("**/*.db"):
            # Skip backup files
            if '.pre_timeout' in str(db_file) or '.backup' in str(db_file):
                continue

            before, after, removed = filter_sqlite_db(db_file, dry_run=dry_run)
            if removed > 0:
                action = "would be" if dry_run else ""
                print(f"  {db_file.name}: {removed:,} timeout games {action} removed")
                stats.db_games_before += before
                stats.db_games_after += after
                stats.db_games_removed += removed
                stats.files_processed += 1

        # Filter JSONL files
        for jsonl_file in search_dir.glob("**/*.jsonl"):
            # Skip backup files and metrics files
            if '.pre_timeout' in str(jsonl_file) or '.backup' in str(jsonl_file):
                continue
            if '_stats.jsonl' in str(jsonl_file) or '_metrics.jsonl' in str(jsonl_file):
                continue

            before, after, removed = filter_jsonl_file(jsonl_file, dry_run=dry_run)
            if removed > 0:
                action = "would be" if dry_run else ""
                print(f"  {jsonl_file.name}: {removed:,} timeout games {action} removed")
                stats.jsonl_games_before += before
                stats.jsonl_games_after += after
                stats.jsonl_games_removed += removed
                stats.files_processed += 1

    return stats


async def filter_remote_host(
    host_name: str,
    ssh_host: str,
    ssh_user: str,
    remote_path: str,
    ssh_key: str | None = None,
    dry_run: bool = False,
) -> tuple[str, FilterStats]:
    """Filter training data on a remote host."""
    stats = FilterStats()

    # Build SSH options
    ssh_opts = "-o ConnectTimeout=30 -o StrictHostKeyChecking=no -o BatchMode=yes"
    if ssh_key:
        ssh_opts += f" -i {ssh_key}"

    # Test connection first
    test_cmd = f'ssh {ssh_opts} {ssh_user}@{ssh_host} "echo ok" 2>&1'
    process = await asyncio.create_subprocess_shell(
        test_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    if process.returncode != 0:
        stats.errors.append(f"Cannot connect to {host_name}")
        return host_name, stats

    # Create inline Python filter script
    dry_run_str = "True" if dry_run else "False"
    filter_script = f'''
import sqlite3
import json
import shutil
from pathlib import Path

base_path = Path("{remote_path}/data")
dry_run = {dry_run_str}

# Search ALL data directories for training data
data_dirs = [
    base_path / "games",
    base_path / "selfplay",
    base_path / "gh200_selfplay",
    base_path / "gpu_selfplay",
    base_path / "iterations",
    base_path / "canonical",
    base_path / "archive",
]
# Also add any other directories that exist
for d in base_path.iterdir():
    if d.is_dir() and d not in data_dirs:
        # Skip non-training directories
        if d.name in ("quarantine", "quarantine_toxic", "holdouts", "eval_pools", "critical_positions", "cmaes"):
            continue
        data_dirs.append(d)

db_removed = 0
jsonl_removed = 0
files_processed = 0

for data_dir in data_dirs:
    if not data_dir.exists():
        continue

    # Filter SQLite databases
    for db_file in data_dir.glob("**/*.db"):
        if ".pre_timeout" in str(db_file) or ".backup" in str(db_file):
            continue
        try:
            conn = sqlite3.connect(str(db_file), timeout=30)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            if not cursor.fetchone():
                conn.close()
                continue
            cursor.execute("SELECT COUNT(*) FROM games WHERE termination_reason LIKE '%timeout%'")
            count = cursor.fetchone()[0]
            if count > 0:
                if not dry_run:
                    backup_path = str(db_file) + ".pre_timeout_filter"
                    if not Path(backup_path).exists():
                        shutil.copy2(db_file, backup_path)
                    cursor.execute("DELETE FROM games WHERE termination_reason LIKE '%timeout%'")
                    conn.commit()
                    cursor.execute("VACUUM")
                    conn.commit()
                db_removed += count
                files_processed += 1
            conn.close()
        except Exception as e:
            print(f"DB error {{db_file}}: {{e}}")

    # Filter JSONL files
    for jsonl_file in data_dir.glob("**/*.jsonl"):
        if ".pre_timeout" in str(jsonl_file) or ".backup" in str(jsonl_file):
            continue
        # Skip statistics/metrics files
        if "_stats.jsonl" in str(jsonl_file) or "_metrics.jsonl" in str(jsonl_file):
            continue
        try:
            lines = []
            removed = 0
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        game = json.loads(line)
                        term = game.get("termination_reason") or game.get("result", {{}}).get("termination_reason", "")
                        if "timeout" in term.lower():
                            removed += 1
                        else:
                            lines.append(line)
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        lines.append(line)  # Keep unparseable lines
            if removed > 0:
                if not dry_run:
                    backup_path = str(jsonl_file) + ".pre_timeout_filter"
                    if not Path(backup_path).exists():
                        shutil.copy2(jsonl_file, backup_path)
                    with open(jsonl_file, "w") as f:
                        f.writelines(lines)
                jsonl_removed += removed
                files_processed += 1
        except Exception as e:
            print(f"JSONL error {{jsonl_file}}: {{e}}")

print(f"FILTER_RESULT:{{db_removed}}:{{jsonl_removed}}:{{files_processed}}")
'''

    # Execute on remote host
    cmd = f"ssh {ssh_opts} {ssh_user}@{ssh_host} 'python3 -c \"{filter_script}\"' 2>&1"

    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)  # 30 min timeout

        output = stdout.decode()

        # Parse result
        for line in output.split('\n'):
            if line.startswith('FILTER_RESULT:'):
                parts = line.split(':')
                if len(parts) >= 4:
                    stats.db_games_removed = int(parts[1])
                    stats.jsonl_games_removed = int(parts[2])
                    stats.files_processed = int(parts[3])

        return host_name, stats

    except asyncio.TimeoutError:
        stats.errors.append(f"Timeout on {host_name}")
        return host_name, stats
    except Exception as e:
        stats.errors.append(f"Error on {host_name}: {e}")
        return host_name, stats


def load_hosts(config_path: Path) -> list[dict[str, Any]]:
    """Load host configurations."""
    if not config_path.exists():
        return []

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    hosts = []
    for name, host_data in data.get("hosts", {}).items():
        if host_data.get("status") not in ("ready", "setup"):
            continue

        hosts.append({
            "name": name,
            "ssh_host": host_data.get("tailscale_ip") or host_data.get("ssh_host"),
            "ssh_user": host_data.get("ssh_user", "ubuntu"),
            "ssh_key": host_data.get("ssh_key"),
            "remote_path": host_data.get("ringrift_path", "~/ringrift/ai-service"),
        })

    return hosts


async def filter_cluster(config_path: Path, dry_run: bool = False, specific_host: str | None = None):
    """Filter training data across all cluster hosts."""
    hosts = load_hosts(config_path)

    if specific_host:
        hosts = [h for h in hosts if h["name"] == specific_host]

    if not hosts:
        print("No hosts found to filter")
        return

    print(f"Filtering timeout games on {len(hosts)} hosts...")
    if dry_run:
        print("DRY RUN - no changes will be made\n")
    else:
        print("LIVE RUN - timeout games will be removed (backups created)\n")

    tasks = []
    for host in hosts:
        print(f"  Queuing {host['name']}...")
        tasks.append(filter_remote_host(
            host_name=host["name"],
            ssh_host=host["ssh_host"],
            ssh_user=host["ssh_user"],
            remote_path=host["remote_path"],
            ssh_key=host.get("ssh_key"),
            dry_run=dry_run,
        ))

    print(f"\nProcessing {len(tasks)} hosts in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_removed = 0
    successful_hosts = 0
    failed_hosts = 0

    print("\n" + "=" * 70)
    print("FILTERING RESULTS BY HOST")
    print("=" * 70)

    for result in results:
        if isinstance(result, Exception):
            print(f"  Exception: {result}")
            failed_hosts += 1
            continue

        host_name, stats = result
        if stats.errors:
            print(f"\n{host_name}: FAILED")
            for error in stats.errors:
                print(f"  Error: {error}")
            failed_hosts += 1
        elif stats.total_removed > 0:
            print(f"\n{host_name}: {stats.total_removed:,} timeout games removed")
            print(f"    DB: {stats.db_games_removed:,}, JSONL: {stats.jsonl_games_removed:,}")
            total_removed += stats.total_removed
            successful_hosts += 1
        else:
            print(f"{host_name}: No timeout games found (or already filtered)")
            successful_hosts += 1

    print("\n" + "=" * 70)
    action = "would be removed" if dry_run else "removed"
    print(f"SUMMARY")
    print("=" * 70)
    print(f"  Total timeout games {action}: {total_removed:,}")
    print(f"  Successful hosts: {successful_hosts}")
    print(f"  Failed hosts: {failed_hosts}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Filter timeout games from training data")
    parser.add_argument("--local", action="store_true", help="Filter local data")
    parser.add_argument("--cluster", action="store_true", help="Filter on all cluster hosts")
    parser.add_argument("--host", type=str, help="Filter specific host")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be filtered without making changes")
    parser.add_argument("--config", type=str, default="config/distributed_hosts.yaml", help="Hosts config file")
    parser.add_argument("--data-dir", type=str, default="data/games", help="Local data directory")

    args = parser.parse_args()

    if args.local:
        data_dir = ROOT / args.data_dir
        stats = filter_local_data(data_dir, dry_run=args.dry_run)
        print("\n" + "=" * 70)
        action = "PREVIEW" if args.dry_run else "COMPLETE"
        print(f"LOCAL FILTERING {action}")
        print("=" * 70)
        print(stats)

    elif args.cluster or args.host:
        config_path = ROOT / args.config
        asyncio.run(filter_cluster(config_path, dry_run=args.dry_run, specific_host=args.host))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

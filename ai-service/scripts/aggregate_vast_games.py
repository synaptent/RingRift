#!/usr/bin/env python3
"""
Aggregate game data from multiple vast.ai nodes to a central coordinator.
All nodes are connected via Tailscale.
"""

import subprocess
import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Node configuration
NODES = [
    {
        "name": "vast-3060ti-64cpu",
        "ip": "100.117.81.49",
        "paths": [
            "/root/ringrift/ai-service/data/games/selfplay.db",
            "/root/ringrift/ai-service/data/games/games.jsonl"
        ]
    },
    {
        "name": "vast-512cpu",
        "ip": "100.118.201.85",
        "paths": [
            "/workspace/ringrift/ai-service/data/games/selfplay.db",
            "/workspace/ringrift/ai-service/data/games/games.jsonl"
        ]
    },
    {
        "name": "vast-2060s-22cpu",
        "ip": "100.75.98.13",
        "paths": [
            "/root/RingRift/ai-service/data/games/selfplay.db",
            "/root/RingRift/ai-service/data/games/games.jsonl"
        ]
    }
]

COORDINATOR = {
    "name": "vast-rtx4060ti",
    "ip": "100.100.242.64",
    "db_path": "/workspace/ringrift/ai-service/data/games/selfplay.db"
}

SSH_OPTS = "-o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=no"
SSH_USER = "root"

# Local temporary directory for staging
LOCAL_STAGING = "/tmp/vast_game_aggregation"


def run_ssh_command(ip: str, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """Run SSH command on remote node."""
    full_cmd = f"ssh {SSH_OPTS} {SSH_USER}@{ip} '{command}'"
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "SSH command timed out"
    except Exception as e:
        return False, "", str(e)


def check_file_exists(ip: str, path: str) -> Tuple[bool, int]:
    """Check if file exists on remote node and get its size."""
    success, stdout, stderr = run_ssh_command(
        ip,
        f"test -f {path} && stat -c%s {path} || echo 0"
    )
    if success and stdout.strip():
        try:
            size = int(stdout.strip())
            return size > 0, size
        except ValueError:
            return False, 0
    return False, 0


def get_db_game_count(ip: str, db_path: str) -> Tuple[bool, Dict]:
    """Get game count from remote database."""
    query = """
    SELECT
        board_type,
        num_players,
        COUNT(*) as count
    FROM games
    GROUP BY board_type, num_players
    ORDER BY board_type, num_players;
    """

    success, stdout, stderr = run_ssh_command(
        ip,
        f"sqlite3 {db_path} \"{query}\"",
        timeout=60
    )

    if not success:
        return False, {}

    counts = {}
    for line in stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|')
        if len(parts) == 3:
            board_type, num_players, count = parts
            key = f"{board_type}_{num_players}p"
            counts[key] = int(count)

    return True, counts


def copy_file_from_node(ip: str, remote_path: str, local_path: str) -> bool:
    """Copy file from remote node to local staging."""
    cmd = f"scp {SSH_OPTS} {SSH_USER}@{ip}:{remote_path} {local_path}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"Error copying file: {e}")
        return False


def copy_file_to_coordinator(local_path: str, remote_path: str) -> bool:
    """Copy file from local staging to coordinator."""
    cmd = f"scp {SSH_OPTS} {local_path} {SSH_USER}@{COORDINATOR['ip']}:{remote_path}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"Error copying file to coordinator: {e}")
        return False


def merge_database_on_coordinator(source_db: str, target_db: str) -> bool:
    """Merge source database into target database on coordinator."""
    merge_script = f"""
    sqlite3 {target_db} <<'EOF'
ATTACH DATABASE '{source_db}' AS source_db;
INSERT OR IGNORE INTO games
SELECT * FROM source_db.games;
DETACH DATABASE source_db;
EOF
    """

    success, stdout, stderr = run_ssh_command(
        COORDINATOR['ip'],
        merge_script,
        timeout=300
    )

    return success


def ensure_coordinator_db_exists() -> bool:
    """Ensure the target database exists on coordinator with proper schema."""
    create_schema = """
    sqlite3 /workspace/ringrift/ai-service/data/games/selfplay.db <<'EOF'
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    board_type TEXT,
    num_players INTEGER,
    moves TEXT,
    winner INTEGER,
    final_scores TEXT,
    metadata TEXT,
    created_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_board_type ON games(board_type);
CREATE INDEX IF NOT EXISTS idx_num_players ON games(num_players);
CREATE INDEX IF NOT EXISTS idx_created_at ON games(created_at);
EOF
    """

    # First ensure directory exists
    success, _, _ = run_ssh_command(
        COORDINATOR['ip'],
        "mkdir -p /workspace/ringrift/ai-service/data/games"
    )

    if not success:
        return False

    success, stdout, stderr = run_ssh_command(
        COORDINATOR['ip'],
        create_schema,
        timeout=60
    )

    return success


def main():
    print("=" * 80)
    print("VAST.AI GAME DATA AGGREGATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Coordinator: {COORDINATOR['name']} ({COORDINATOR['ip']})")
    print()

    # Create local staging directory
    os.makedirs(LOCAL_STAGING, exist_ok=True)
    print(f"Local staging directory: {LOCAL_STAGING}")
    print()

    # Step 1: Check all nodes for game data
    print("STEP 1: Checking all nodes for game data...")
    print("-" * 80)

    node_data = []
    for node in NODES:
        print(f"\n{node['name']} ({node['ip']}):")
        node_info = {
            "name": node['name'],
            "ip": node['ip'],
            "databases": [],
            "jsonl_files": []
        }

        for path in node['paths']:
            exists, size = check_file_exists(node['ip'], path)
            if exists:
                print(f"  ✓ Found: {path} ({size:,} bytes)")

                if path.endswith('.db'):
                    success, counts = get_db_game_count(node['ip'], path)
                    if success:
                        total = sum(counts.values())
                        print(f"    Games: {total:,}")
                        for key, count in counts.items():
                            print(f"      - {key}: {count:,}")
                        node_info['databases'].append({
                            'path': path,
                            'size': size,
                            'counts': counts,
                            'total': total
                        })
                    else:
                        print(f"    ⚠ Could not query database")
                        node_info['databases'].append({
                            'path': path,
                            'size': size,
                            'counts': {},
                            'total': 0
                        })
                elif path.endswith('.jsonl'):
                    node_info['jsonl_files'].append({
                        'path': path,
                        'size': size
                    })
            else:
                print(f"  ✗ Not found: {path}")

        if node_info['databases'] or node_info['jsonl_files']:
            node_data.append(node_info)

    if not node_data:
        print("\n⚠ No game data found on any nodes!")
        return 1

    # Step 2: Ensure coordinator database exists
    print("\n\nSTEP 2: Ensuring coordinator database exists...")
    print("-" * 80)

    if not ensure_coordinator_db_exists():
        print("✗ Failed to ensure coordinator database exists")
        return 1

    print("✓ Coordinator database ready")

    # Get initial coordinator state
    success, initial_counts = get_db_game_count(
        COORDINATOR['ip'],
        COORDINATOR['db_path']
    )

    if success:
        initial_total = sum(initial_counts.values())
        print(f"Initial games on coordinator: {initial_total:,}")
        for key, count in initial_counts.items():
            print(f"  - {key}: {count:,}")
    else:
        initial_counts = {}
        initial_total = 0
        print("Initial games on coordinator: 0 (new database)")

    # Step 3: Copy and merge databases
    print("\n\nSTEP 3: Copying and merging databases...")
    print("-" * 80)

    merged_count = 0
    for node in node_data:
        for db in node['databases']:
            print(f"\nProcessing {node['name']}: {db['path']}")

            # Copy to local staging
            local_filename = f"{node['name']}_{os.path.basename(db['path'])}"
            local_path = os.path.join(LOCAL_STAGING, local_filename)

            print(f"  Copying to local staging: {local_path}")
            if not copy_file_from_node(node['ip'], db['path'], local_path):
                print(f"  ✗ Failed to copy from {node['name']}")
                continue
            print(f"  ✓ Copied to local staging")

            # Copy to coordinator
            coordinator_staging = f"/tmp/{local_filename}"
            print(f"  Copying to coordinator: {coordinator_staging}")
            if not copy_file_to_coordinator(local_path, coordinator_staging):
                print(f"  ✗ Failed to copy to coordinator")
                continue
            print(f"  ✓ Copied to coordinator")

            # Merge into main database
            print(f"  Merging into {COORDINATOR['db_path']}")
            if not merge_database_on_coordinator(coordinator_staging, COORDINATOR['db_path']):
                print(f"  ✗ Failed to merge database")
                continue
            print(f"  ✓ Merged successfully")

            # Clean up staging file on coordinator
            run_ssh_command(COORDINATOR['ip'], f"rm -f {coordinator_staging}")

            merged_count += 1

    print(f"\n✓ Successfully merged {merged_count} database(s)")

    # Step 4: Report final counts
    print("\n\nSTEP 4: Final game counts on coordinator...")
    print("-" * 80)

    success, final_counts = get_db_game_count(
        COORDINATOR['ip'],
        COORDINATOR['db_path']
    )

    if not success:
        print("✗ Failed to get final counts")
        return 1

    final_total = sum(final_counts.values())
    games_added = final_total - initial_total

    print(f"\nFinal game count: {final_total:,}")
    print(f"Games added: {games_added:,}")
    print("\nBreakdown by board_type and num_players:")
    for key, count in sorted(final_counts.items()):
        print(f"  - {key}: {count:,}")

    # Step 5: Verify integrity
    print("\n\nSTEP 5: Verifying database integrity...")
    print("-" * 80)

    verify_query = "PRAGMA integrity_check;"
    success, stdout, stderr = run_ssh_command(
        COORDINATOR['ip'],
        f"sqlite3 {COORDINATOR['db_path']} '{verify_query}'"
    )

    if success and "ok" in stdout.lower():
        print("✓ Database integrity check passed")
    else:
        print(f"⚠ Database integrity check: {stdout}")

    # Summary
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Coordinator: {COORDINATOR['name']} ({COORDINATOR['ip']})")
    print(f"Database: {COORDINATOR['db_path']}")
    print(f"Total games: {final_total:,}")
    print(f"Games added: {games_added:,}")
    print(f"Nodes processed: {len(node_data)}")
    print(f"Databases merged: {merged_count}")
    print(f"Completed at: {datetime.now().isoformat()}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

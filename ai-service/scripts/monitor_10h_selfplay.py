#!/usr/bin/env python3
"""Monitor 10-hour selfplay deployment and sync data periodically.

This script:
1. Checks node status every 15 minutes
2. Restarts failed jobs automatically
3. Syncs game databases from cluster every 30 minutes
4. Logs progress and estimates completion

Usage:
    python scripts/monitor_10h_selfplay.py              # Run monitor
    python scripts/monitor_10h_selfplay.py --once       # Single check
    python scripts/monitor_10h_selfplay.py --sync-only  # Just sync data
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from spawn script
from scripts.spawn_10h_selfplay import NODES, get_config_for_node, run_ssh, spawn_selfplay

SYNC_DIR = Path(__file__).parent.parent / "data" / "games" / "cluster_sync"
SYNC_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path(__file__).parent.parent / "logs" / "monitor_10h.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_node_status(node) -> dict:
    """Get detailed status from a node."""
    # Check selfplay count
    cmd = "ps aux | grep 'selfplay.py' | grep -v grep | wc -l"
    success, output = run_ssh(node, cmd, timeout=15)

    status = {
        "name": node.name,
        "reachable": success,
        "selfplay_count": 0,
        "games": {},
    }

    if success:
        try:
            status["selfplay_count"] = int(output.strip())
        except (ValueError, AttributeError):
            pass

    # Get game counts
    if success:
        cmd = """cd ~/ringrift/ai-service 2>/dev/null || cd /workspace/ringrift/ai-service 2>/dev/null
for db in data/games/*.db; do
  [ -f "$db" ] || continue
  count=$(sqlite3 "$db" 'SELECT COUNT(*) FROM games' 2>/dev/null || echo 0)
  [ "$count" -gt 0 ] && echo "$(basename $db):$count"
done"""
        success2, output2 = run_ssh(node, cmd, timeout=30)
        if success2:
            for line in output2.strip().split("\n"):
                if ":" in line:
                    db, count = line.split(":")
                    try:
                        status["games"][db] = int(count)
                    except (ValueError, IndexError):
                        pass

    return status


def sync_from_node(node, dry_run: bool = False) -> tuple:
    """Sync game databases from a node."""
    work_dir = node.work_dir.replace("~", str(Path.home()))
    key = node.key.replace("~", str(Path.home()))

    # List remote databases
    cmd = """cd ~/ringrift/ai-service 2>/dev/null || cd /workspace/ringrift/ai-service 2>/dev/null
ls data/games/*.db 2>/dev/null | xargs -I{} basename {}"""
    success, output = run_ssh(node, cmd, timeout=15)

    if not success:
        return False, f"{node.name}: unreachable"

    dbs = [db.strip() for db in output.strip().split("\n") if db.strip().endswith(".db")]
    if not dbs:
        return True, f"{node.name}: no databases"

    synced = 0
    for db in dbs:
        if dry_run:
            log(f"  [DRY-RUN] Would sync {db} from {node.name}")
            continue

        remote_path = f"{work_dir}/data/games/{db}"
        if "workspace" in node.work_dir:
            remote_path = f"/workspace/ringrift/ai-service/data/games/{db}"

        local_path = SYNC_DIR / f"{node.name}_{db}"

        # Use scp (more reliable than rsync on some nodes)
        scp_cmd = [
            "scp", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            "-i", key, "-P", str(node.port),
            f"{node.target}:{remote_path}",
            str(local_path)
        ]

        try:
            result = subprocess.run(scp_cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                synced += 1
        except (subprocess.TimeoutExpired, OSError):
            pass

    return True, f"{node.name}: synced {synced}/{len(dbs)} databases"


def merge_synced_databases():
    """Merge synced databases into canonical databases."""
    canonical_dir = Path(__file__).parent.parent / "data" / "games"
    merged = 0

    for synced_db in SYNC_DIR.glob("*.db"):
        # Parse node name and original db name
        parts = synced_db.name.split("_", 1)
        if len(parts) != 2:
            continue

        node_name, db_name = parts

        # Determine board type and player count from db name
        for board in ["hexagonal", "square19", "square8", "hex8"]:
            for players in [2, 3, 4]:
                if f"{board}_{players}p" in db_name or (board in db_name and f"{players}p" in db_name):
                    canonical_db = canonical_dir / f"canonical_{board}_{players}p.db"
                    if canonical_db.exists():
                        try:
                            # Check if source has games
                            conn = sqlite3.connect(str(synced_db))
                            count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                            conn.close()

                            if count > 0:
                                # Merge
                                conn = sqlite3.connect(str(canonical_db))
                                conn.execute(f"ATTACH DATABASE '{synced_db}' AS source")
                                conn.execute("INSERT OR IGNORE INTO games SELECT * FROM source.games")
                                try:
                                    conn.execute("INSERT OR IGNORE INTO game_moves SELECT * FROM source.game_moves WHERE game_id IN (SELECT game_id FROM source.games)")
                                    conn.execute("INSERT OR IGNORE INTO game_players SELECT * FROM source.game_players WHERE game_id IN (SELECT game_id FROM source.games)")
                                except sqlite3.OperationalError:
                                    pass  # Tables might not exist
                                conn.execute("DETACH DATABASE source")
                                conn.commit()
                                conn.close()
                                merged += 1
                        except Exception as e:
                            log(f"  Error merging {synced_db.name}: {e}")
                    break

    return merged


def check_and_restart_nodes():
    """Check all nodes and restart failed jobs."""
    log("Checking node status...")

    tier_indices = {1: 0, 2: 0, 3: 0}
    restarted = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_node_status, node): node for node in NODES}

        for future in as_completed(futures):
            node = futures[future]
            status = future.result()

            if not status["reachable"]:
                log(f"  {node.name}: UNREACHABLE")
                continue

            if status["selfplay_count"] == 0:
                # Restart selfplay
                board, players, games = get_config_for_node(node.tier, tier_indices[node.tier])
                log(f"  {node.name}: IDLE - restarting {board} {players}p...")
                if spawn_selfplay(node, board, players, games):
                    restarted += 1
            else:
                total_games = sum(status["games"].values())
                log(f"  {node.name}: {status['selfplay_count']} job(s), {total_games} games total")

            tier_indices[node.tier] += 1

    return restarted


def run_sync(dry_run: bool = False):
    """Sync data from all nodes."""
    log("Syncing data from cluster...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(sync_from_node, node, dry_run): node for node in NODES}

        for future in as_completed(futures):
            success, msg = future.result()
            log(f"  {msg}")

    if not dry_run:
        merged = merge_synced_databases()
        log(f"Merged {merged} databases into canonical")


def print_summary():
    """Print current data status."""
    canonical_dir = Path(__file__).parent.parent / "data" / "games"

    log("\n=== Current Canonical Database Status ===")
    for db in sorted(canonical_dir.glob("canonical_*.db")):
        try:
            conn = sqlite3.connect(str(db))
            count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            conn.close()

            config = db.stem.replace("canonical_", "")
            if count < 100:
                status = "CRITICAL"
            elif count < 500:
                status = "LOW"
            else:
                status = "OK"

            log(f"  {config}: {count} games [{status}]")
        except sqlite3.Error:
            pass


def main():
    parser = argparse.ArgumentParser(description="Monitor 10-hour selfplay deployment")
    parser.add_argument("--once", action="store_true", help="Single check only")
    parser.add_argument("--sync-only", action="store_true", help="Just sync data")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually sync")
    args = parser.parse_args()

    log("=" * 60)
    log("10-Hour Selfplay Monitor Started")
    log("=" * 60)

    if args.sync_only:
        run_sync(args.dry_run)
        print_summary()
        return

    if args.once:
        check_and_restart_nodes()
        run_sync(args.dry_run)
        print_summary()
        return

    # Continuous monitoring
    check_interval = 15 * 60  # 15 minutes
    sync_interval = 30 * 60   # 30 minutes
    last_sync = 0

    try:
        while True:
            check_and_restart_nodes()

            # Sync every 30 minutes
            if time.time() - last_sync >= sync_interval:
                run_sync()
                print_summary()
                last_sync = time.time()

            log(f"Next check in {check_interval // 60} minutes...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        log("Monitor stopped by user")


if __name__ == "__main__":
    main()

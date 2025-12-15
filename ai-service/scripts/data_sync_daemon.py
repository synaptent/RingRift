#!/usr/bin/env python3
"""Data Sync Daemon - Continuous data pull from selfplay workers.

This daemon runs on the coordinator and:
1. Polls all configured worker nodes for new selfplay data
2. Uses rsync for efficient incremental sync
3. Merges synced data into a unified training database
4. Emits events for training triggers

Part of the positive feedback loop:
  selfplay → DATA_SYNC → training → evaluation → promotion → model_sync → selfplay

Usage:
    python scripts/data_sync_daemon.py --interval 60  # Sync every 60 seconds
    python scripts/data_sync_daemon.py --once         # Single sync run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.unified_config import get_training_threshold

# Import event bus for pipeline integration
try:
    from app.distributed.data_events import (
        DataEventType,
        DataEvent,
        get_event_bus,
        emit_new_games,
        emit_training_threshold,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    print("[DataSync] Warning: Event bus not available, events will not be emitted")

# Import cluster coordination
try:
    from app.distributed.cluster_coordinator import ClusterCoordinator, TaskRole
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False

# New coordination features: sync_lock, bandwidth management
try:
    from app.coordination import (
        # Sync mutex for rsync coordination
        sync_lock,
        # Bandwidth management
        request_bandwidth,
        release_bandwidth,
        TransferPriority,
    )
    HAS_NEW_COORDINATION = True
except ImportError:
    HAS_NEW_COORDINATION = False

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
SYNCED_DIR = AI_SERVICE_ROOT / "data" / "games" / "synced"
MERGED_DB = AI_SERVICE_ROOT / "data" / "games" / "merged_training.db"

# Worker configuration - Tailscale IPs for reliable connectivity
WORKERS = [
    {"name": "gh200_e", "host": "100.88.176.74", "user": "ubuntu", "path": "~/ringrift/ai-service/data/games"},
    {"name": "gh200_i", "host": "100.99.27.56", "user": "ubuntu", "path": "~/ringrift/ai-service/data/games"},
    # Add more workers as they recover
]

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSH_OPTS = f"-i {SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=no"


def sync_from_worker(worker: Dict, pattern: str = "*.db") -> Tuple[bool, int, str]:
    """Sync database files from a worker node.

    Uses sync_lock and bandwidth management when available for coordinated
    rsync operations across the cluster.

    Returns:
        (success, bytes_transferred, message)
    """
    name = worker["name"]
    host = worker["host"]
    user = worker["user"]
    remote_path = worker["path"]

    local_dir = SYNCED_DIR / name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Use rsync for efficient incremental sync
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh {SSH_OPTS}",
        f"{user}@{host}:{remote_path}/{pattern}",
        str(local_dir) + "/"
    ]

    # Use new coordination if available: sync_lock + bandwidth
    if HAS_NEW_COORDINATION:
        bandwidth_alloc = None
        try:
            # Acquire sync lock to prevent concurrent rsync operations
            with sync_lock(source_host=host, target_host="localhost", operation="data_sync"):
                # Request bandwidth allocation (estimate ~100MB for DB sync)
                bandwidth_alloc = request_bandwidth(
                    source_host=host,
                    target_host="localhost",
                    estimated_bytes=100 * 1024 * 1024,  # 100MB estimate
                    priority=TransferPriority.NORMAL,
                )

                if bandwidth_alloc and not bandwidth_alloc.granted:
                    return False, 0, f"Bandwidth not available for {name}"

                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                transfer_duration = time.time() - start_time

                if result.returncode == 0:
                    # Parse bytes transferred from rsync output
                    bytes_transferred = 0
                    for line in result.stdout.split("\n"):
                        if "total size is" in line:
                            try:
                                bytes_transferred = int(line.split("total size is")[1].split()[0].replace(",", ""))
                            except:
                                pass

                    # Release bandwidth with actual transfer stats
                    if bandwidth_alloc and bandwidth_alloc.granted:
                        release_bandwidth(
                            bandwidth_alloc.allocation_id,
                            bytes_transferred=bytes_transferred,
                            duration_seconds=transfer_duration
                        )

                    return True, bytes_transferred, f"Synced from {name}"
                else:
                    return False, 0, f"rsync failed: {result.stderr[:200]}"

        except subprocess.TimeoutExpired:
            return False, 0, f"Timeout syncing from {name}"
        except Exception as e:
            return False, 0, f"Error: {e}"
        finally:
            # Ensure bandwidth is released even on error
            if bandwidth_alloc and bandwidth_alloc.granted:
                try:
                    release_bandwidth(bandwidth_alloc.allocation_id)
                except Exception:
                    pass
    else:
        # Fallback: no coordination
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Parse bytes transferred from rsync output
                bytes_transferred = 0
                for line in result.stdout.split("\n"):
                    if "total size is" in line:
                        try:
                            bytes_transferred = int(line.split("total size is")[1].split()[0].replace(",", ""))
                        except:
                            pass
                return True, bytes_transferred, f"Synced from {name}"
            else:
                return False, 0, f"rsync failed: {result.stderr[:200]}"

        except subprocess.TimeoutExpired:
            return False, 0, f"Timeout syncing from {name}"
        except Exception as e:
            return False, 0, f"Error: {e}"


def count_games_in_db(db_path: Path) -> int:
    """Count games in a SQLite database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def merge_databases(output_db: Path) -> Tuple[int, int]:
    """Merge all synced databases into a single training database.

    Returns:
        (total_games, new_games)
    """
    # Get existing game count
    existing_games = count_games_in_db(output_db)

    # Find all synced databases
    synced_dbs = list(SYNCED_DIR.rglob("*.db"))

    if not synced_dbs:
        return existing_games, 0

    # Create/connect to merged database
    conn = sqlite3.connect(str(output_db))

    # Ensure games table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            winner INTEGER,
            game_length INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Merge each synced database
    total_new = 0
    for db_path in synced_dbs:
        try:
            # Attach the source database
            conn.execute(f"ATTACH DATABASE '{db_path}' AS source")

            # Get column names from source
            cursor = conn.execute("PRAGMA source.table_info(games)")
            source_columns = [row[1] for row in cursor.fetchall()]

            # Find common columns
            cursor = conn.execute("PRAGMA main.table_info(games)")
            target_columns = [row[1] for row in cursor.fetchall()]
            common_columns = [c for c in source_columns if c in target_columns]

            if "game_id" in common_columns:
                # Insert new games (ignore duplicates)
                cols = ", ".join(common_columns)
                conn.execute(f"""
                    INSERT OR IGNORE INTO games ({cols})
                    SELECT {cols} FROM source.games
                """)
                total_new += conn.total_changes

            conn.execute("DETACH DATABASE source")

        except Exception as e:
            print(f"  Warning: Could not merge {db_path.name}: {e}")
            try:
                conn.execute("DETACH DATABASE source")
            except:
                pass

    conn.commit()

    # Get final count
    final_games = count_games_in_db(output_db)
    new_games = final_games - existing_games

    conn.close()
    return final_games, new_games


def run_sync_cycle() -> Dict:
    """Run one sync cycle from all workers.

    Returns:
        Summary dictionary with sync results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "workers": {},
        "total_bytes": 0,
        "total_games_before": count_games_in_db(MERGED_DB),
        "total_games_after": 0,
        "new_games": 0,
    }

    print(f"\n[DataSync] Starting sync cycle at {results['timestamp']}")

    # Sync from each worker
    for worker in WORKERS:
        name = worker["name"]
        print(f"  Syncing from {name}...")

        success, bytes_transferred, message = sync_from_worker(worker)
        results["workers"][name] = {
            "success": success,
            "bytes": bytes_transferred,
            "message": message,
        }
        results["total_bytes"] += bytes_transferred

        if success:
            print(f"    ✓ {message} ({bytes_transferred:,} bytes)")
        else:
            print(f"    ✗ {message}")

    # Merge databases
    print("  Merging databases...")
    total_games, new_games = merge_databases(MERGED_DB)
    results["total_games_after"] = total_games
    results["new_games"] = new_games

    print(f"  ✓ Merged: {total_games} total games ({new_games} new)")

    # Emit NEW_GAMES_AVAILABLE event if we have new games
    if HAS_EVENT_BUS and new_games > 0:
        import asyncio
        try:
            asyncio.run(emit_new_games(
                host="merged",
                new_games=new_games,
                total_games=total_games,
                source="data_sync_daemon",
            ))
            print(f"  📡 Emitted NEW_GAMES_AVAILABLE event ({new_games} new games)")
        except Exception as e:
            print(f"  ⚠ Failed to emit event: {e}")

    # Check if training threshold reached (from unified config)
    training_threshold = get_training_threshold()
    if total_games >= training_threshold:
        print(f"  📊 Training threshold reached: {total_games} >= {training_threshold}")
        # Emit training trigger event
        if HAS_EVENT_BUS:
            import asyncio
            try:
                asyncio.run(emit_training_threshold(
                    config="all",  # All configs since this is merged database
                    games=total_games,
                    source="data_sync_daemon",
                ))
                print(f"  📡 Emitted TRAINING_THRESHOLD_REACHED event")
            except Exception as e:
                print(f"  ⚠ Failed to emit threshold event: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Data Sync Daemon")
    parser.add_argument("--interval", type=int, default=60, help="Sync interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    SYNCED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DataSync] Starting daemon")
    print(f"[DataSync] Workers: {[w['name'] for w in WORKERS]}")
    print(f"[DataSync] Interval: {args.interval}s")
    print(f"[DataSync] Output: {MERGED_DB}")

    if args.once:
        run_sync_cycle()
        return

    # Daemon loop
    while True:
        try:
            run_sync_cycle()
        except Exception as e:
            print(f"[DataSync] Error in sync cycle: {e}")

        print(f"[DataSync] Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

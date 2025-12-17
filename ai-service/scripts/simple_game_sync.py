#!/usr/bin/env python3
"""Simple game database sync using rsync.

Periodically pulls game databases from remote hosts and merges them
into the central selfplay.db database.

Usage:
    python scripts/simple_game_sync.py --daemon --interval 300
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Use shared modules from app/
from app.distributed.hosts import HostConfig, load_remote_hosts
from app.distributed.sync_utils import rsync_file
from app.execution.executor import run_ssh_command_sync

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("simple_game_sync", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent

# Local paths
LOCAL_GAMES_DIR = AI_SERVICE_ROOT / "data" / "games"
LOCAL_SELFPLAY_DB = LOCAL_GAMES_DIR / "selfplay.db"
SYNC_TEMP_DIR = AI_SERVICE_ROOT / "data" / "sync_temp"

SSH_TIMEOUT = 15


def load_active_hosts() -> List[HostConfig]:
    """Load active hosts from distributed_hosts.yaml using shared module."""
    all_hosts = load_remote_hosts()
    active_hosts = []

    for host in all_hosts.values():
        # Skip stopped/disabled hosts by checking properties
        status = host.properties.get('status', 'active')
        if status in ('stopped', 'disabled', 'setup', 'unstable'):
            continue
        active_hosts.append(host)

    logger.info(f"Loaded {len(active_hosts)} active hosts from config")
    return active_hosts


def run_ssh(host: HostConfig, cmd: str, timeout: int = SSH_TIMEOUT) -> Optional[str]:
    """Run SSH command and return output using shared executor."""
    result = run_ssh_command_sync(
        host=host.ssh_host,
        command=cmd,
        user=host.ssh_user,
        port=host.ssh_port,
        key_path=host.ssh_key_path if host.ssh_key else None,
        timeout=timeout,
    )
    if result.success:
        return result.stdout.strip()
    return None


def find_remote_dbs(host: HostConfig) -> List[str]:
    """Find all game databases on a remote host."""
    dbs = []
    work_dir = host.work_directory

    # Find all .db files with games
    cmd = f"cd {work_dir} && find data -name '*.db' -size +10k 2>/dev/null | head -50"
    result = run_ssh(host, cmd, timeout=30)

    if result:
        for line in result.strip().split('\n'):
            if line and '.db' in line:
                dbs.append(line.strip())

    return dbs


def get_game_ids(db_path: Path) -> Set[str]:
    """Get all game IDs from a database."""
    if not db_path.exists():
        return set()

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        cursor = conn.execute("SELECT game_id FROM games")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return ids
    except Exception:
        return set()


def merge_database(source_db: Path, target_db: Path) -> Tuple[int, int]:
    """Merge games AND game_moves from source into target. Returns (new_games, total_games)."""
    if not source_db.exists():
        return 0, 0

    try:
        # Get existing game IDs in target
        existing_ids = get_game_ids(target_db)

        # Connect to source
        src_conn = sqlite3.connect(str(source_db), timeout=10)
        src_cursor = src_conn.cursor()

        # Check if games table exists
        src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not src_cursor.fetchone():
            src_conn.close()
            return 0, 0

        # Get games not in target
        src_cursor.execute("SELECT * FROM games")
        columns = [desc[0] for desc in src_cursor.description]

        new_games = []
        new_game_ids = []
        for row in src_cursor.fetchall():
            game_dict = dict(zip(columns, row))
            game_id = game_dict.get('game_id')
            if game_id and game_id not in existing_ids:
                new_games.append(row)
                new_game_ids.append(game_id)

        if not new_games:
            src_conn.close()
            return 0, len(existing_ids)

        # Get moves for new games
        new_moves = []
        moves_columns = None
        src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
        if src_cursor.fetchone() and new_game_ids:
            placeholders = ','.join(['?' for _ in new_game_ids])
            src_cursor.execute(f"SELECT * FROM game_moves WHERE game_id IN ({placeholders})", new_game_ids)
            moves_columns = [desc[0] for desc in src_cursor.description]
            new_moves = src_cursor.fetchall()

        # Get choices for new games
        new_choices = []
        choices_columns = None
        src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_choices'")
        if src_cursor.fetchone() and new_game_ids:
            placeholders = ','.join(['?' for _ in new_game_ids])
            src_cursor.execute(f"SELECT * FROM game_choices WHERE game_id IN ({placeholders})", new_game_ids)
            choices_columns = [desc[0] for desc in src_cursor.description]
            new_choices = src_cursor.fetchall()

        src_conn.close()

        # Connect to target and insert
        tgt_conn = sqlite3.connect(str(target_db), timeout=30)
        tgt_cursor = tgt_conn.cursor()

        # Ensure games table exists
        tgt_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not tgt_cursor.fetchone():
            tgt_cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT,
                    num_players INTEGER,
                    winner INTEGER,
                    final_scores TEXT,
                    move_count INTEGER,
                    game_length_seconds REAL,
                    created_at TEXT,
                    config TEXT
                )
            """)

        # Ensure game_moves table exists
        tgt_cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_moves (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                turn_number INTEGER,
                player INTEGER,
                phase TEXT,
                move_type TEXT,
                move_json TEXT,
                PRIMARY KEY (game_id, move_number)
            )
        """)

        # Ensure game_choices table exists
        tgt_cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_choices (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                player INTEGER,
                legal_moves_json TEXT,
                chosen_move_idx INTEGER,
                PRIMARY KEY (game_id, move_number)
            )
        """)

        # Insert new games
        placeholders = ','.join(['?' for _ in columns])
        col_names = ','.join(columns)
        for row in new_games:
            try:
                tgt_cursor.execute(
                    f"INSERT OR IGNORE INTO games ({col_names}) VALUES ({placeholders})",
                    row
                )
            except sqlite3.OperationalError:
                pass

        # Insert game_moves
        if new_moves and moves_columns:
            placeholders = ','.join(['?' for _ in moves_columns])
            col_names = ','.join(moves_columns)
            for row in new_moves:
                try:
                    tgt_cursor.execute(
                        f"INSERT OR IGNORE INTO game_moves ({col_names}) VALUES ({placeholders})",
                        row
                    )
                except sqlite3.OperationalError:
                    pass

        # Insert game_choices
        if new_choices and choices_columns:
            placeholders = ','.join(['?' for _ in choices_columns])
            col_names = ','.join(choices_columns)
            for row in new_choices:
                try:
                    tgt_cursor.execute(
                        f"INSERT OR IGNORE INTO game_choices ({col_names}) VALUES ({placeholders})",
                        row
                    )
                except sqlite3.OperationalError:
                    pass

        tgt_conn.commit()
        tgt_conn.close()

        moves_info = f" (+{len(new_moves)} moves)" if new_moves else ""
        logger.debug(f"Merged {len(new_games)} games{moves_info}")

        return len(new_games), len(existing_ids) + len(new_games)

    except Exception as e:
        logger.error(f"Merge error for {source_db}: {e}")
        return 0, 0


def sync_from_host(host: HostConfig) -> Tuple[int, int]:
    """Sync all game databases from a single host. Returns (new_games, dbs_synced)."""
    logger.info(f"Syncing from {host.name} ({host.ssh_host})...")

    # Check if host is reachable
    if not run_ssh(host, "echo ok"):
        logger.warning(f"  {host.name} unreachable")
        return 0, 0

    # Find databases on remote host
    remote_dbs = find_remote_dbs(host)
    if not remote_dbs:
        logger.info(f"  {host.name}: no databases found")
        return 0, 0

    logger.info(f"  {host.name}: found {len(remote_dbs)} databases")

    total_new = 0
    dbs_synced = 0
    work_dir = host.work_directory

    for remote_db in remote_dbs[:20]:  # Limit to 20 dbs per host
        # Download to temp location
        temp_path = SYNC_TEMP_DIR / host.name / Path(remote_db).name

        if rsync_file(host, f"{work_dir}/{remote_db}", temp_path):
            # Merge into main database
            new_games, total = merge_database(temp_path, LOCAL_SELFPLAY_DB)
            if new_games > 0:
                logger.info(f"    {remote_db}: +{new_games} games")
                total_new += new_games
                dbs_synced += 1

            # Clean up temp file
            try:
                temp_path.unlink()
            except:
                pass

    return total_new, dbs_synced


def run_sync_cycle() -> Tuple[int, int, int]:
    """Run one sync cycle. Returns (new_games, hosts_synced, dbs_synced)."""
    SYNC_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load hosts using shared module
    hosts = load_active_hosts()
    if not hosts:
        logger.warning("No hosts loaded from config")
        return 0, 0, 0

    total_new = 0
    hosts_synced = 0
    total_dbs = 0

    for host in hosts:
        new_games, dbs = sync_from_host(host)
        if new_games > 0:
            total_new += new_games
            hosts_synced += 1
            total_dbs += dbs

    return total_new, hosts_synced, total_dbs


def run_daemon(interval: int = 300):
    """Run sync daemon."""
    logger.info(f"Starting game sync daemon (interval: {interval}s)")

    while True:
        try:
            start = time.time()
            new_games, hosts, dbs = run_sync_cycle()
            duration = time.time() - start

            if new_games > 0:
                logger.info(f"Cycle complete: +{new_games} games from {hosts} hosts ({dbs} dbs) in {duration:.1f}s")
            else:
                logger.info(f"Cycle complete: no new games ({duration:.1f}s)")

        except KeyboardInterrupt:
            logger.info("Daemon stopped")
            break
        except Exception as e:
            logger.error(f"Sync error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Simple game database sync")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=300, help="Sync interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")

    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.interval)
    else:
        new_games, hosts, dbs = run_sync_cycle()
        print(f"Synced {new_games} new games from {hosts} hosts ({dbs} databases)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simple game database sync using rsync.

Periodically pulls game databases from remote hosts and merges them
into the central selfplay.db database.

Usage:
    python scripts/simple_game_sync.py --daemon --interval 300
"""

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"

# Local paths
LOCAL_GAMES_DIR = AI_SERVICE_ROOT / "data" / "games"
LOCAL_SELFPLAY_DB = LOCAL_GAMES_DIR / "selfplay.db"
SYNC_TEMP_DIR = AI_SERVICE_ROOT / "data" / "sync_temp"

SSH_TIMEOUT = 15


def load_hosts_from_config() -> List[Dict]:
    """Load active hosts from distributed_hosts.yaml."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found: {CONFIG_PATH}")
        return []

    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        hosts = []
        for name, cfg in config.get('hosts', {}).items():
            # Skip stopped/disabled hosts
            if cfg.get('status') in ('stopped', 'disabled', 'setup', 'unstable'):
                continue

            # Prefer Tailscale IP for reliable connectivity, fall back to ssh_host
            host_ip = cfg.get('tailscale_ip') or cfg.get('ssh_host')
            if not host_ip:
                continue

            hosts.append({
                'name': name,
                'ip': host_ip,
                'user': cfg.get('ssh_user', 'ubuntu'),
                'ssh_key': cfg.get('ssh_key'),
                'ssh_port': cfg.get('ssh_port', 22),
                'ringrift_path': cfg.get('ringrift_path', '~/ringrift/ai-service'),
            })

        logger.info(f"Loaded {len(hosts)} active hosts from config")
        return hosts
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return []


def run_ssh(host_cfg: Dict, cmd: str, timeout: int = SSH_TIMEOUT) -> Optional[str]:
    """Run SSH command and return output."""
    try:
        ssh_args = [
            "ssh",
            "-o", f"ConnectTimeout={timeout}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
        ]

        # Add SSH key if specified
        if host_cfg.get('ssh_key'):
            key_path = os.path.expanduser(host_cfg['ssh_key'])
            ssh_args.extend(["-i", key_path])

        # Add port if non-standard
        if host_cfg.get('ssh_port', 22) != 22:
            ssh_args.extend(["-p", str(host_cfg['ssh_port'])])

        # Build host string
        host_str = f"{host_cfg['user']}@{host_cfg['ip']}"
        ssh_args.extend([host_str, cmd])

        result = subprocess.run(
            ssh_args,
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def rsync_file(host_cfg: Dict, remote_path: str, local_path: Path) -> bool:
    """Rsync a single file from remote host."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        rsync_args = ["rsync", "-az", "--timeout=60"]

        # Build SSH command with options
        ssh_cmd = "ssh -o StrictHostKeyChecking=no -o BatchMode=yes"
        if host_cfg.get('ssh_key'):
            key_path = os.path.expanduser(host_cfg['ssh_key'])
            ssh_cmd += f" -i {key_path}"
        if host_cfg.get('ssh_port', 22) != 22:
            ssh_cmd += f" -p {host_cfg['ssh_port']}"

        rsync_args.extend(["-e", ssh_cmd])

        # Build source path
        host_str = f"{host_cfg['user']}@{host_cfg['ip']}"
        rsync_args.extend([f"{host_str}:{remote_path}", str(local_path)])

        result = subprocess.run(
            rsync_args,
            capture_output=True, timeout=120
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Rsync failed for {host_cfg['name']}:{remote_path}: {e}")
        return False


def find_remote_dbs(host_cfg: Dict) -> List[str]:
    """Find all game databases on a remote host."""
    dbs = []
    work_dir = host_cfg.get('ringrift_path', '~/ringrift/ai-service')

    # Find all .db files with games
    cmd = f"cd {work_dir} && find data -name '*.db' -size +10k 2>/dev/null | head -50"
    result = run_ssh(host_cfg, cmd, timeout=30)

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
    """Merge games from source into target. Returns (new_games, total_games)."""
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
        for row in src_cursor.fetchall():
            game_dict = dict(zip(columns, row))
            game_id = game_dict.get('game_id')
            if game_id and game_id not in existing_ids:
                new_games.append(row)

        src_conn.close()

        if not new_games:
            return 0, len(existing_ids)

        # Connect to target and insert new games
        tgt_conn = sqlite3.connect(str(target_db), timeout=30)
        tgt_cursor = tgt_conn.cursor()

        # Ensure table exists with same schema
        tgt_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not tgt_cursor.fetchone():
            # Create table with standard schema
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
                # Column mismatch, try inserting just core columns
                pass

        tgt_conn.commit()
        tgt_conn.close()

        return len(new_games), len(existing_ids) + len(new_games)

    except Exception as e:
        logger.error(f"Merge error for {source_db}: {e}")
        return 0, 0


def sync_from_host(host_cfg: Dict) -> Tuple[int, int]:
    """Sync all game databases from a single host. Returns (new_games, dbs_synced)."""
    host_name = host_cfg['name']
    logger.info(f"Syncing from {host_name} ({host_cfg['ip']})...")

    # Check if host is reachable
    if not run_ssh(host_cfg, "echo ok"):
        logger.warning(f"  {host_name} unreachable")
        return 0, 0

    # Find databases on remote host
    remote_dbs = find_remote_dbs(host_cfg)
    if not remote_dbs:
        logger.info(f"  {host_name}: no databases found")
        return 0, 0

    logger.info(f"  {host_name}: found {len(remote_dbs)} databases")

    total_new = 0
    dbs_synced = 0
    work_dir = host_cfg.get('ringrift_path', '~/ringrift/ai-service')

    for remote_db in remote_dbs[:20]:  # Limit to 20 dbs per host
        # Download to temp location
        temp_path = SYNC_TEMP_DIR / host_name / Path(remote_db).name

        if rsync_file(host_cfg, f"{work_dir}/{remote_db}", temp_path):
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

    # Load hosts from config
    hosts = load_hosts_from_config()
    if not hosts:
        logger.warning("No hosts loaded from config")
        return 0, 0, 0

    total_new = 0
    hosts_synced = 0
    total_dbs = 0

    for host_cfg in hosts:
        new_games, dbs = sync_from_host(host_cfg)
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

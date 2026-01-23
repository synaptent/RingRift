#!/usr/bin/env python3
"""
Elo Database Synchronization Service

Keeps unified_elo.db synchronized across all cluster nodes.
Multi-transport support: Tailscale (preferred), aria2 (parallel), HTTP fallback.

Architecture:
- The coordinator (configured in distributed_hosts.yaml) is the authoritative Elo source
- Each node runs this script as a daemon to push new matches
- Coordinator merges and recalculates Elo ratings
- Updated database is distributed back to all nodes via multi-transport

Transport Priority:
1. Tailscale direct (fastest for mesh network)
2. aria2 parallel download (for large dbs, multiple sources)
3. HTTP direct (fallback)

Usage:
    # Run as sync daemon (worker mode) - auto-discovers coordinator:
    python elo_db_sync.py --mode worker --interval 300

    # Run as coordinator (receives and merges):
    python elo_db_sync.py --mode coordinator --port 8766

    # One-time sync (pull latest from coordinator):
    python elo_db_sync.py --mode pull

    # One-time sync (push local changes to coordinator):
    python elo_db_sync.py --mode push

    # Cluster-wide sync (push to all reachable nodes):
    python elo_db_sync.py --mode cluster-sync

    # Check sync status across cluster:
    python elo_db_sync.py --mode status
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Jan 13, 2026: Import harness extraction for preserving harness_type in sync
try:
    from app.training.composite_participant import extract_harness_type
except ImportError:
    # Fallback if import fails (e.g., on minimal nodes)
    def extract_harness_type(participant_id: str) -> str | None:
        """Extract harness type from composite participant ID (model:harness:config)."""
        if not participant_id or ":" not in participant_id:
            return None
        parts = participant_id.split(":")
        if len(parts) >= 2:
            return parts[1]
        return None

DEFAULT_DB_PATH = ROOT / "data" / "unified_elo.db"
SYNC_STATE_FILE = ROOT / "data" / "elo_sync_state.json"
HOSTS_CONFIG = ROOT / "config" / "distributed_hosts.yaml"

# Default coordinator fallback (if config unavailable)
DEFAULT_COORDINATOR_HOST = "macbook-pro"
DEFAULT_PORT = 8766

# Transport timeouts
TAILSCALE_TIMEOUT = 10
HTTP_TIMEOUT = 30
ARIA2_TIMEOUT = 120


# ============================================
# Safe Database Operations (WAL-aware)
# ============================================

def safe_replace_wal_database(source_path: Path, dest_path: Path, backup: bool = True) -> bool:
    """Safely replace a WAL-mode SQLite database.

    SQLite WAL mode uses 3 files: .db, .db-wal, .db-shm
    All must be handled together to prevent corruption.

    Args:
        source_path: Path to source database file
        dest_path: Path to destination database file
        backup: Whether to create a backup of existing destination

    Returns:
        True if replacement succeeded, False otherwise
    """
    wal_path = Path(str(dest_path) + '-wal')
    shm_path = Path(str(dest_path) + '-shm')

    try:
        # Create backup if requested
        if backup and dest_path.exists():
            backup_path = Path(str(dest_path) + '.backup')
            shutil.copy(dest_path, backup_path)

        # Delete stale WAL/SHM files BEFORE replacing main DB
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()

        # Replace main DB atomically when possible
        if source_path != dest_path:
            os.replace(source_path, dest_path)

        # Verify integrity
        conn = sqlite3.connect(str(dest_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        if result != "ok":
            print(f"Warning: Database integrity check returned: {result}")
            return False

        return True

    except Exception as e:
        print(f"Error replacing database: {e}")
        return False


def checkpoint_wal_database(db_path: Path) -> bool:
    """Checkpoint WAL to ensure all data is in main database file.

    This should be called before copying/syncing a WAL-mode database.

    Args:
        db_path: Path to database file

    Returns:
        True if checkpoint succeeded, False otherwise
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        return True
    except Exception as e:
        print(f"Warning: WAL checkpoint failed: {e}")
        return False


# ============================================
# Host Configuration Loading
# ============================================

# Try to use unified hosts module
try:
    from scripts.lib.hosts import get_elo_sync_config, get_host, get_hosts
    USE_UNIFIED_HOSTS = True
except ImportError:
    USE_UNIFIED_HOSTS = False


def load_hosts_config() -> dict[str, Any]:
    """Load cluster hosts from distributed_hosts.yaml."""
    # Prefer unified hosts module
    if USE_UNIFIED_HOSTS:
        hosts = {}
        for h in get_hosts():
            hosts[h.name] = {
                'ssh_host': h.ssh_host,
                'tailscale_ip': h.tailscale_ip,
                'ssh_user': h.ssh_user,
                'status': h.status,
                'role': h.role,
            }
        return hosts

    # Fallback to direct YAML loading
    if not HOSTS_CONFIG.exists():
        return {}

    try:
        import yaml
        with open(HOSTS_CONFIG) as f:
            config = yaml.safe_load(f)
        return config.get('hosts', {})
    except ImportError:
        # Fallback: parse YAML manually for basic fields
        hosts = {}
        current_host = None
        with open(HOSTS_CONFIG) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('  ') and ':' in line and not line.startswith('    '):
                    # Host name line
                    current_host = line.strip().rstrip(':')
                    hosts[current_host] = {}
                elif current_host and line.startswith('    ') and ':' in line:
                    # Property line
                    key, _, value = line.strip().partition(':')
                    value = value.strip().strip('"\'')
                    if value.isdigit():
                        value = int(value)
                    hosts[current_host][key] = value
        return hosts
    except Exception as e:
        print(f"Warning: Could not load hosts config: {e}")
        return {}


def get_coordinator_address() -> tuple[str, int]:
    """Get coordinator address from config or use environment variable.

    Priority:
    1. Environment variable ELO_COORDINATOR_IP
    2. Unified hosts module (elo_sync config)
    3. Config file (elo_sync.coordinator)
    4. Error if none is set
    """
    # Check environment variable first
    env_ip = os.environ.get('ELO_COORDINATOR_IP')
    if env_ip:
        return env_ip, DEFAULT_PORT

    # Prefer unified hosts module
    if USE_UNIFIED_HOSTS:
        try:
            elo_config = get_elo_sync_config()
            if elo_config and elo_config.coordinator:
                coord_host = get_host(elo_config.coordinator)
                if coord_host:
                    ip = coord_host.tailscale_ip or coord_host.ssh_host
                    if ip:
                        return ip, elo_config.sync_port
        except Exception as e:
            print(f"[Script] Warning: Error getting coordinator from unified hosts: {e}")

    # Fallback to direct config loading
    hosts = load_hosts_config()
    if not hosts:
        print("[Script] Warning: No config found and ELO_COORDINATOR_IP not set")
        return None, DEFAULT_PORT

    # Get coordinator from config
    if not HOSTS_CONFIG.exists():
        print("[Script] Warning: Config file missing and ELO_COORDINATOR_IP not set")
        return None, DEFAULT_PORT

    try:
        import yaml
        with open(HOSTS_CONFIG) as f:
            config = yaml.safe_load(f) or {}

        coordinator_name = config.get('elo_sync', {}).get('coordinator')
        if not coordinator_name:
            print("[Script] Warning: No coordinator specified in config and ELO_COORDINATOR_IP not set")
            return None, DEFAULT_PORT

        # Get coordinator host info
        coord_host = hosts.get(coordinator_name)
        if not coord_host:
            print(f"[Script] Warning: Coordinator '{coordinator_name}' not found in hosts")
            return None, DEFAULT_PORT

        # Prefer Tailscale IP
        ip = coord_host.get('tailscale_ip') or coord_host.get('ssh_host')
        if not ip:
            print(f"[Script] Warning: No IP found for coordinator '{coordinator_name}'")
            return None, DEFAULT_PORT

        port = config.get('elo_sync', {}).get('sync_port', DEFAULT_PORT)
        return ip, port

    except Exception as e:
        print(f"[Script] Error loading coordinator from config: {e}")
        print("[Script] Set ELO_COORDINATOR_IP environment variable to override")
        return None, DEFAULT_PORT


def get_all_sync_targets() -> list[dict[str, Any]]:
    """Get all nodes that should receive Elo sync."""
    hosts = load_hosts_config()
    targets = []

    for name, config in hosts.items():
        if config.get('status') == 'terminated':
            continue

        # Get best IP (prefer Tailscale)
        ip = config.get('tailscale_ip') or config.get('ssh_host')
        if not ip:
            continue

        targets.append({
            'name': name,
            'ip': ip,
            'port': config.get('worker_port', DEFAULT_PORT),
            'tailscale_ip': config.get('tailscale_ip'),
            'ssh_host': config.get('ssh_host'),
        })

    return targets


def check_node_reachable(ip: str, port: int = DEFAULT_PORT, timeout: int = 5) -> bool:
    """Check if a node's sync endpoint is reachable."""
    try:
        url = f"http://{ip}:{port}/status"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (ConnectionError, TimeoutError, urllib.error.URLError, OSError):
        return False


# ============================================
# Multi-Transport Support
# ============================================

def check_aria2_available() -> bool:
    """Check if aria2c is available for parallel downloads."""
    return shutil.which("aria2c") is not None


def pull_with_aria2(sources: list[str], output_path: Path, timeout: int = ARIA2_TIMEOUT) -> bool:
    """Pull database using aria2 with multiple sources for parallel download."""
    if not sources:
        return False

    # Create temp file for download
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / "unified_elo.db"

    cmd = [
        "aria2c",
        "--max-connection-per-server", "16",
        "--split", "4",
        "--min-split-size", "1M",
        "--timeout", str(timeout),
        "--connect-timeout", "10",
        "--max-tries", "3",
        "--auto-file-renaming", "false",
        "--allow-overwrite", "true",
        "--dir", str(temp_dir),
        "--out", "unified_elo.db",
        "--quiet", "true",
    ]
    cmd.extend(sources)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)

        if result.returncode == 0 and temp_file.exists():
            # Verify the database
            conn = sqlite3.connect(str(temp_file))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM match_history")
            count = cursor.fetchone()[0]
            conn.close()

            if count > 0:
                shutil.copy(temp_file, output_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return True

        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"aria2 download failed: {e}")
        return False


def get_node_db_url(ip: str, port: int = DEFAULT_PORT) -> str:
    """Get URL for a node's Elo database."""
    return f"http://{ip}:{port}/db"


def discover_available_sources(timeout: int = 5) -> list[str]:
    """Discover all available Elo database sources in the cluster."""
    targets = get_all_sync_targets()
    available = []

    def check_source(target):
        ip = target['ip']
        port = target['port']
        if check_node_reachable(ip, port, timeout):
            return get_node_db_url(ip, port)
        return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_source, t): t for t in targets}
        for future in as_completed(futures, timeout=timeout + 5):
            try:
                url = future.result()
                if url:
                    available.append(url)
            except (ConnectionError, TimeoutError, RuntimeError):
                pass

    return available


def get_db_hash(db_path: Path) -> str:
    """Get hash of match_history for change detection."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MAX(timestamp) FROM match_history")
    count, max_ts = cursor.fetchone()
    conn.close()
    return hashlib.md5(f"{count}:{max_ts}".encode()).hexdigest()


def get_new_matches(db_path: Path, since_timestamp: float) -> list[dict]:
    """Get matches added since a timestamp."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM match_history
        WHERE timestamp > ?
        ORDER BY timestamp
    """, (since_timestamp,))
    matches = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return matches


def get_last_sync_timestamp() -> float:
    """Get timestamp of last sync."""
    if SYNC_STATE_FILE.exists():
        with open(SYNC_STATE_FILE) as f:
            state = json.load(f)
            return state.get('last_sync_timestamp', 0)
    return 0


def save_sync_timestamp(ts: float):
    """Save timestamp of last sync."""
    state = {'last_sync_timestamp': ts, 'synced_at': datetime.now().isoformat()}
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, 'w') as f:
        json.dump(state, f)


def calculate_elo(winner_elo: float, loser_elo: float, k: float = 32, draw: bool = False) -> tuple[float, float]:
    """Calculate new Elo ratings after a match."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner

    if draw:
        score_winner, score_loser = 0.5, 0.5
    else:
        score_winner, score_loser = 1.0, 0.0

    new_winner = winner_elo + k * (score_winner - expected_winner)
    new_loser = loser_elo + k * (score_loser - expected_loser)
    return new_winner, new_loser


def merge_matches_into_db(db_path: Path, new_matches: list[dict]):
    """Merge new matches into database and recalculate Elo.

    Uses explicit transaction to ensure atomicity - if anything fails,
    the entire batch is rolled back to prevent database corruption.
    """
    if not new_matches:
        return 0

    conn = sqlite3.connect(db_path)
    conn.isolation_level = None  # Manual transaction control
    cursor = conn.cursor()
    inserted = 0

    try:
        cursor.execute("BEGIN IMMEDIATE")  # Lock DB for writes

        # Get existing game_ids to avoid duplicates
        cursor.execute("SELECT game_id FROM match_history WHERE game_id IS NOT NULL")
        existing_game_ids = {row[0] for row in cursor.fetchall()}

        for match in new_matches:
            game_id = match.get('game_id')
            if game_id and game_id in existing_game_ids:
                continue

            # Jan 13, 2026: Extract harness_type from participant_id or use provided value
            harness_type = match.get('harness_type')
            if not harness_type:
                harness_type = extract_harness_type(match.get('participant_a', ''))

            cursor.execute("""
                INSERT INTO match_history
                (participant_a, participant_b, board_type, num_players, winner,
                 game_length, duration_sec, timestamp, tournament_id, game_id, metadata, worker, harness_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match['participant_a'], match['participant_b'], match['board_type'],
                match['num_players'], match.get('winner'), match.get('game_length'),
                match.get('duration_sec'), match['timestamp'], match.get('tournament_id'),
                match.get('game_id'), match.get('metadata'), match.get('worker'), harness_type
            ))

            # Update Elo ratings
            p_a, p_b = match['participant_a'], match['participant_b']
            winner = match.get('winner')
            board_type = match['board_type']
            num_players = match['num_players']

            # Get current ratings
            cursor.execute("""
                SELECT rating FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (p_a, board_type, num_players))
            row = cursor.fetchone()
            elo_a = row[0] if row else 1500.0

            cursor.execute("""
                SELECT rating FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (p_b, board_type, num_players))
            row = cursor.fetchone()
            elo_b = row[0] if row else 1500.0

            # Calculate new ratings
            if winner == p_a:
                new_a, new_b = calculate_elo(elo_a, elo_b)
                win_a, loss_a, draw_a = 1, 0, 0
                win_b, loss_b, draw_b = 0, 1, 0
            elif winner == p_b:
                new_b, new_a = calculate_elo(elo_b, elo_a)
                win_a, loss_a, draw_a = 0, 1, 0
                win_b, loss_b, draw_b = 1, 0, 0
            else:
                new_a, new_b = calculate_elo(elo_a, elo_b, draw=True)
                win_a, loss_a, draw_a = 0, 0, 1
                win_b, loss_b, draw_b = 0, 0, 1

            # Update ratings
            now = time.time()
            for pid, new_elo, wins, losses, draws in [(p_a, new_a, win_a, loss_a, draw_a),
                                                       (p_b, new_b, win_b, loss_b, draw_b)]:
                cursor.execute("""
                    INSERT INTO elo_ratings
                    (participant_id, board_type, num_players, rating, games_played, wins, losses, draws, last_update)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(participant_id, board_type, num_players) DO UPDATE SET
                        rating = ?,
                        games_played = games_played + 1,
                        wins = wins + ?,
                        losses = losses + ?,
                        draws = draws + ?,
                        last_update = ?
                """, (pid, board_type, num_players, new_elo, wins, losses, draws, now,
                      new_elo, wins, losses, draws, now))

            inserted += 1
            if game_id:
                existing_game_ids.add(game_id)

        cursor.execute("COMMIT")
    except Exception as e:
        cursor.execute("ROLLBACK")
        print(f"Error merging matches, rolled back: {e}")
        raise
    finally:
        conn.close()

    return inserted


def push_to_coordinator(coordinator: str, db_path: Path, port: int = 8766):
    """Push new matches to coordinator."""
    last_sync = get_last_sync_timestamp()
    new_matches = get_new_matches(db_path, last_sync)

    if not new_matches:
        print("No new matches to push")
        return

    print(f"Pushing {len(new_matches)} new matches to {coordinator}")

    # Send matches via HTTP POST
    url = f"http://{coordinator}:{port}/sync"
    data = json.dumps(new_matches).encode()

    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            print(f"Coordinator accepted {result.get('inserted', 0)} matches")
            save_sync_timestamp(time.time())
    except Exception as e:
        print(f"Push failed: {e}")


def pull_from_coordinator(coordinator: str, db_path: Path, port: int = DEFAULT_PORT, use_aria2: bool = True):
    """Pull latest database from coordinator with multi-transport support.

    Transport priority:
    1. aria2 with multiple sources (if available and use_aria2=True)
    2. Direct HTTP from coordinator
    """
    # Try aria2 first if available
    if use_aria2 and check_aria2_available():
        print("Discovering available Elo sources...")
        sources = discover_available_sources()

        # Ensure coordinator is in sources (prioritize it)
        coordinator_url = get_node_db_url(coordinator, port)
        if coordinator_url not in sources:
            sources.insert(0, coordinator_url)
        elif sources[0] != coordinator_url:
            sources.remove(coordinator_url)
            sources.insert(0, coordinator_url)

        if len(sources) >= 2:
            print(f"Using aria2 with {len(sources)} sources for parallel download")
            if pull_with_aria2(sources, db_path):
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM match_history")
                count = cursor.fetchone()[0]
                conn.close()
                print(f"Pulled database with {count} matches via aria2")
                save_sync_timestamp(time.time())
                return
            else:
                print("aria2 failed, falling back to direct HTTP")

    # Fallback: direct HTTP
    url = f"http://{coordinator}:{port}/db"

    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
                f.write(response.read())
                temp_path = f.name

        # Verify the downloaded database
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM match_history")
        count = cursor.fetchone()[0]
        conn.close()

        if not safe_replace_wal_database(Path(temp_path), db_path, backup=True):
            print("Warning: Database replacement failed integrity checks.")

        print(f"Pulled database with {count} matches from {coordinator}")
        save_sync_timestamp(time.time())

    except Exception as e:
        print(f"Pull failed: {e}")


class SyncHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for coordinator sync endpoint."""

    db_path = DEFAULT_DB_PATH
    # Metrics counters
    sync_requests_total = 0
    sync_matches_total = 0
    sync_errors_total = 0
    last_sync_timestamp = 0

    def do_POST(self):
        if self.path == '/sync':
            content_length = int(self.headers['Content-Length'])
            data = self.rfile.read(content_length)
            matches = json.loads(data.decode())

            try:
                inserted = merge_matches_into_db(self.db_path, matches)
                SyncHandler.sync_requests_total += 1
                SyncHandler.sync_matches_total += inserted
                SyncHandler.last_sync_timestamp = time.time()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'inserted': inserted}).encode())
            except Exception as e:
                SyncHandler.sync_errors_total += 1
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/db':
            # Checkpoint WAL before serving to ensure consistent snapshot
            # This flushes pending writes from WAL to main database file
            try:
                checkpoint_conn = sqlite3.connect(self.db_path)
                checkpoint_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                checkpoint_conn.close()
            except Exception as e:
                print(f"Warning: WAL checkpoint failed: {e}")

            # Serve the database file
            with open(self.db_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/status':
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM match_history")
            match_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM elo_ratings")
            model_count = cursor.fetchone()[0]
            conn.close()

            status = {
                'matches': match_count,
                'models': model_count,
                'db_hash': get_db_hash(self.db_path),
                'timestamp': time.time()
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        elif self.path == '/metrics':
            # Prometheus metrics endpoint
            self._serve_metrics()
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_metrics(self):
        """Serve Prometheus-format metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM match_history")
        match_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM elo_ratings")
        model_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM elo_ratings WHERE archived_at IS NULL OR archived_at = 0")
        active_models = cursor.fetchone()[0]
        conn.close()

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        metrics = f"""# HELP ringrift_elo_matches_total Total number of matches in Elo database
# TYPE ringrift_elo_matches_total gauge
ringrift_elo_matches_total {match_count}

# HELP ringrift_elo_models_total Total number of models in Elo database
# TYPE ringrift_elo_models_total gauge
ringrift_elo_models_total {model_count}

# HELP ringrift_elo_active_models Number of active (non-archived) models
# TYPE ringrift_elo_active_models gauge
ringrift_elo_active_models {active_models}

# HELP ringrift_elo_db_size_bytes Size of Elo database in bytes
# TYPE ringrift_elo_db_size_bytes gauge
ringrift_elo_db_size_bytes {db_size}

# HELP ringrift_elo_sync_requests_total Total sync requests received
# TYPE ringrift_elo_sync_requests_total counter
ringrift_elo_sync_requests_total {SyncHandler.sync_requests_total}

# HELP ringrift_elo_sync_matches_total Total matches synced
# TYPE ringrift_elo_sync_matches_total counter
ringrift_elo_sync_matches_total {SyncHandler.sync_matches_total}

# HELP ringrift_elo_sync_errors_total Total sync errors
# TYPE ringrift_elo_sync_errors_total counter
ringrift_elo_sync_errors_total {SyncHandler.sync_errors_total}

# HELP ringrift_elo_last_sync_timestamp Unix timestamp of last sync
# TYPE ringrift_elo_last_sync_timestamp gauge
ringrift_elo_last_sync_timestamp {SyncHandler.last_sync_timestamp}
"""

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; version=0.0.4')
        self.end_headers()
        self.wfile.write(metrics.encode())

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def run_coordinator(port: int, db_path: Path):
    """Run the sync coordinator server."""
    SyncHandler.db_path = db_path

    with socketserver.TCPServer(("", port), SyncHandler) as httpd:
        print(f"Elo Sync Coordinator running on port {port}")
        print(f"Database: {db_path}")
        print("Endpoints:")
        print("  POST /sync - Submit new matches")
        print("  GET /db - Download full database")
        print("  GET /status - Get sync status")
        print("  GET /metrics - Prometheus metrics")
        print("  GET /health - Health check")
        httpd.serve_forever()


def run_worker(coordinator: str, interval: int, db_path: Path, port: int = DEFAULT_PORT):
    """Run as a sync worker, periodically pushing to coordinator."""
    print("Elo Sync Worker started")
    print(f"Coordinator: {coordinator}:{port}")
    print(f"Sync interval: {interval}s")
    print(f"Database: {db_path}")

    while True:
        try:
            push_to_coordinator(coordinator, db_path, port)
        except Exception as e:
            print(f"Sync error: {e}")

        time.sleep(interval)


# ============================================
# Cluster-Wide Operations
# ============================================

def get_node_status(target: dict[str, Any], timeout: int = 5) -> dict[str, Any]:
    """Get sync status from a single node."""
    ip = target['ip']
    port = target.get('port', DEFAULT_PORT)
    name = target['name']

    result = {
        'name': name,
        'ip': ip,
        'reachable': False,
        'matches': 0,
        'models': 0,
        'db_hash': None,
    }

    try:
        url = f"http://{ip}:{port}/status"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            result['reachable'] = True
            result['matches'] = data.get('matches', 0)
            result['models'] = data.get('models', 0)
            result['db_hash'] = data.get('db_hash')
    except (ConnectionError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, OSError):
        pass

    return result


def cluster_sync_status(db_path: Path) -> dict[str, Any]:
    """Get Elo sync status across all cluster nodes."""
    targets = get_all_sync_targets()

    # Get local status
    local_status = {
        'name': 'local',
        'reachable': True,
        'matches': 0,
        'models': 0,
        'db_hash': None,
    }

    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM match_history")
            local_status['matches'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM elo_ratings")
            local_status['models'] = cursor.fetchone()[0]
            conn.close()
            local_status['db_hash'] = get_db_hash(db_path)
        except Exception as e:
            print(f"Error reading local db: {e}")

    # Get remote statuses in parallel
    results = [local_status]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_node_status, t): t for t in targets}
        for future in as_completed(futures, timeout=15):
            try:
                result = future.result()
                results.append(result)
            except (ConnectionError, TimeoutError, RuntimeError):
                pass

    return {
        'local': local_status,
        'nodes': results,
        'timestamp': time.time(),
    }


def print_cluster_status(status: dict[str, Any]):
    """Print cluster sync status in a formatted table."""
    print("\n=== Elo Sync Cluster Status ===\n")

    local = status['local']
    print(f"Local: {local['matches']} matches, {local['models']} models")
    print(f"Hash: {local['db_hash']}\n")

    # Group by hash to identify divergence
    hash_groups: dict[str, list[str]] = {}
    reachable_count = 0

    for node in status['nodes']:
        if node['name'] == 'local':
            continue

        if node['reachable']:
            reachable_count += 1
            h = node['db_hash'] or 'unknown'
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(f"{node['name']} ({node['matches']})")

    print(f"Reachable nodes: {reachable_count}/{len(status['nodes']) - 1}\n")

    if len(hash_groups) > 1:
        print("WARNING: Database divergence detected!\n")
        for h, nodes in hash_groups.items():
            print(f"  Hash {h[:8]}...: {', '.join(nodes)}")
    elif hash_groups:
        h = next(iter(hash_groups.keys()))
        print(f"All nodes in sync (hash: {h[:8]}...)")
        print(f"  Nodes: {', '.join(next(iter(hash_groups.values())))}")
    else:
        print("No reachable nodes found")


def cluster_sync_push(db_path: Path):
    """Push database to all reachable nodes in the cluster."""
    targets = get_all_sync_targets()
    last_sync = get_last_sync_timestamp()
    new_matches = get_new_matches(db_path, last_sync)

    if not new_matches:
        print("No new matches to sync")
        return

    print(f"Pushing {len(new_matches)} matches to cluster...")

    def push_to_node(target):
        ip = target['ip']
        port = target.get('port', DEFAULT_PORT)
        name = target['name']

        try:
            url = f"http://{ip}:{port}/sync"
            data = json.dumps(new_matches).encode()
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/json')

            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
                result = json.loads(response.read().decode())
                return name, True, result.get('inserted', 0)
        except Exception as e:
            return name, False, str(e)

    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(push_to_node, t) for t in targets]
        for future in as_completed(futures, timeout=60):
            try:
                name, ok, result = future.result()
                if ok:
                    print(f"  {name}: synced {result} matches")
                    success_count += 1
                else:
                    print(f"  {name}: failed - {result}")
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nSynced to {success_count}/{len(targets)} nodes")
    if success_count > 0:
        save_sync_timestamp(time.time())


def main():
    parser = argparse.ArgumentParser(
        description="Elo Database Synchronization - Multi-transport cluster sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  coordinator   Run as sync coordinator server (receives pushes)
  worker        Run as sync daemon, periodically pushing to coordinator
  push          One-time push local matches to coordinator
  pull          One-time pull database from coordinator (uses aria2 if available)
  cluster-sync  Push matches to all reachable nodes in cluster
  status        Show sync status across all cluster nodes

Examples:
  # Run coordinator on mac-studio (authority):
  python elo_db_sync.py --mode coordinator

  # Run worker daemon (auto-discovers coordinator):
  python elo_db_sync.py --mode worker --interval 300

  # Check cluster sync status:
  python elo_db_sync.py --mode status

  # Sync to entire cluster:
  python elo_db_sync.py --mode cluster-sync
        """
    )
    parser.add_argument('--mode', choices=['coordinator', 'worker', 'push', 'pull', 'cluster-sync', 'status'],
                       required=True, help='Operation mode')
    parser.add_argument('--coordinator', type=str, help='Coordinator hostname/IP (auto-discovered if not specified)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Sync port')
    parser.add_argument('--interval', type=int, default=300, help='Sync interval in seconds (worker mode)')
    parser.add_argument('--db', type=Path, default=DEFAULT_DB_PATH, help='Database path')
    parser.add_argument('--no-aria2', action='store_true', help='Disable aria2 for pull operations')

    args = parser.parse_args()

    # Auto-discover coordinator if not specified
    coordinator = args.coordinator
    if not coordinator and args.mode in ('worker', 'push', 'pull'):
        coord_ip, coord_port = get_coordinator_address()
        if coord_ip:
            coordinator = coord_ip
            if args.port == DEFAULT_PORT:
                args.port = coord_port
            print(f"Auto-discovered coordinator: {coordinator}:{args.port}")
        else:
            print("[Script] Error: Could not determine coordinator address")
            print("[Script] Either:")
            print("[Script]   1. Set ELO_COORDINATOR_IP environment variable")
            print("[Script]   2. Configure elo_sync.coordinator in config/distributed_hosts.yaml")
            print("[Script]   3. Specify --coordinator argument")
            sys.exit(1)

    if args.mode == 'coordinator':
        run_coordinator(args.port, args.db)

    elif args.mode == 'worker':
        if not coordinator:
            print("Error: --coordinator required for worker mode")
            sys.exit(1)
        run_worker(coordinator, args.interval, args.db, args.port)

    elif args.mode == 'push':
        if not coordinator:
            print("Error: --coordinator required for push mode")
            sys.exit(1)
        push_to_coordinator(coordinator, args.db, args.port)

    elif args.mode == 'pull':
        if not coordinator:
            print("Error: --coordinator required for pull mode")
            sys.exit(1)
        pull_from_coordinator(coordinator, args.db, args.port, use_aria2=not args.no_aria2)

    elif args.mode == 'cluster-sync':
        cluster_sync_push(args.db)

    elif args.mode == 'status':
        status = cluster_sync_status(args.db)
        print_cluster_status(status)


if __name__ == '__main__':
    main()

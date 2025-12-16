#!/usr/bin/env python3
"""
Elo Database Synchronization Service

Keeps unified_elo.db synchronized across all cluster nodes.
Uses Tailscale for secure node-to-node communication.

Architecture:
- Each node runs this script as a daemon
- Nodes periodically push new match_history records to a coordinator
- Coordinator merges and recalculates Elo ratings
- Updated database is distributed back to all nodes

Usage:
    # Run as sync daemon (worker mode):
    python elo_db_sync.py --mode worker --coordinator lambda-h100 --interval 300

    # Run as coordinator (receives and merges):
    python elo_db_sync.py --mode coordinator --port 8766

    # One-time sync (pull latest from coordinator):
    python elo_db_sync.py --mode pull --coordinator lambda-h100

    # One-time sync (push local changes to coordinator):
    python elo_db_sync.py --mode push --coordinator lambda-h100
"""

import argparse
import sqlite3
import subprocess
import sys
import time
import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import http.server
import socketserver
import threading
import urllib.request

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "unified_elo.db"
SYNC_STATE_FILE = Path(__file__).parent.parent / "data" / "elo_sync_state.json"


def get_db_hash(db_path: Path) -> str:
    """Get hash of match_history for change detection."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MAX(timestamp) FROM match_history")
    count, max_ts = cursor.fetchone()
    conn.close()
    return hashlib.md5(f"{count}:{max_ts}".encode()).hexdigest()


def get_new_matches(db_path: Path, since_timestamp: float) -> List[Dict]:
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


def calculate_elo(winner_elo: float, loser_elo: float, k: float = 32, draw: bool = False) -> Tuple[float, float]:
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


def merge_matches_into_db(db_path: Path, new_matches: List[Dict]):
    """Merge new matches into database and recalculate Elo."""
    if not new_matches:
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get existing game_ids to avoid duplicates
    cursor.execute("SELECT game_id FROM match_history WHERE game_id IS NOT NULL")
    existing_game_ids = {row[0] for row in cursor.fetchall()}

    inserted = 0
    for match in new_matches:
        game_id = match.get('game_id')
        if game_id and game_id in existing_game_ids:
            continue

        cursor.execute("""
            INSERT INTO match_history
            (participant_a, participant_b, board_type, num_players, winner,
             game_length, duration_sec, timestamp, tournament_id, game_id, metadata, worker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match['participant_a'], match['participant_b'], match['board_type'],
            match['num_players'], match.get('winner'), match.get('game_length'),
            match.get('duration_sec'), match['timestamp'], match.get('tournament_id'),
            match.get('game_id'), match.get('metadata'), match.get('worker')
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
        for pid, new_elo, w, l, d in [(p_a, new_a, win_a, loss_a, draw_a),
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
            """, (pid, board_type, num_players, new_elo, w, l, d, now,
                  new_elo, w, l, d, now))

        inserted += 1
        if game_id:
            existing_game_ids.add(game_id)

    conn.commit()
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


def pull_from_coordinator(coordinator: str, db_path: Path, port: int = 8766):
    """Pull latest database from coordinator."""
    url = f"http://{coordinator}:{port}/db"

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
                f.write(response.read())
                temp_path = f.name

        # Verify the downloaded database
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM match_history")
        count = cursor.fetchone()[0]
        conn.close()

        # Replace local database
        import shutil
        shutil.copy(temp_path, db_path)
        os.unlink(temp_path)

        print(f"Pulled database with {count} matches from {coordinator}")
        save_sync_timestamp(time.time())

    except Exception as e:
        print(f"Pull failed: {e}")


class SyncHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for coordinator sync endpoint."""

    db_path = DEFAULT_DB_PATH

    def do_POST(self):
        if self.path == '/sync':
            content_length = int(self.headers['Content-Length'])
            data = self.rfile.read(content_length)
            matches = json.loads(data.decode())

            inserted = merge_matches_into_db(self.db_path, matches)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'inserted': inserted}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/db':
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
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def run_coordinator(port: int, db_path: Path):
    """Run the sync coordinator server."""
    SyncHandler.db_path = db_path

    with socketserver.TCPServer(("", port), SyncHandler) as httpd:
        print(f"Elo Sync Coordinator running on port {port}")
        print(f"Database: {db_path}")
        print(f"Endpoints:")
        print(f"  POST /sync - Submit new matches")
        print(f"  GET /db - Download full database")
        print(f"  GET /status - Get sync status")
        httpd.serve_forever()


def run_worker(coordinator: str, interval: int, db_path: Path, port: int = 8766):
    """Run as a sync worker, periodically pushing to coordinator."""
    print(f"Elo Sync Worker started")
    print(f"Coordinator: {coordinator}:{port}")
    print(f"Sync interval: {interval}s")
    print(f"Database: {db_path}")

    while True:
        try:
            push_to_coordinator(coordinator, db_path, port)
        except Exception as e:
            print(f"Sync error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Elo Database Synchronization")
    parser.add_argument('--mode', choices=['coordinator', 'worker', 'push', 'pull'],
                       required=True, help='Operation mode')
    parser.add_argument('--coordinator', type=str, help='Coordinator hostname')
    parser.add_argument('--port', type=int, default=8766, help='Sync port')
    parser.add_argument('--interval', type=int, default=300, help='Sync interval (seconds)')
    parser.add_argument('--db', type=Path, default=DEFAULT_DB_PATH, help='Database path')

    args = parser.parse_args()

    if args.mode == 'coordinator':
        run_coordinator(args.port, args.db)
    elif args.mode == 'worker':
        if not args.coordinator:
            print("Error: --coordinator required for worker mode")
            sys.exit(1)
        run_worker(args.coordinator, args.interval, args.db, args.port)
    elif args.mode == 'push':
        if not args.coordinator:
            print("Error: --coordinator required for push mode")
            sys.exit(1)
        push_to_coordinator(args.coordinator, args.db, args.port)
    elif args.mode == 'pull':
        if not args.coordinator:
            print("Error: --coordinator required for pull mode")
            sys.exit(1)
        pull_from_coordinator(args.coordinator, args.db, args.port)


if __name__ == '__main__':
    main()

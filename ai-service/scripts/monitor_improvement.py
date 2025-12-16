#!/usr/bin/env python3
"""Monitor AI improvement progress and alert on significant events.

Checks for:
- New models surpassing baseline ELO
- Training completion
- Models ready for culling
- Gauntlet evaluation progress

Usage:
    python scripts/monitor_improvement.py              # Single check
    python scripts/monitor_improvement.py --watch      # Continuous monitoring
    python scripts/monitor_improvement.py --slack URL  # Send alerts to Slack
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
BASELINE_ELO = 1650  # Models above this are considered strong
CULL_THRESHOLD = 1450  # Models below this with 30+ games get culled
MIN_GAMES_SIGNIFICANT = 30  # Minimum games for statistical significance
CHECK_INTERVAL = 300  # 5 minutes between checks in watch mode

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
STATE_FILE = AI_SERVICE_ROOT / "data" / "monitor_state.json"


@dataclass
class MonitorState:
    """Persisted state for change detection."""
    last_check: float = 0
    last_match_count: int = 0
    known_strong_models: List[str] = None
    last_training_status: str = "unknown"

    def __post_init__(self):
        if self.known_strong_models is None:
            self.known_strong_models = []


def load_state() -> MonitorState:
    """Load persisted state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            return MonitorState(**data)
        except Exception:
            pass
    return MonitorState()


def save_state(state: MonitorState):
    """Save state for next run."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump({
            'last_check': state.last_check,
            'last_match_count': state.last_match_count,
            'known_strong_models': state.known_strong_models,
            'last_training_status': state.last_training_status,
        }, f)


def get_elo_stats() -> Dict:
    """Get current ELO database statistics."""
    if not ELO_DB_PATH.exists():
        return {}

    conn = sqlite3.connect(ELO_DB_PATH)
    cur = conn.cursor()

    stats = {}

    # Total matches
    cur.execute('SELECT COUNT(*) FROM match_history')
    stats['total_matches'] = cur.fetchone()[0]

    # Active models
    cur.execute('''
        SELECT COUNT(*) FROM elo_ratings
        WHERE participant_id LIKE 'ringrift_%' AND archived_at IS NULL
    ''')
    stats['active_models'] = cur.fetchone()[0]

    # Archived models
    cur.execute('SELECT COUNT(*) FROM elo_ratings WHERE archived_at IS NOT NULL')
    stats['archived_models'] = cur.fetchone()[0]

    # Strong models (above baseline with 30+ games)
    cur.execute('''
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE participant_id LIKE 'ringrift_%'
        AND archived_at IS NULL
        AND games_played >= ?
        AND rating >= ?
        ORDER BY rating DESC
    ''', (MIN_GAMES_SIGNIFICANT, BASELINE_ELO))
    stats['strong_models'] = cur.fetchall()

    # Top performer
    cur.execute('''
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE participant_id LIKE 'ringrift_%'
        AND archived_at IS NULL
        AND games_played >= ?
        ORDER BY rating DESC
        LIMIT 1
    ''', (MIN_GAMES_SIGNIFICANT,))
    row = cur.fetchone()
    if row:
        stats['top_model'] = {'name': row[0], 'elo': row[1], 'games': row[2]}

    # Cull candidates
    cur.execute('''
        SELECT COUNT(*) FROM elo_ratings
        WHERE participant_id LIKE 'ringrift_%'
        AND archived_at IS NULL
        AND games_played >= ?
        AND rating < ?
    ''', (MIN_GAMES_SIGNIFICANT, CULL_THRESHOLD))
    stats['cull_candidates'] = cur.fetchone()[0]

    conn.close()
    return stats


def check_training_status() -> Tuple[str, str]:
    """Check if training is running on RTX 4060Ti."""
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
             '-o', 'BatchMode=yes', '-p', '14400', 'root@ssh1.vast.ai',
             'pgrep -c -f "multi_config_training_loop" 2>/dev/null || echo 0'],
            capture_output=True, text=True, timeout=20
        )
        count = int(result.stdout.strip().split()[-1])
        if count > 0:
            return "running", f"{count} training process(es)"
        else:
            return "stopped", "No training processes"
    except Exception as e:
        return "unknown", str(e)


def send_alert(title: str, message: str, slack_url: Optional[str] = None):
    """Send alert via console and optionally Slack."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Console output
    print(f"\n{'='*60}")
    print(f"🚨 ALERT: {title}")
    print(f"   Time: {timestamp}")
    print(f"   {message}")
    print(f"{'='*60}\n")

    # Slack webhook if configured
    if slack_url:
        try:
            import urllib.request
            payload = json.dumps({
                "text": f"*{title}*\n{message}",
                "username": "RingRift Monitor",
                "icon_emoji": ":robot_face:"
            }).encode()
            req = urllib.request.Request(slack_url, data=payload,
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"Slack alert failed: {e}")


def run_check(state: MonitorState, slack_url: Optional[str] = None) -> MonitorState:
    """Run a single monitoring check."""
    stats = get_elo_stats()

    if not stats:
        print("Could not load ELO database")
        return state

    alerts = []

    # Check for new strong models
    current_strong = [m[0] for m in stats.get('strong_models', [])]
    new_strong = [m for m in current_strong if m not in state.known_strong_models]

    if new_strong:
        for model in new_strong:
            model_data = next((m for m in stats['strong_models'] if m[0] == model), None)
            if model_data:
                alerts.append((
                    "New Strong Model!",
                    f"{model_data[0]} reached ELO {model_data[1]:.0f} ({model_data[2]} games)"
                ))

    # Check training status change
    training_status, training_detail = check_training_status()
    if state.last_training_status == "running" and training_status == "stopped":
        alerts.append((
            "Training Completed",
            f"Training has stopped. Check logs for new model."
        ))
    elif state.last_training_status != "running" and training_status == "running":
        alerts.append((
            "Training Started",
            training_detail
        ))

    # Check for significant match increase
    match_increase = stats['total_matches'] - state.last_match_count
    if match_increase > 10000:
        alerts.append((
            "High Gauntlet Throughput",
            f"+{match_increase:,} matches since last check"
        ))

    # Send alerts
    for title, message in alerts:
        send_alert(title, message, slack_url)

    # Print status
    print(f"\n📊 Status @ {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Matches: {stats['total_matches']:,} (+{match_increase:,})")
    print(f"   Active models: {stats['active_models']}")
    print(f"   Strong models (ELO≥{BASELINE_ELO}): {len(current_strong)}")
    print(f"   Cull candidates: {stats['cull_candidates']}")
    print(f"   Training: {training_status}")

    if stats.get('top_model'):
        top = stats['top_model']
        print(f"   🏆 Top: {top['name'][:40]} (ELO {top['elo']:.0f})")

    # Update state
    state.last_check = time.time()
    state.last_match_count = stats['total_matches']
    state.known_strong_models = current_strong
    state.last_training_status = training_status

    return state


def main():
    parser = argparse.ArgumentParser(description="Monitor AI improvement progress")
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring')
    parser.add_argument('--slack', type=str, help='Slack webhook URL for alerts')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL,
                        help=f'Check interval in seconds (default: {CHECK_INTERVAL})')
    args = parser.parse_args()

    state = load_state()

    if args.watch:
        print(f"Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                state = run_check(state, args.slack)
                save_state(state)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    else:
        state = run_check(state, args.slack)
        save_state(state)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Automated ELO tournament runner with Slack alerts.

Runs periodic tournaments to track model quality and alert on regressions.
Designed to run as a cron job or systemd timer.

Usage:
    # Run once
    python scripts/auto_elo_tournament.py

    # Run as daemon (every 4 hours)
    python scripts/auto_elo_tournament.py --daemon --interval 14400

    # Dry run (check what would happen)
    python scripts/auto_elo_tournament.py --dry-run
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Configuration
DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"
MODELS_DIR = AI_SERVICE_ROOT / "models"
SLACK_WEBHOOK_URL = os.environ.get("RINGRIFT_SLACK_WEBHOOK_URL", "")
ALERT_WEBHOOK_URL = os.environ.get("RINGRIFT_WEBHOOK_URL", "")

# Tournament settings
GAMES_PER_MATCHUP = 5
TOP_N_MODELS = 5  # Test latest N models
MIN_HOURS_BETWEEN_TOURNAMENTS = 2

# Regression thresholds
ELO_REGRESSION_THRESHOLD = 100  # Alert if best model drops by this much
MIN_WINRATE_VS_BASELINE = 0.6  # Alert if new models can't beat baselines


def send_slack_message(message: str, alert_type: str = "info"):
    """Send message to Slack webhook."""
    webhook = SLACK_WEBHOOK_URL or ALERT_WEBHOOK_URL
    if not webhook:
        print(f"[{alert_type.upper()}] {message}")
        return False

    # Format for Slack
    emoji = {
        "info": ":information_source:",
        "success": ":white_check_mark:",
        "warning": ":warning:",
        "error": ":x:",
        "trophy": ":trophy:",
    }.get(alert_type, ":robot_face:")

    payload = {
        "text": f"{emoji} *RingRift ELO Tournament*\n{message}"
    }

    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
             "-d", json.dumps(payload), webhook],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to send Slack message: {e}")
        return False


def get_recent_models(n: int = 10) -> list[Path]:
    """Get the N most recently modified model files."""
    if not MODELS_DIR.exists():
        return []

    pth_files = list(MODELS_DIR.glob("*.pth"))
    # Sort by modification time, newest first
    pth_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pth_files[:n]


def get_elo_ratings(db_path: Path) -> dict[str, float]:
    """Get current ELO ratings from database."""
    if not db_path.exists():
        return {}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT participant_id, rating FROM elo_ratings ORDER BY rating DESC"
        )
        ratings = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return ratings
    except Exception as e:
        print(f"Error reading ELO database: {e}")
        return {}


def get_last_tournament_time(db_path: Path) -> datetime | None:
    """Get timestamp of last tournament from database."""
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT MAX(played_at) FROM match_history"
        )
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return datetime.fromisoformat(result[0].replace("Z", "+00:00"))
        return None
    except Exception:
        return None


def run_tournament(
    board_type: str = "square8",
    num_players: int = 2,
    top_n: int = 5,
    games: int = 5,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Run ELO tournament and return (success, message)."""

    # Get pre-tournament ratings for comparison
    pre_ratings = get_elo_ratings(DEFAULT_DB)
    best_model_before = max(pre_ratings.items(), key=lambda x: x[1]) if pre_ratings else None

    # Build command
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_model_elo_tournament.py"),
        "--board", board_type,
        "--players", str(num_players),
        "--top-n", str(top_n),
        "--games", str(games),
        "--run",
        "--include-baselines",
        "--db", str(DEFAULT_DB),
    ]

    if dry_run:
        return True, f"Would run: {' '.join(cmd)}"

    print(f"Running tournament: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            cwd=str(AI_SERVICE_ROOT),
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return False, f"Tournament failed after {duration:.1f}s: {result.stderr[:500]}"

        # Get post-tournament ratings
        post_ratings = get_elo_ratings(DEFAULT_DB)

        # Find best model now
        if post_ratings:
            best_model_after = max(post_ratings.items(), key=lambda x: x[1])

            # Check for regressions
            if best_model_before and best_model_after:
                old_best_rating = pre_ratings.get(best_model_before[0], 1500)
                new_rating_of_old_best = post_ratings.get(best_model_before[0], old_best_rating)

                if old_best_rating - new_rating_of_old_best > ELO_REGRESSION_THRESHOLD:
                    return True, (
                        f":warning: *Regression detected!*\n"
                        f"Previous best `{best_model_before[0]}` dropped from "
                        f"{old_best_rating:.0f} to {new_rating_of_old_best:.0f} ELO\n"
                        f"New best: `{best_model_after[0]}` ({best_model_after[1]:.0f} ELO)"
                    )

            # Success message with top models
            top_5 = sorted(post_ratings.items(), key=lambda x: -x[1])[:5]
            leaderboard = "\n".join(
                f"  {i+1}. `{m}`: {r:.0f}" for i, (m, r) in enumerate(top_5)
            )

            return True, (
                f"Tournament complete ({duration:.0f}s)\n"
                f"*Top 5:*\n{leaderboard}"
            )

        return True, f"Tournament complete ({duration:.0f}s)"

    except subprocess.TimeoutExpired:
        return False, "Tournament timed out after 1 hour"
    except Exception as e:
        return False, f"Tournament error: {e}"


def should_run_tournament(db_path: Path, min_hours: float = 2) -> tuple[bool, str]:
    """Check if enough time has passed since last tournament."""
    last_time = get_last_tournament_time(db_path)

    if not last_time:
        return True, "No previous tournament found"

    hours_since = (datetime.now(last_time.tzinfo) - last_time).total_seconds() / 3600

    if hours_since < min_hours:
        return False, f"Only {hours_since:.1f}h since last tournament (min: {min_hours}h)"

    return True, f"Last tournament was {hours_since:.1f}h ago"


def run_daemon(interval_seconds: int = 14400):
    """Run tournament daemon (default: every 4 hours)."""
    print(f"Starting ELO tournament daemon (interval: {interval_seconds}s)")
    send_slack_message(
        f"ELO tournament daemon started (interval: {interval_seconds/3600:.1f}h)",
        "info"
    )

    while True:
        try:
            # Check if we should run
            should_run, reason = should_run_tournament(
                DEFAULT_DB,
                min_hours=MIN_HOURS_BETWEEN_TOURNAMENTS
            )

            if should_run:
                print(f"[{datetime.now().strftime('%H:%M')}] Running tournament: {reason}")

                success, message = run_tournament(
                    board_type="square8",
                    num_players=2,
                    top_n=TOP_N_MODELS,
                    games=GAMES_PER_MATCHUP,
                )

                alert_type = "success" if success else "error"
                send_slack_message(message, alert_type)
            else:
                print(f"[{datetime.now().strftime('%H:%M')}] Skipping: {reason}")

        except Exception as e:
            print(f"Daemon error: {e}")
            send_slack_message(f"Tournament daemon error: {e}", "error")

        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Automated ELO tournament runner")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=14400,
                        help="Daemon interval in seconds (default: 4 hours)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--top-n", type=int, default=TOP_N_MODELS,
                        help="Number of recent models to test")
    parser.add_argument("--games", type=int, default=GAMES_PER_MATCHUP,
                        help="Games per matchup")

    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.interval)
    else:
        # Single run
        _should_run, reason = should_run_tournament(DEFAULT_DB, min_hours=0)
        print(f"Tournament status: {reason}")

        success, message = run_tournament(
            board_type=args.board,
            num_players=args.players,
            top_n=args.top_n,
            games=args.games,
            dry_run=args.dry_run,
        )

        print(f"\n{'='*50}")
        print(message)

        if not args.dry_run and success:
            send_slack_message(message, "success" if success else "error")


if __name__ == "__main__":
    main()

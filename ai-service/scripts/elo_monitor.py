#!/usr/bin/env python3
"""
Elo Monitoring Script

Tracks Elo ratings across all configurations and alerts on:
- Stagnation (no improvement for N hours)
- Regression (Elo dropped significantly)
- Missing tournaments (no games for N hours)

Usage:
    python scripts/elo_monitor.py [--alert-hours 6] [--regression-threshold 30]

Cron example (every 30 minutes):
    */30 * * * * cd ~/ringrift/ai-service && PYTHONPATH=. python scripts/elo_monitor.py >> logs/elo_monitor.log 2>&1
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_elo_history(history_file: Path) -> Dict:
    """Load Elo history from file."""
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return {"configs": {}, "last_check": None}


def save_elo_history(history: Dict, history_file: Path):
    """Save Elo history to file."""
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def get_current_elo_ratings() -> Dict[str, Dict]:
    """Get current Elo ratings from the tracker or file."""
    ratings = {}

    # Try loading from common locations
    elo_files = [
        Path("data/elo_ratings.json"),
        Path("data/p2p_hybrid/elo_ratings.json"),
        Path("models/elo_ratings.json"),
    ]

    for elo_file in elo_files:
        if elo_file.exists():
            try:
                with open(elo_file) as f:
                    data = json.load(f)
                    for config, info in data.items():
                        if config not in ratings:
                            ratings[config] = {
                                "elo": info.get("elo", info.get("rating", 1500)),
                                "games": info.get("games", info.get("total_games", 0)),
                                "last_update": info.get("last_update", info.get("updated_at")),
                                "source": str(elo_file)
                            }
            except Exception as e:
                print(f"Warning: Could not load {elo_file}: {e}")

    # Also check for model-specific Elo from training logs
    training_logs = Path("logs").glob("training_*.log")
    for log_file in training_logs:
        try:
            with open(log_file) as f:
                content = f.read()
                # Look for Elo updates in log
                # Format: "New Elo: 1725.2" or "Elo improved: 1700 -> 1725"
                import re
                matches = re.findall(r'Elo[:\s]+(\d+\.?\d*)', content[-10000:])  # Last 10k chars
                if matches:
                    # Extract config from filename
                    config_match = re.search(r'training_(\w+)_(\d+p)', log_file.name)
                    if config_match:
                        config = f"{config_match.group(1)}_{config_match.group(2)}"
                        if config not in ratings:
                            ratings[config] = {
                                "elo": float(matches[-1]),
                                "games": 0,
                                "last_update": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                                "source": str(log_file)
                            }
        except Exception:
            pass

    return ratings


def check_training_activity() -> Dict[str, Dict]:
    """Check recent training activity from logs."""
    activity = {}

    log_dir = Path("logs")
    if not log_dir.exists():
        return activity

    now = datetime.now()

    for log_file in log_dir.glob("training_*.log"):
        try:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            age_hours = (now - mtime).total_seconds() / 3600

            # Extract config from filename
            import re
            config_match = re.search(r'training_(\w+_\d+p)', log_file.name)
            if config_match:
                config = config_match.group(1)

                # Check if actively writing (modified in last 5 min)
                is_active = age_hours < 0.1

                activity[config] = {
                    "log_file": str(log_file),
                    "last_modified": mtime.isoformat(),
                    "age_hours": round(age_hours, 2),
                    "is_active": is_active
                }
        except Exception:
            pass

    return activity


def analyze_elo_status(
    current: Dict[str, Dict],
    history: Dict,
    alert_hours: float,
    regression_threshold: float
) -> List[Dict]:
    """Analyze Elo status and generate alerts."""
    alerts = []
    now = datetime.now(timezone.utc)

    for config, data in current.items():
        elo = data.get("elo", 1500)
        last_update = data.get("last_update")

        # Get historical data
        hist = history.get("configs", {}).get(config, {})
        prev_elo = hist.get("elo", elo)
        prev_check = hist.get("last_check")
        peak_elo = hist.get("peak_elo", elo)
        stagnant_since = hist.get("stagnant_since")

        # Check for regression
        if prev_elo - elo > regression_threshold:
            alerts.append({
                "type": "regression",
                "severity": "high",
                "config": config,
                "message": f"Elo dropped {prev_elo:.1f} -> {elo:.1f} ({prev_elo - elo:.1f} points)",
                "elo": elo,
                "prev_elo": prev_elo
            })

        # Check for stagnation
        if last_update:
            try:
                if isinstance(last_update, str):
                    last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                else:
                    last_dt = last_update

                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)

                hours_since = (now - last_dt).total_seconds() / 3600

                if hours_since > alert_hours:
                    alerts.append({
                        "type": "stagnation",
                        "severity": "medium",
                        "config": config,
                        "message": f"No updates for {hours_since:.1f} hours (threshold: {alert_hours}h)",
                        "elo": elo,
                        "hours_since": hours_since
                    })
            except Exception:
                pass

        # Check if below peak
        if peak_elo - elo > 50:
            alerts.append({
                "type": "below_peak",
                "severity": "low",
                "config": config,
                "message": f"Current {elo:.1f} is {peak_elo - elo:.1f} below peak {peak_elo:.1f}",
                "elo": elo,
                "peak_elo": peak_elo
            })

        # Update history
        if "configs" not in history:
            history["configs"] = {}

        history["configs"][config] = {
            "elo": elo,
            "peak_elo": max(peak_elo, elo),
            "last_check": now.isoformat(),
            "stagnant_since": stagnant_since if abs(elo - prev_elo) < 5 else None
        }

    history["last_check"] = now.isoformat()

    return alerts


def print_status_report(
    ratings: Dict[str, Dict],
    activity: Dict[str, Dict],
    alerts: List[Dict]
):
    """Print a formatted status report."""
    print(f"\n{Colors.BOLD}{'='*60}")
    print(f"  ELO MONITORING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{Colors.END}\n")

    # Current Ratings
    print(f"{Colors.BLUE}{Colors.BOLD}Current Elo Ratings:{Colors.END}")
    print("-" * 50)

    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1].get("elo", 0), reverse=True)
    for config, data in sorted_ratings:
        elo = data.get("elo", 1500)
        games = data.get("games", 0)

        # Color based on Elo
        if elo >= 1800:
            color = Colors.GREEN
        elif elo >= 1600:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        print(f"  {config:20s} {color}{elo:7.1f}{Colors.END} Elo  ({games:5d} games)")

    # Training Activity
    print(f"\n{Colors.BLUE}{Colors.BOLD}Training Activity:{Colors.END}")
    print("-" * 50)

    if activity:
        for config, info in sorted(activity.items()):
            status = f"{Colors.GREEN}ACTIVE{Colors.END}" if info["is_active"] else f"{Colors.YELLOW}idle {info['age_hours']:.1f}h{Colors.END}"
            print(f"  {config:20s} {status}")
    else:
        print(f"  {Colors.YELLOW}No training logs found{Colors.END}")

    # Alerts
    if alerts:
        print(f"\n{Colors.RED}{Colors.BOLD}ALERTS:{Colors.END}")
        print("-" * 50)

        severity_colors = {
            "high": Colors.RED,
            "medium": Colors.YELLOW,
            "low": Colors.BLUE
        }

        for alert in sorted(alerts, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3)):
            color = severity_colors.get(alert["severity"], Colors.END)
            print(f"  {color}[{alert['severity'].upper()}]{Colors.END} {alert['config']}: {alert['message']}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}No alerts - all systems healthy{Colors.END}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor Elo ratings and alert on issues")
    parser.add_argument("--alert-hours", type=float, default=6, help="Hours without update before alerting")
    parser.add_argument("--regression-threshold", type=float, default=30, help="Elo drop threshold for alert")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of formatted text")
    parser.add_argument("--history-file", type=str, default="data/elo_monitor_history.json", help="History file path")
    args = parser.parse_args()

    history_file = Path(args.history_file)

    # Load history
    history = load_elo_history(history_file)

    # Get current state
    ratings = get_current_elo_ratings()
    activity = check_training_activity()

    # Analyze and generate alerts
    alerts = analyze_elo_status(ratings, history, args.alert_hours, args.regression_threshold)

    # Save updated history
    save_elo_history(history, history_file)

    # Output
    if args.json:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ratings": ratings,
            "activity": activity,
            "alerts": alerts
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_status_report(ratings, activity, alerts)

    # Exit with error code if high severity alerts
    high_alerts = [a for a in alerts if a["severity"] == "high"]
    if high_alerts:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Training Loop Monitor - Tracks health and progress of the unified AI loop.

Unified monitoring and alerting for the training pipeline. This script:
- Monitors training runs and model production
- Checks data collection rates and database health
- Tracks model promotions and Elo progression
- Generates alerts for issues (critical/warning/info levels)
- Checks disk usage and system health

Usage:
    # One-shot status check
    python scripts/training_monitor.py

    # Detailed report with alerts
    python scripts/training_monitor.py --verbose

    # JSON output for automation
    python scripts/training_monitor.py --json

    # Cron mode (exit 1 if critical alerts)
    python scripts/training_monitor.py --cron

    # Save alerts to file
    python scripts/training_monitor.py --output alerts.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.paths import AI_SERVICE_ROOT
from app.monitoring.thresholds import get_threshold

# Use shared logging from scripts/lib
from scripts.lib.logging_config import setup_script_logging

LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_script_logging("training_monitor", log_dir=str(LOG_DIR))

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        get_disk_usage as unified_get_disk_usage,
        get_gpu_memory_usage as unified_get_gpu_usage,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_get_disk_usage = None
    unified_get_gpu_usage = None
    RESOURCE_LIMITS = None

# Alert thresholds - using centralized monitoring thresholds
THRESHOLDS = {
    "training_stale_hours": get_threshold("training", "stale_hours", default=24),
    "model_stale_hours": get_threshold("training", "model_stale_hours", default=48),
    "disk_warning_percent": get_threshold("disk", "warning", default=70),
    "disk_critical_percent": get_threshold("disk", "critical", default=85),
    "consecutive_failures_warning": 3,    # Failures before warning
    "low_gpu_utilization": get_threshold("gpu_utilization", "low", default=10),
}

# Priority configs that need attention (sparse data configs)
PRIORITY_CONFIGS = [
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    "square19_2p", "square19_3p", "square19_4p",
]


@dataclass
class Alert:
    """Represents a monitoring alert."""
    level: str  # "info", "warning", "critical"
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def load_unified_state() -> Optional[Dict[str, Any]]:
    """Load the unified loop state file."""
    state_path = AI_SERVICE_ROOT / "logs" / "unified_loop" / "unified_loop_state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return None


def check_recent_models(hours: int = 24) -> List[Tuple[str, datetime, int]]:
    """Find models created in the last N hours."""
    models_dir = AI_SERVICE_ROOT / "models"
    cutoff = time.time() - (hours * 3600)
    recent = []

    for pattern in ["*.pt", "*.pth"]:
        for model_path in models_dir.glob(pattern):
            if model_path.stat().st_mtime > cutoff:
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                size = model_path.stat().st_size
                recent.append((model_path.name, mtime, size))

    return sorted(recent, key=lambda x: x[1], reverse=True)


def check_db_health(db_path: Path) -> Tuple[bool, str, int, int, int]:
    """Check if a SQLite database is healthy.

    Returns: (healthy, message, total_games, trainable_games, games_with_moves)
    """
    if not db_path.exists():
        return False, "File not found", 0, 0, 0

    if db_path.stat().st_size == 0:
        return False, "Empty file", 0, 0, 0

    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            conn.close()
            return False, f"Integrity check failed: {result[0]}", 0, 0, 0

        # Try to count games
        total_count = 0
        trainable_count = 0
        with_moves_count = 0

        try:
            total_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        except sqlite3.Error:
            pass

        # Check columns
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}

            has_excluded = "excluded_from_training" in cols
            has_moves_table = "game_moves" in tables

            # Count trainable games (not excluded)
            if has_excluded:
                trainable_count = conn.execute(
                    "SELECT COUNT(*) FROM games WHERE COALESCE(excluded_from_training, 0) = 0"
                ).fetchone()[0]
            else:
                trainable_count = total_count

            # Count games with move data
            if has_moves_table:
                if has_excluded:
                    with_moves_count = conn.execute("""
                        SELECT COUNT(DISTINCT g.game_id) FROM games g
                        INNER JOIN game_moves gm ON g.game_id = gm.game_id
                        WHERE COALESCE(g.excluded_from_training, 0) = 0
                    """).fetchone()[0]
                else:
                    with_moves_count = conn.execute("""
                        SELECT COUNT(DISTINCT game_id) FROM game_moves
                    """).fetchone()[0]
        except sqlite3.Error:
            pass

        conn.close()
        return True, "OK", total_count, trainable_count, with_moves_count
    except Exception as e:
        return False, str(e), 0, 0, 0


def check_gpu_processes() -> List[Dict[str, Any]]:
    """Check running GPU processes."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        processes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(", ")
                if len(parts) >= 3:
                    processes.append({
                        "pid": parts[0],
                        "name": parts[1],
                        "memory_mb": int(parts[2]) if parts[2].isdigit() else 0
                    })
        return processes
    except Exception:
        return []


def check_gpu_utilization() -> Optional[int]:
    """Get current GPU utilization percentage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def check_disk_usage() -> Tuple[float, float]:
    """Check disk usage. Returns (percent_used, free_gb).

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement across the codebase.
    """
    data_dir = AI_SERVICE_ROOT / "data"

    # Use unified utilities when available
    if HAS_RESOURCE_GUARD and unified_get_disk_usage is not None:
        try:
            percent, free_gb, _ = unified_get_disk_usage(str(data_dir))
            return percent, free_gb
        except Exception:
            pass  # Fall through to original implementation

    # Fallback to original implementation
    try:
        total, used, free = shutil.disk_usage(str(data_dir))
        percent = (used / total) * 100
        return percent, free / (1024**3)
    except Exception:
        return 0.0, 0.0


def check_model_age() -> Tuple[Optional[float], Optional[str]]:
    """Check age of most recent model. Returns (hours_since, model_name)."""
    models_dir = AI_SERVICE_ROOT / "models"
    if not models_dir.exists():
        return None, None

    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        return None, None

    most_recent = max(model_files, key=lambda p: p.stat().st_mtime)
    hours_since = (time.time() - most_recent.stat().st_mtime) / 3600
    return hours_since, most_recent.name


def generate_report(verbose: bool = False) -> Dict[str, Any]:
    """Generate a comprehensive training status report with alerts."""
    alerts: List[Alert] = []
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "alerts": [],
        "metrics": {}
    }

    # Load state
    state = load_unified_state()
    if not state:
        alerts.append(Alert("critical", "training_loop", "State file not found - loop may not be running"))
    else:
        # Basic metrics
        report["metrics"]["total_training_runs"] = state.get("total_training_runs", 0)
        report["metrics"]["total_promotions"] = state.get("total_promotions", 0)
        report["metrics"]["total_data_syncs"] = state.get("total_data_syncs", 0)
        report["metrics"]["consecutive_failures"] = state.get("consecutive_failures", 0)
        report["metrics"]["training_in_progress"] = state.get("training_in_progress", False)

        # Check consecutive failures
        failures = state.get("consecutive_failures", 0)
        if failures >= THRESHOLDS["consecutive_failures_warning"]:
            alerts.append(Alert("warning", "training_loop", f"High consecutive failures: {failures}"))

        # Check training staleness per config
        now = time.time()
        for config_name, config_data in state.get("configs", {}).items():
            last_training = config_data.get("last_training_time", 0)
            hours_since = (now - last_training) / 3600 if last_training > 0 else float('inf')

            if config_name in PRIORITY_CONFIGS and hours_since > THRESHOLDS["training_stale_hours"]:
                alerts.append(Alert(
                    "warning", "training_stale",
                    f"Priority config {config_name} hasn't trained in {hours_since:.1f}h",
                    {"config": config_name, "hours_since": hours_since}
                ))

        # Config status
        configs_status = {}
        for config_key, config in state.get("configs", {}).items():
            configs_status[config_key] = {
                "games_since_training": config.get("games_since_training", 0),
                "current_elo": config.get("current_elo", 1500),
            }
        report["metrics"]["configs"] = configs_status

    # Check recent models
    recent_models = check_recent_models(24)
    report["metrics"]["models_last_24h"] = len(recent_models)

    # Check model age
    model_hours, model_name = check_model_age()
    if model_hours is not None:
        report["metrics"]["model_age_hours"] = round(model_hours, 1)
        if model_hours > THRESHOLDS["model_stale_hours"]:
            alerts.append(Alert(
                "warning", "model_stale",
                f"No new models in {model_hours:.1f}h (last: {model_name})",
                {"hours_since": model_hours, "last_model": model_name}
            ))
    elif model_name is None:
        alerts.append(Alert("warning", "models", "No model files found"))

    # Check GPU
    gpu_util = check_gpu_utilization()
    if gpu_util is not None:
        report["metrics"]["gpu_utilization"] = gpu_util
        if gpu_util < THRESHOLDS["low_gpu_utilization"]:
            alerts.append(Alert("warning", "gpu", f"Low GPU utilization: {gpu_util}%"))

    # Check disk usage
    disk_percent, disk_free_gb = check_disk_usage()
    report["metrics"]["disk_percent"] = round(disk_percent, 1)
    report["metrics"]["disk_free_gb"] = round(disk_free_gb, 1)
    if disk_percent >= THRESHOLDS["disk_critical_percent"]:
        alerts.append(Alert("critical", "disk_space", f"Disk usage critical: {disk_percent:.1f}%"))
    elif disk_percent >= THRESHOLDS["disk_warning_percent"]:
        alerts.append(Alert("warning", "disk_space", f"Disk usage high: {disk_percent:.1f}%"))

    # Check key databases used by multi_config_training_loop.py
    # Note: all_jsonl_training.db was deprecated in favor of selfplay.db with game_moves
    key_dbs = [
        "data/games/selfplay.db",  # Canonical DB with game_moves table
        "data/unified_elo.db",
    ]

    db_status = {}
    total_trainable = 0
    total_with_moves = 0
    for db_rel_path in key_dbs:
        db_path = AI_SERVICE_ROOT / db_rel_path
        healthy, msg, total_count, trainable_count, with_moves_count = check_db_health(db_path)
        db_status[db_rel_path] = {
            "healthy": healthy,
            "message": msg,
            "total_games": total_count,
            "trainable_games": trainable_count,
            "games_with_moves": with_moves_count,
        }
        total_trainable += trainable_count
        total_with_moves += with_moves_count
        if not healthy:
            alerts.append(Alert("warning", "database", f"DB issue - {db_rel_path}: {msg}"))
        elif total_count > 0 and trainable_count == 0:
            alerts.append(Alert("warning", "database",
                f"{db_rel_path}: {total_count} games but 0 trainable (all excluded)"))
        elif total_count > 0 and with_moves_count == 0:
            alerts.append(Alert("warning", "database",
                f"{db_rel_path}: {total_count} games but 0 with move data"))

    report["metrics"]["databases"] = db_status
    report["metrics"]["total_trainable_games"] = total_trainable
    report["metrics"]["total_games_with_moves"] = total_with_moves

    # Convert alerts to serializable format
    report["alerts"] = [
        {"level": a.level, "category": a.category, "message": a.message, "details": a.details}
        for a in alerts
    ]

    # Count by level
    critical_count = sum(1 for a in alerts if a.level == "critical")
    warning_count = sum(1 for a in alerts if a.level == "warning")
    report["critical_count"] = critical_count
    report["warning_count"] = warning_count

    # Determine overall status
    if critical_count > 0:
        report["status"] = "critical"
    elif warning_count > 0:
        report["status"] = "warning"
    else:
        report["status"] = "healthy"

    return report


def main():
    parser = argparse.ArgumentParser(description="Training Loop Monitor and Alerting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--cron", action="store_true", help="Exit with code 1 if critical alerts")
    parser.add_argument("--output", type=str, help="Save report to JSON file")
    args = parser.parse_args()

    report = generate_report(verbose=args.verbose)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {args.output}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        # Print summary
        status_emoji = {"healthy": "✓", "warning": "⚠", "critical": "✗", "unknown": "?"}
        print(f"\n{'='*60}")
        print(f"Training Loop Status: {status_emoji.get(report['status'], '?')} {report['status'].upper()}")
        print(f"{'='*60}")
        print(f"Timestamp: {report['timestamp']}")
        print()

        # Metrics
        m = report["metrics"]
        print("Metrics:")
        print(f"  Training runs: {m.get('total_training_runs', 'N/A')}")
        print(f"  Promotions: {m.get('total_promotions', 'N/A')}")
        print(f"  Data syncs: {m.get('total_data_syncs', 'N/A')}")
        print(f"  Models (24h): {m.get('models_last_24h', 'N/A')}")
        print(f"  GPU utilization: {m.get('gpu_utilization', 'N/A')}%")
        print(f"  Disk usage: {m.get('disk_percent', 'N/A')}% ({m.get('disk_free_gb', 'N/A')}GB free)")
        print(f"  Training in progress: {m.get('training_in_progress', 'N/A')}")
        print()

        # Alerts
        alerts = report.get("alerts", [])
        if alerts:
            print(f"Alerts ({report['critical_count']} critical, {report['warning_count']} warning):")
            for alert in alerts:
                icon = {"critical": "X", "warning": "!", "info": "i"}.get(alert["level"], "?")
                print(f"  [{icon}] {alert['category']}: {alert['message']}")
            print()
        else:
            print("No alerts - pipeline healthy\n")

        # Database status (verbose)
        if args.verbose and "databases" in m:
            print("Databases:")
            for db_path, status in m["databases"].items():
                icon = "✓" if status["healthy"] else "✗"
                trainable = status.get('trainable_games', 0)
                total = status.get('total_games', 0)
                print(f"  {icon} {db_path}: {status['message']} ({trainable}/{total} trainable)")
            print()

        # Config status (verbose)
        if args.verbose and "configs" in m:
            print("Configs:")
            for config_key, status in m["configs"].items():
                print(f"  {config_key}: {status['games_since_training']} games pending, Elo={status['current_elo']:.0f}")
            print()

    # Exit with error if critical alerts (for cron)
    if args.cron and report.get("critical_count", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

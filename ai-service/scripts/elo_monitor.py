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
    */30 * * * * cd ~/ringrift/ai-service && PYTHONPATH=. python scripts/elo_monitor.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.logging_config import (
    setup_script_logging,
    get_logger,
    get_metrics_logger,
)
from scripts.lib.alerts import (
    Alert,
    AlertSeverity,
    AlertType,
    create_alert,
)

logger = get_logger(__name__)


# Elo-specific dataclasses (keep separate from generic alerts)
@dataclass
class EloRating:
    """Current Elo rating for a configuration."""
    config_key: str
    elo: float
    games: int
    last_update: str | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingActivity:
    """Training activity status for a configuration."""
    config_key: str
    log_file: str
    last_modified: str
    age_hours: float
    is_active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EloAlert:
    """An Elo-specific alert (wrapper around shared Alert)."""
    alert_type: AlertType
    severity: AlertSeverity
    config_key: str
    message: str
    elo: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "config": self.config_key,
            "message": self.message,
            "elo": self.elo,
            **self.details,
        }

    def to_alert(self) -> Alert:
        """Convert to generic Alert."""
        details = {**self.details, "config_key": self.config_key, "elo": self.elo}
        return create_alert(
            self.severity,
            self.alert_type,
            self.message,
            details=details,
            source="elo_monitor",
        )


@dataclass
class EloHistory:
    """Historical Elo tracking for a configuration."""
    elo: float
    peak_elo: float
    last_check: str
    stagnant_since: str | None


class EloMonitor:
    """Monitors Elo ratings and generates alerts."""

    ELO_FILE_LOCATIONS = [
        Path("data/elo_ratings.json"),
        Path("data/p2p_hybrid/elo_ratings.json"),
        Path("models/elo_ratings.json"),
    ]

    def __init__(
        self,
        history_file: Path,
        alert_hours: float = 6.0,
        regression_threshold: float = 30.0,
    ):
        """Initialize the Elo monitor.

        Args:
            history_file: Path to store monitoring history
            alert_hours: Hours without update before stagnation alert
            regression_threshold: Elo drop threshold for regression alert
        """
        self.history_file = history_file
        self.alert_hours = alert_hours
        self.regression_threshold = regression_threshold
        self.metrics = get_metrics_logger("elo_monitor", log_interval=300)
        self._history: dict[str, Any] = {}

    def load_history(self) -> dict[str, Any]:
        """Load monitoring history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self._history = json.load(f)
                    logger.debug(f"Loaded history from {self.history_file}")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = {"configs": {}, "last_check": None}
        else:
            self._history = {"configs": {}, "last_check": None}

        return self._history

    def save_history(self) -> None:
        """Save monitoring history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, "w") as f:
                json.dump(self._history, f, indent=2, default=str)
            logger.debug(f"Saved history to {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def get_current_ratings(self) -> dict[str, EloRating]:
        """Get current Elo ratings from all sources."""
        ratings: dict[str, EloRating] = {}

        # Load from Elo rating files
        for elo_file in self.ELO_FILE_LOCATIONS:
            if elo_file.exists():
                ratings.update(self._load_ratings_from_file(elo_file))

        # Also check training logs for Elo updates
        ratings.update(self._extract_ratings_from_logs())

        self.metrics.set("configs_tracked", len(ratings))
        return ratings

    def _load_ratings_from_file(self, elo_file: Path) -> dict[str, EloRating]:
        """Load ratings from a single Elo file."""
        ratings: dict[str, EloRating] = {}

        try:
            with open(elo_file) as f:
                data = json.load(f)

            for config_key, info in data.items():
                if config_key not in ratings:
                    ratings[config_key] = EloRating(
                        config_key=config_key,
                        elo=info.get("elo", info.get("rating", 1500)),
                        games=info.get("games", info.get("total_games", 0)),
                        last_update=info.get("last_update", info.get("updated_at")),
                        source=str(elo_file),
                    )
            logger.debug(f"Loaded {len(ratings)} ratings from {elo_file}")

        except Exception as e:
            logger.warning(f"Could not load {elo_file}: {e}")

        return ratings

    def _extract_ratings_from_logs(self) -> dict[str, EloRating]:
        """Extract Elo ratings from training log files."""
        ratings: dict[str, EloRating] = {}
        log_dir = Path("logs")

        if not log_dir.exists():
            return ratings

        for log_file in log_dir.glob("training_*.log"):
            try:
                with open(log_file) as f:
                    content = f.read()

                # Look for Elo updates in last 10k chars
                matches = re.findall(r'Elo[:\s]+(\d+\.?\d*)', content[-10000:])
                if not matches:
                    continue

                # Extract config from filename
                config_match = re.search(r'training_(\w+)_(\d+p)', log_file.name)
                if config_match:
                    config_key = f"{config_match.group(1)}_{config_match.group(2)}"
                    if config_key not in ratings:
                        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                        ratings[config_key] = EloRating(
                            config_key=config_key,
                            elo=float(matches[-1]),
                            games=0,
                            last_update=mtime.isoformat(),
                            source=str(log_file),
                        )

            except Exception as e:
                logger.debug(f"Error reading log {log_file}: {e}")

        return ratings

    def check_training_activity(self) -> dict[str, TrainingActivity]:
        """Check recent training activity from logs."""
        activity: dict[str, TrainingActivity] = {}
        log_dir = Path("logs")

        if not log_dir.exists():
            return activity

        now = datetime.now()

        for log_file in log_dir.glob("training_*.log"):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                age_hours = (now - mtime).total_seconds() / 3600

                config_match = re.search(r'training_(\w+_\d+p)', log_file.name)
                if config_match:
                    config_key = config_match.group(1)

                    # Active if modified in last 5 minutes
                    is_active = age_hours < (5 / 60)

                    activity[config_key] = TrainingActivity(
                        config_key=config_key,
                        log_file=str(log_file),
                        last_modified=mtime.isoformat(),
                        age_hours=round(age_hours, 2),
                        is_active=is_active,
                    )

            except Exception as e:
                logger.debug(f"Error checking {log_file}: {e}")

        active_count = sum(1 for a in activity.values() if a.is_active)
        self.metrics.set("active_training_jobs", active_count)
        self.metrics.set("total_training_configs", len(activity))

        return activity

    def analyze(
        self,
        ratings: dict[str, EloRating],
    ) -> list[EloAlert]:
        """Analyze Elo status and generate alerts."""
        alerts: list[EloAlert] = []
        now = datetime.now(timezone.utc)

        if "configs" not in self._history:
            self._history["configs"] = {}

        for config_key, rating in ratings.items():
            # Get historical data
            hist = self._history["configs"].get(config_key, {})
            prev_elo = hist.get("elo", rating.elo)
            peak_elo = hist.get("peak_elo", rating.elo)
            stagnant_since = hist.get("stagnant_since")

            # Check for regression
            elo_drop = prev_elo - rating.elo
            if elo_drop > self.regression_threshold:
                alerts.append(EloAlert(
                    alert_type=AlertType.ELO_REGRESSION,
                    severity=AlertSeverity.ERROR,
                    config_key=config_key,
                    message=f"Elo dropped {prev_elo:.1f} -> {rating.elo:.1f} ({elo_drop:.1f} points)",
                    elo=rating.elo,
                    details={"prev_elo": prev_elo, "drop": elo_drop},
                ))
                self.metrics.increment("alerts_regression")

            # Check for stagnation
            if rating.last_update:
                hours_since = self._hours_since_update(rating.last_update, now)
                if hours_since is not None and hours_since > self.alert_hours:
                    alerts.append(EloAlert(
                        alert_type=AlertType.ELO_STAGNATION,
                        severity=AlertSeverity.WARNING,
                        config_key=config_key,
                        message=f"No updates for {hours_since:.1f} hours (threshold: {self.alert_hours}h)",
                        elo=rating.elo,
                        details={"hours_since": hours_since},
                    ))
                    self.metrics.increment("alerts_stagnation")

            # Check if below peak
            below_peak = peak_elo - rating.elo
            if below_peak > 50:
                alerts.append(EloAlert(
                    alert_type=AlertType.MODEL_DEGRADATION,
                    severity=AlertSeverity.INFO,
                    config_key=config_key,
                    message=f"Current {rating.elo:.1f} is {below_peak:.1f} below peak {peak_elo:.1f}",
                    elo=rating.elo,
                    details={"peak_elo": peak_elo, "below_peak": below_peak},
                ))

            # Update history
            is_stagnant = abs(rating.elo - prev_elo) < 5
            self._history["configs"][config_key] = {
                "elo": rating.elo,
                "peak_elo": max(peak_elo, rating.elo),
                "last_check": now.isoformat(),
                "stagnant_since": stagnant_since if is_stagnant else None,
            }

        self._history["last_check"] = now.isoformat()
        self.metrics.set("total_alerts", len(alerts))

        return alerts

    def _hours_since_update(
        self,
        last_update: str,
        now: datetime,
    ) -> float | None:
        """Calculate hours since last update."""
        try:
            if isinstance(last_update, str):
                last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            else:
                last_dt = last_update

            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)

            return (now - last_dt).total_seconds() / 3600

        except Exception:
            return None


class StatusReporter:
    """Formats and outputs status reports."""

    # ANSI colors
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        """Initialize reporter.

        Args:
            use_colors: Whether to use ANSI colors in output
        """
        self.use_colors = use_colors and sys.stdout.isatty()

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['end']}"

    def print_report(
        self,
        ratings: dict[str, EloRating],
        activity: dict[str, TrainingActivity],
        alerts: list[EloAlert],
    ) -> None:
        """Print a formatted status report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"ELO MONITORING REPORT - {timestamp}"

        print(f"\n{self._color('=' * 60, 'bold')}")
        print(f"  {self._color(header, 'bold')}")
        print(f"{self._color('=' * 60, 'bold')}\n")

        self._print_ratings(ratings)
        self._print_activity(activity)
        self._print_alerts(alerts)

        print(f"{'=' * 60}\n")

    def _print_ratings(self, ratings: dict[str, EloRating]) -> None:
        """Print current Elo ratings section."""
        print(f"{self._color('Current Elo Ratings:', 'blue')}")
        print("-" * 50)

        sorted_ratings = sorted(
            ratings.values(),
            key=lambda x: x.elo,
            reverse=True,
        )

        for rating in sorted_ratings:
            # Color based on Elo
            if rating.elo >= 1800:
                color = "green"
            elif rating.elo >= 1600:
                color = "yellow"
            else:
                color = "red"

            elo_str = self._color(f"{rating.elo:7.1f}", color)
            print(f"  {rating.config_key:20s} {elo_str} Elo  ({rating.games:5d} games)")

    def _print_activity(self, activity: dict[str, TrainingActivity]) -> None:
        """Print training activity section."""
        print(f"\n{self._color('Training Activity:', 'blue')}")
        print("-" * 50)

        if not activity:
            print(f"  {self._color('No training logs found', 'yellow')}")
            return

        for act in sorted(activity.values(), key=lambda x: x.config_key):
            if act.is_active:
                status = self._color("ACTIVE", "green")
            else:
                status = self._color(f"idle {act.age_hours:.1f}h", "yellow")
            print(f"  {act.config_key:20s} {status}")

    def _print_alerts(self, alerts: list[EloAlert]) -> None:
        """Print alerts section."""
        if not alerts:
            print(f"\n{self._color('No alerts - all systems healthy', 'green')}")
            return

        print(f"\n{self._color('ALERTS:', 'red')}")
        print("-" * 50)

        severity_colors = {
            AlertSeverity.HIGH: "red",
            AlertSeverity.MEDIUM: "yellow",
            AlertSeverity.LOW: "blue",
        }
        severity_order = {
            AlertSeverity.HIGH: 0,
            AlertSeverity.MEDIUM: 1,
            AlertSeverity.LOW: 2,
        }

        sorted_alerts = sorted(alerts, key=lambda x: severity_order.get(x.severity, 3))

        for alert in sorted_alerts:
            color = severity_colors.get(alert.severity, "end")
            severity_label = self._color(f"[{alert.severity.value.upper()}]", color)
            print(f"  {severity_label} {alert.config_key}: {alert.message}")

    def to_json(
        self,
        ratings: dict[str, EloRating],
        activity: dict[str, TrainingActivity],
        alerts: list[EloAlert],
    ) -> str:
        """Convert report to JSON format."""
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ratings": {k: v.to_dict() for k, v in ratings.items()},
            "activity": {k: v.to_dict() for k, v in activity.items()},
            "alerts": [a.to_dict() for a in alerts],
        }
        return json.dumps(output, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor Elo ratings and alert on issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Basic monitoring
  %(prog)s --alert-hours 12             # Alert after 12h stagnation
  %(prog)s --json                       # Output as JSON
  %(prog)s --regression-threshold 50    # Alert on 50+ Elo drop
        """,
    )

    parser.add_argument(
        "--alert-hours",
        type=float,
        default=6,
        help="Hours without update before alerting (default: 6)",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=30,
        help="Elo drop threshold for alert (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--history-file",
        type=str,
        default="data/elo_monitor_history.json",
        help="History file path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON format for log files",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_script_logging(
        script_name="elo_monitor",
        level=log_level,
        json_logs=args.json_logs,
    )

    logger.info("Starting Elo monitor")

    # Initialize monitor
    monitor = EloMonitor(
        history_file=Path(args.history_file),
        alert_hours=args.alert_hours,
        regression_threshold=args.regression_threshold,
    )

    # Load history
    monitor.load_history()

    # Get current state
    ratings = monitor.get_current_ratings()
    activity = monitor.check_training_activity()

    # Analyze and generate alerts
    alerts = monitor.analyze(ratings)

    # Save updated history
    monitor.save_history()

    # Output
    reporter = StatusReporter(use_colors=not args.json)

    if args.json:
        print(reporter.to_json(ratings, activity, alerts))
    else:
        reporter.print_report(ratings, activity, alerts)

    # Log summary
    high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
    logger.info(
        f"Monitor complete: {len(ratings)} configs, {len(alerts)} alerts "
        f"({len(high_alerts)} high severity)"
    )

    # Exit with error code if high severity alerts
    if high_alerts:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

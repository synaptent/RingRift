#!/usr/bin/env python3
"""Daemon Health Monitor - Monitors background daemons and alerts on failures.

Runs as a standalone process to detect:
- Daemon crashes
- Health check failures
- Restart loops
- Unresponsive daemons

Usage:
    # Run in foreground
    python scripts/daemon_health_monitor.py

    # Run in background
    python scripts/daemon_health_monitor.py &

    # Custom check interval (seconds)
    python scripts/daemon_health_monitor.py --interval 30

    # One-shot mode (check once and exit)
    python scripts/daemon_health_monitor.py --once

Environment Variables:
    RINGRIFT_SLACK_WEBHOOK - Slack webhook URL for alerts
    RINGRIFT_MONITOR_INTERVAL - Check interval in seconds (default: 60)
    RINGRIFT_MONITOR_LOG_FILE - Log file path (default: logs/daemon_monitor.log)
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add ai-service to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Configure logging
LOG_FILE = Path(os.environ.get(
    "RINGRIFT_MONITOR_LOG_FILE",
    AI_SERVICE_ROOT / "logs" / "daemon_monitor.log"
))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


def get_slack_webhook() -> str | None:
    """Get Slack webhook URL from environment or config file."""
    webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK") or os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        return webhook
    webhook_file = Path.home() / ".ringrift_slack_webhook"
    if webhook_file.exists():
        return webhook_file.read_text().strip()
    return None


def send_slack_alert(message: str, alert_type: str = "error") -> bool:
    """Send alert to Slack webhook if configured."""
    webhook = get_slack_webhook()
    if not webhook:
        return False

    emoji = {
        "error": ":x:",
        "warning": ":warning:",
        "info": ":information_source:",
        "recovery": ":white_check_mark:",
    }.get(alert_type, ":robot_face:")

    payload = {
        "text": f"{emoji} *Daemon Monitor*\n{message}",
        "username": "Daemon Monitor",
    }

    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
             "-d", json.dumps(payload), webhook],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Failed to send Slack alert: {e}")
        return False


class DaemonHealthMonitor:
    """Monitors daemon health and alerts on failures."""

    def __init__(self, check_interval: float = 60.0):
        """Initialize monitor.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self._last_unhealthy: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}
        self._alert_cooldown: dict[str, datetime] = {}

    def get_daemon_health(self) -> dict[str, dict[str, Any]]:
        """Get health status of all daemons.

        Returns:
            Dict mapping daemon names to health status dicts
        """
        try:
            from app.coordination.daemon_manager import get_daemon_manager
            dm = get_daemon_manager()
            return dm.get_all_daemon_health()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get daemon health: {e}")
            return {}

    def check_http_health(self) -> dict[str, Any]:
        """Check HTTP health endpoint.

        Returns:
            Health response or error dict
        """
        import urllib.request
        import urllib.error

        try:
            with urllib.request.urlopen("http://localhost:8790/health", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            return {"healthy": False, "error": str(e)}
        except (json.JSONDecodeError, OSError) as e:
            return {"healthy": False, "error": str(e)}

    def _should_alert(self, daemon_name: str) -> bool:
        """Check if we should send an alert for this daemon.

        Implements cooldown to avoid alert storms.
        """
        cooldown_minutes = 15
        last_alert = self._alert_cooldown.get(daemon_name)
        if last_alert:
            elapsed = (datetime.now() - last_alert).total_seconds() / 60
            if elapsed < cooldown_minutes:
                return False
        return True

    def _record_alert(self, daemon_name: str):
        """Record that we sent an alert for this daemon."""
        self._alert_cooldown[daemon_name] = datetime.now()

    def run_check(self) -> dict[str, Any]:
        """Run a single health check cycle.

        Returns:
            Dict with check results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "healthy": True,
            "unhealthy_daemons": [],
            "recovered_daemons": [],
            "total_daemons": 0,
        }

        # Get daemon health
        daemon_health = self.get_daemon_health()
        results["total_daemons"] = len(daemon_health)

        current_unhealthy: set[str] = set()

        for daemon_name, health in daemon_health.items():
            is_healthy = health.get("healthy", True)

            if not is_healthy:
                current_unhealthy.add(daemon_name)
                self._consecutive_failures[daemon_name] = (
                    self._consecutive_failures.get(daemon_name, 0) + 1
                )

                results["unhealthy_daemons"].append({
                    "name": daemon_name,
                    "status": health.get("status", "unknown"),
                    "message": health.get("message", ""),
                    "consecutive_failures": self._consecutive_failures[daemon_name],
                })

                # Alert on first failure or after 3 consecutive failures
                failures = self._consecutive_failures[daemon_name]
                if failures == 1 or failures == 3:
                    if self._should_alert(daemon_name):
                        self._send_unhealthy_alert(daemon_name, health, failures)
                        self._record_alert(daemon_name)

            else:
                # Check for recovery
                if daemon_name in self._last_unhealthy:
                    results["recovered_daemons"].append(daemon_name)
                    self._send_recovery_alert(daemon_name)

                self._consecutive_failures[daemon_name] = 0

        # Track unhealthy state changes
        self._last_unhealthy = current_unhealthy
        results["healthy"] = len(current_unhealthy) == 0

        return results

    def _send_unhealthy_alert(
        self,
        daemon_name: str,
        health: dict[str, Any],
        failures: int,
    ):
        """Send alert for unhealthy daemon."""
        severity = "error" if failures >= 3 else "warning"
        message = (
            f"Daemon `{daemon_name}` is unhealthy\n"
            f"  Status: {health.get('status', 'unknown')}\n"
            f"  Message: {health.get('message', 'N/A')}\n"
            f"  Consecutive failures: {failures}"
        )
        logger.error(message)
        send_slack_alert(message, severity)

    def _send_recovery_alert(self, daemon_name: str):
        """Send alert for daemon recovery."""
        message = f"Daemon `{daemon_name}` has recovered"
        logger.info(message)
        send_slack_alert(message, "recovery")

    async def run_forever(self):
        """Run continuous health monitoring."""
        logger.info(f"Starting daemon health monitor (interval: {self.check_interval}s)")

        while True:
            try:
                results = self.run_check()

                if results["healthy"]:
                    logger.debug(
                        f"All {results['total_daemons']} daemons healthy"
                    )
                else:
                    unhealthy_names = [d["name"] for d in results["unhealthy_daemons"]]
                    logger.warning(
                        f"Unhealthy daemons: {unhealthy_names}"
                    )

                if results["recovered_daemons"]:
                    logger.info(
                        f"Recovered daemons: {results['recovered_daemons']}"
                    )

            except (KeyboardInterrupt, SystemExit):
                logger.info("Shutting down daemon health monitor")
                break
            except Exception as e:
                logger.exception(f"Error in health check cycle: {e}")

            await asyncio.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor daemon health and alert on failures",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("RINGRIFT_MONITOR_INTERVAL", "60")),
        help="Check interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (with --once)",
    )
    args = parser.parse_args()

    monitor = DaemonHealthMonitor(check_interval=args.interval)

    if args.once:
        results = monitor.run_check()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"Daemon Health Check - {results['timestamp']}")
            print(f"  Total daemons: {results['total_daemons']}")
            print(f"  Status: {'HEALTHY' if results['healthy'] else 'UNHEALTHY'}")
            if results["unhealthy_daemons"]:
                print("  Unhealthy:")
                for d in results["unhealthy_daemons"]:
                    print(f"    - {d['name']}: {d['message']}")
        sys.exit(0 if results["healthy"] else 1)
    else:
        try:
            asyncio.run(monitor.run_forever())
        except KeyboardInterrupt:
            logger.info("Interrupted by user")


if __name__ == "__main__":
    main()

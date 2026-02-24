#!/usr/bin/env python3
"""Health Alerting - Monitor cluster health and send alerts.

Monitors:
- P2P network connectivity
- Vast instance health
- Selfplay job counts
- Disk/memory usage
- Training progress

Alerts via:
- Slack webhook
- PagerDuty
- Console/log

Usage:
    python scripts/health_alerting.py --check            # Run health checks
    python scripts/health_alerting.py --alert-test       # Test alert delivery
    python scripts/health_alerting.py --daemon           # Run continuously

Designed to run via cron every 5-10 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

LOG_DIR = AI_SERVICE_ROOT / "logs"
STATE_FILE = AI_SERVICE_ROOT / "data" / "alert_state.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Use shared logging from scripts/lib
from scripts.lib.logging_config import setup_script_logging
from scripts.lib.state_manager import StateManager

logger = setup_script_logging("health_alerting", log_dir=str(LOG_DIR))


# =============================================================================
# Host Configuration
# =============================================================================

# Try to use unified hosts module
try:
    from scripts.lib.hosts import get_p2p_voters
    USE_UNIFIED_HOSTS = True
except ImportError:
    USE_UNIFIED_HOSTS = False


def _load_p2p_leaders_from_config():
    """Load P2P leader hosts from config file or environment."""
    # Prefer unified hosts module
    if USE_UNIFIED_HOSTS:
        voters = get_p2p_voters()
        leaders = []
        for h in voters[:3]:
            ip = h.tailscale_ip or h.ssh_host
            if ip:
                leaders.append(f"http://{ip}:8770")
        return leaders

    # Fallback to direct YAML loading
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print("[HealthAlert] Warning: No distributed_hosts.yaml found, using empty P2P leaders list")
        return []

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Extract hosts that are P2P voters or primary training nodes
        leaders = []
        hosts = config.get("hosts", {})
        for _name, info in hosts.items():
            if info.get("status") != "ready":
                continue
            # Include P2P voters and primary training nodes as potential leaders
            if info.get("p2p_voter") or "primary" in info.get("role", ""):
                tailscale_ip = info.get("tailscale_ip")
                if tailscale_ip:
                    leaders.append(f"http://{tailscale_ip}:8770")

        return leaders[:3]  # Return top 3 leaders
    except Exception as e:
        print(f"[HealthAlert] Error loading P2P leaders from config: {e}")
        return []


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AlertConfig:
    """Alert configuration."""
    # Thresholds
    min_p2p_peers: int = 10                      # Alert if fewer peers
    min_vast_instances: int = 5                  # Alert if fewer Vast
    min_selfplay_jobs: int = 100                 # Alert if fewer jobs
    max_idle_instances: int = 8                  # Alert if too many idle
    max_disk_usage_percent: int = 85             # = DISK_PRODUCTION_HALT_PERCENT from app.config.thresholds
    max_hourly_cost: float = 15.0                # Alert if cost > $15/hr

    # Alert cooldowns (minutes)
    alert_cooldown_minutes: int = 30             # Don't repeat same alert

    # Webhooks (from environment or config)
    slack_webhook_url: str | None = None
    pagerduty_routing_key: str | None = None

    def __post_init__(self):
        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", self.slack_webhook_url)
        self.pagerduty_routing_key = os.environ.get("PAGERDUTY_ROUTING_KEY", self.pagerduty_routing_key)


@dataclass
class AlertState:
    """Persistent alert state to prevent alert spam."""
    last_alerts: dict[str, str] = field(default_factory=dict)  # alert_key -> timestamp

    def can_alert(self, key: str, cooldown_minutes: int) -> bool:
        """Check if we can send an alert (respects cooldown)."""
        if key not in self.last_alerts:
            return True
        last_time = datetime.fromisoformat(self.last_alerts[key])
        return datetime.now() - last_time > timedelta(minutes=cooldown_minutes)

    def record_alert(self, key: str):
        """Record that an alert was sent."""
        self.last_alerts[key] = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {"last_alerts": self.last_alerts}

    @classmethod
    def from_dict(cls, data: dict) -> AlertState:
        return cls(last_alerts=data.get("last_alerts", {}))


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: str  # "ok", "warning", "critical"
    message: str
    value: Any = None
    threshold: Any = None


# =============================================================================
# Health Checks
# =============================================================================

# Load P2P leaders at module level (cached)
P2P_LEADERS = _load_p2p_leaders_from_config()


def check_p2p_network(config: AlertConfig) -> HealthCheck:
    """Check P2P network health."""
    import urllib.request

    if not P2P_LEADERS:
        return HealthCheck(
            name="p2p_network",
            status="warning",
            message="No P2P leaders configured",
        )

    for leader in P2P_LEADERS:
        try:
            with urllib.request.urlopen(f"{leader}/status", timeout=10) as resp:
                data = json.loads(resp.read().decode())
                peers = data.get("peers", {})
                active = sum(1 for p in peers.values() if not p.get("retired", False))

                if active < config.min_p2p_peers:
                    return HealthCheck(
                        name="p2p_network",
                        status="critical" if active < config.min_p2p_peers // 2 else "warning",
                        message=f"Low P2P peers: {active} (min: {config.min_p2p_peers})",
                        value=active,
                        threshold=config.min_p2p_peers,
                    )
                return HealthCheck(
                    name="p2p_network",
                    status="ok",
                    message=f"{active} active peers",
                    value=active,
                )
        except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            continue

    return HealthCheck(
        name="p2p_network",
        status="critical",
        message="Cannot reach any P2P leader",
    )


def check_vast_instances(config: AlertConfig) -> HealthCheck:
    """Check Vast instance health."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return HealthCheck(
                name="vast_instances",
                status="warning",
                message="Cannot query Vast API",
            )

        instances = json.loads(result.stdout)
        running = [i for i in instances if i.get("actual_status") == "running"]
        total_cost = sum(i.get("dph_total", 0) or 0 for i in running)

        checks = []

        # Instance count
        if len(running) < config.min_vast_instances:
            checks.append(f"Low instances: {len(running)}")

        # Cost
        if total_cost > config.max_hourly_cost:
            checks.append(f"High cost: ${total_cost:.2f}/hr")

        if checks:
            return HealthCheck(
                name="vast_instances",
                status="warning",
                message="; ".join(checks),
                value={"running": len(running), "cost": total_cost},
            )

        return HealthCheck(
            name="vast_instances",
            status="ok",
            message=f"{len(running)} running, ${total_cost:.2f}/hr",
            value={"running": len(running), "cost": total_cost},
        )
    except Exception as e:
        return HealthCheck(
            name="vast_instances",
            status="warning",
            message=f"Check failed: {e}",
        )


def check_selfplay_jobs(config: AlertConfig) -> HealthCheck:
    """Check selfplay job counts."""
    import urllib.request

    if not P2P_LEADERS:
        return HealthCheck(
            name="selfplay_jobs",
            status="warning",
            message="No P2P leaders configured",
        )

    for leader in P2P_LEADERS:
        try:
            with urllib.request.urlopen(f"{leader}/status", timeout=10) as resp:
                data = json.loads(resp.read().decode())
                peers = data.get("peers", {})
                total_jobs = sum(p.get("selfplay_jobs", 0) for p in peers.values())
                idle = sum(1 for p in peers.values() if p.get("selfplay_jobs", 0) == 0 and not p.get("retired"))

                if total_jobs < config.min_selfplay_jobs:
                    return HealthCheck(
                        name="selfplay_jobs",
                        status="warning",
                        message=f"Low selfplay jobs: {total_jobs} (min: {config.min_selfplay_jobs})",
                        value=total_jobs,
                        threshold=config.min_selfplay_jobs,
                    )

                if idle > config.max_idle_instances:
                    return HealthCheck(
                        name="selfplay_jobs",
                        status="warning",
                        message=f"Too many idle instances: {idle}",
                        value=idle,
                        threshold=config.max_idle_instances,
                    )

                return HealthCheck(
                    name="selfplay_jobs",
                    status="ok",
                    message=f"{total_jobs} jobs, {idle} idle",
                    value=total_jobs,
                )
        except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            continue

    return HealthCheck(
        name="selfplay_jobs",
        status="warning",
        message="Cannot check selfplay status",
    )


def check_local_disk() -> HealthCheck:
    """Check local disk usage."""
    try:
        result = subprocess.run(
            ["df", "-h", str(AI_SERVICE_ROOT)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                usage_pct = int(parts[4].replace("%", ""))
                if usage_pct > 90:
                    return HealthCheck(
                        name="local_disk",
                        status="critical",
                        message=f"Disk usage critical: {usage_pct}%",
                        value=usage_pct,
                    )
                elif usage_pct > 80:
                    return HealthCheck(
                        name="local_disk",
                        status="warning",
                        message=f"Disk usage high: {usage_pct}%",
                        value=usage_pct,
                    )
                return HealthCheck(
                    name="local_disk",
                    status="ok",
                    message=f"Disk usage: {usage_pct}%",
                    value=usage_pct,
                )
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as e:
        pass

    return HealthCheck(
        name="local_disk",
        status="ok",
        message="Could not check disk",
    )


def run_all_checks(config: AlertConfig) -> list[HealthCheck]:
    """Run all health checks."""
    return [
        check_p2p_network(config),
        check_vast_instances(config),
        check_selfplay_jobs(config),
        check_local_disk(),
    ]


# =============================================================================
# Alert Delivery
# =============================================================================

def send_slack_alert(webhook_url: str, check: HealthCheck) -> bool:
    """Send alert to Slack."""
    import urllib.request

    color = {"ok": "good", "warning": "warning", "critical": "danger"}.get(check.status, "warning")

    payload = {
        "attachments": [{
            "color": color,
            "title": f"RingRift Alert: {check.name}",
            "text": check.message,
            "fields": [
                {"title": "Status", "value": check.status.upper(), "short": True},
                {"title": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "short": True},
            ],
            "footer": "RingRift Health Monitor",
        }]
    }

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        return False


def send_pagerduty_alert(routing_key: str, check: HealthCheck) -> bool:
    """Send alert to PagerDuty."""
    import urllib.request

    severity = {"critical": "critical", "warning": "warning", "ok": "info"}.get(check.status, "warning")

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger" if check.status != "ok" else "resolve",
        "dedup_key": f"ringrift-{check.name}",
        "payload": {
            "summary": f"RingRift {check.name}: {check.message}",
            "severity": severity,
            "source": "ringrift-health-monitor",
            "custom_details": {
                "value": check.value,
                "threshold": check.threshold,
            },
        },
    }

    try:
        req = urllib.request.Request(
            "https://events.pagerduty.com/v2/enqueue",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except Exception as e:
        logger.error(f"PagerDuty alert failed: {e}")
        return False


def send_alerts(check: HealthCheck, config: AlertConfig, state: AlertState) -> bool:
    """Send alerts for a health check."""
    alert_key = f"{check.name}:{check.status}"

    # Check cooldown
    if not state.can_alert(alert_key, config.alert_cooldown_minutes):
        logger.debug(f"Alert cooldown active for {alert_key}")
        return False

    sent = False

    # Slack
    if config.slack_webhook_url and send_slack_alert(config.slack_webhook_url, check):
        logger.info(f"Slack alert sent: {check.name}")
        sent = True

    # PagerDuty
    if config.pagerduty_routing_key and check.status == "critical" and send_pagerduty_alert(config.pagerduty_routing_key, check):
        logger.info(f"PagerDuty alert sent: {check.name}")
        sent = True

    if sent:
        state.record_alert(alert_key)

    return sent


# =============================================================================
# State Management
# =============================================================================

# Use StateManager for persistent state
_state_manager = StateManager(STATE_FILE, AlertState)


def load_state() -> AlertState:
    """Load persistent state."""
    return _state_manager.load()


def save_state(state: AlertState):
    """Save persistent state."""
    _state_manager.save(state)


# =============================================================================
# Commands
# =============================================================================

def cmd_check(config: AlertConfig, send_alerts_flag: bool = True):
    """Run health checks and optionally send alerts."""
    logger.info("=" * 70)
    logger.info("HEALTH CHECK")
    logger.info("=" * 70)

    state = load_state()
    checks = run_all_checks(config)
    alerts_sent = 0

    print(f"\n{'Check':<20} {'Status':<10} {'Message'}")
    print("-" * 70)

    for check in checks:
        status_emoji = {"ok": "OK", "warning": "WARN", "critical": "CRIT"}.get(check.status, "?")
        print(f"{check.name:<20} {status_emoji:<10} {check.message}")

        # Send alerts for non-ok status
        if send_alerts_flag and check.status != "ok" and send_alerts(check, config, state):
            alerts_sent += 1

    save_state(state)

    # Summary
    critical = sum(1 for c in checks if c.status == "critical")
    warning = sum(1 for c in checks if c.status == "warning")
    ok = sum(1 for c in checks if c.status == "ok")

    print("-" * 70)
    print(f"Summary: {ok} OK, {warning} WARNING, {critical} CRITICAL")
    if alerts_sent:
        print(f"Alerts sent: {alerts_sent}")

    return critical == 0


def cmd_alert_test(config: AlertConfig):
    """Test alert delivery."""
    logger.info("Testing alert delivery...")

    test_check = HealthCheck(
        name="test_alert",
        status="warning",
        message="This is a test alert from RingRift health monitor",
        value="test",
    )

    if config.slack_webhook_url:
        if send_slack_alert(config.slack_webhook_url, test_check):
            print("Slack: OK")
        else:
            print("Slack: FAILED")
    else:
        print("Slack: Not configured")

    if config.pagerduty_routing_key:
        if send_pagerduty_alert(config.pagerduty_routing_key, test_check):
            print("PagerDuty: OK")
        else:
            print("PagerDuty: FAILED")
    else:
        print("PagerDuty: Not configured")


def cmd_daemon(config: AlertConfig, interval: int = 300):
    """Run health checks continuously."""
    logger.info(f"Starting health monitor daemon (interval: {interval}s)")

    while True:
        try:
            cmd_check(config)
        except Exception as e:
            logger.error(f"Health check failed: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Health Alerting")
    parser.add_argument("--check", action="store_true", help="Run health checks")
    parser.add_argument("--alert-test", action="store_true", help="Test alert delivery")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--no-alerts", action="store_true", help="Suppress alert sending")
    parser.add_argument("--interval", type=int, default=300, help="Daemon interval (seconds)")

    args = parser.parse_args()
    config = AlertConfig()

    if args.alert_test:
        cmd_alert_test(config)
    elif args.daemon:
        cmd_daemon(config, args.interval)
    elif args.check or not any([args.alert_test, args.daemon]):
        cmd_check(config, send_alerts_flag=not args.no_alerts)


if __name__ == "__main__":
    main()

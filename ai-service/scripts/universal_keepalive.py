#!/usr/bin/env python3
"""Universal Keepalive Daemon - Keeps cluster nodes online across all platforms.

This script provides unified keepalive functionality for all node types:
- Vast.ai instances: Prevents idle termination, monitors workers
- Mac nodes: Prevents sleep, monitors Tailscale
- Lambda/cloud: Monitors connectivity, restarts services
- Hetzner/bare metal: Service monitoring

Features:
- Platform detection and appropriate keepalive methods
- P2P orchestrator health monitoring
- Tailscale connectivity validation
- Automatic service restart on failure
- Heartbeat pings to prevent network timeouts

Usage:
    python scripts/universal_keepalive.py --node-id my-node

    # As systemd service or launchd daemon:
    python scripts/universal_keepalive.py --node-id my-node --daemon
"""

import argparse
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CHECK_INTERVAL = 30  # seconds between health checks
P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))
LAMBDA_IPS = ["100.97.104.89", "100.91.25.13"]  # Primary Lambda nodes for connectivity check

# Notification settings
SLACK_WEBHOOK_URL = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
NOTIFICATION_COOLDOWN = 300  # 5 minutes between repeated alerts
_last_notification_time: dict[str, float] = {}


def send_notification(event_type: str, message: str, node_id: str, severity: str = "warning") -> bool:
    """Send notification via Slack webhook.

    Args:
        event_type: Type of event (p2p_down, tailscale_down, node_offline, etc.)
        message: Human-readable message
        node_id: Affected node identifier
        severity: warning, error, or info

    Returns:
        True if notification sent successfully
    """
    global _last_notification_time

    # Check cooldown
    cache_key = f"{node_id}:{event_type}"
    now = time.time()
    if cache_key in _last_notification_time and now - _last_notification_time[cache_key] < NOTIFICATION_COOLDOWN:
        return False  # Skip due to cooldown

    if not SLACK_WEBHOOK_URL:
        logger.debug(f"No Slack webhook configured, skipping notification: {message}")
        return False

    # Color based on severity
    colors = {"error": "#ff0000", "warning": "#ffaa00", "info": "#00aa00"}
    color = colors.get(severity, "#808080")

    # Build Slack payload
    payload = {
        "attachments": [{
            "color": color,
            "title": f"RingRift Cluster Alert: {event_type}",
            "text": message,
            "fields": [
                {"title": "Node", "value": node_id, "short": True},
                {"title": "Severity", "value": severity.upper(), "short": True},
                {"title": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
            ],
            "footer": "RingRift Keepalive Daemon"
        }]
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                _last_notification_time[cache_key] = now
                logger.info(f"Sent notification: {event_type}")
                return True
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")

    return False


def send_recovery_notification(event_type: str, node_id: str) -> bool:
    """Send recovery notification when an issue is resolved."""
    message = f"Node {node_id} has recovered from {event_type}"
    return send_notification(f"{event_type}_recovered", message, node_id, severity="info")


class NodeType:
    """Detected node types."""
    MAC = "mac"
    VAST = "vast"
    LAMBDA = "lambda"
    LINUX = "linux"
    UNKNOWN = "unknown"


def detect_node_type() -> str:
    """Detect what type of node we're running on."""
    system = platform.system().lower()

    if system == "darwin":
        return NodeType.MAC

    # Check for Vast.ai indicators
    if os.path.exists("/workspace") or "vast" in socket.gethostname().lower():
        return NodeType.VAST

    # Check for Lambda indicators
    hostname = socket.gethostname().lower()
    if "lambda" in hostname or "gh200" in hostname or "h100" in hostname:
        return NodeType.LAMBDA

    if system == "linux":
        return NodeType.LINUX

    return NodeType.UNKNOWN


def get_ai_service_root() -> Path:
    """Get the AI service root directory."""
    # Try common locations
    candidates = [
        Path(__file__).parent.parent,  # Relative to script
        Path.home() / "ringrift" / "ai-service",
        Path.home() / "Development" / "RingRift" / "ai-service",
        Path("/workspace/ringrift/ai-service"),
    ]

    for path in candidates:
        if path.exists() and (path / "scripts").exists():
            return path

    return Path(__file__).parent.parent


class UniversalKeepalive:
    """Universal keepalive daemon for all node types."""

    def __init__(self, node_id: str, daemon_mode: bool = False):
        self.node_id = node_id
        self.daemon_mode = daemon_mode
        self.node_type = detect_node_type()
        self.ai_service_root = get_ai_service_root()
        self.running = True
        self.caffeinate_pid: int | None = None

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(f"Initialized keepalive for {node_id} (type: {self.node_type})")
        logger.info(f"AI service root: {self.ai_service_root}")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def check_p2p_health(self) -> tuple[bool, dict | None]:
        """Check if P2P orchestrator is healthy."""
        try:
            url = f"http://localhost:{P2P_PORT}/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return True, data
        except Exception as e:
            logger.debug(f"P2P health check failed: {e}")
            return False, None

    def is_p2p_running(self) -> bool:
        """Check if P2P orchestrator process is running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "p2p_orchestrator"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def start_p2p(self) -> bool:
        """Start the P2P orchestrator."""
        logger.info("Starting P2P orchestrator...")

        try:
            # Check for systemd service first
            if self.node_type in [NodeType.VAST, NodeType.LAMBDA, NodeType.LINUX]:
                result = subprocess.run(
                    ["systemctl", "is-enabled", "ringrift-p2p.service"],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["sudo", "systemctl", "restart", "ringrift-p2p.service"],
                        timeout=30
                    )
                    logger.info("Restarted P2P via systemctl")
                    return True

            # Fallback to direct start
            python_cmd = sys.executable
            p2p_script = self.ai_service_root / "scripts" / "p2p_orchestrator.py"

            if not p2p_script.exists():
                logger.error(f"P2P script not found: {p2p_script}")
                return False

            log_file = self.ai_service_root / "logs" / "p2p_orchestrator.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as log:
                subprocess.Popen(
                    [python_cmd, str(p2p_script), "--node-id", self.node_id],
                    cwd=str(self.ai_service_root),
                    stdout=log,
                    stderr=log,
                    start_new_session=True
                )

            logger.info(f"Started P2P orchestrator with node-id: {self.node_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start P2P: {e}")
            return False

    def check_tailscale(self) -> bool:
        """Check Tailscale connectivity."""
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False

            data = json.loads(result.stdout)
            return data.get("BackendState") == "Running"
        except Exception as e:
            logger.debug(f"Tailscale check failed: {e}")
            return False

    def restart_tailscale(self):
        """Attempt to restart Tailscale."""
        logger.warning("Attempting to restart Tailscale...")
        try:
            if self.node_type == NodeType.MAC:
                subprocess.run(
                    ["sudo", "launchctl", "kickstart", "-k", "system/com.tailscale.tailscaled"],
                    timeout=30
                )
            else:
                subprocess.run(["sudo", "systemctl", "restart", "tailscaled"], timeout=30)
        except Exception as e:
            logger.error(f"Failed to restart Tailscale: {e}")

    def send_keepalive_ping(self) -> bool:
        """Send keepalive ping to maintain connectivity."""
        for ip in LAMBDA_IPS:
            try:
                # Quick ping to keep network alive
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", ip],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return True
            except Exception:
                continue
        return False

    # Mac-specific methods
    def mac_prevent_sleep(self):
        """Prevent Mac from sleeping (Mac only)."""
        if self.node_type != NodeType.MAC:
            return

        if self.caffeinate_pid is not None:
            # Check if still running
            try:
                os.kill(self.caffeinate_pid, 0)
                return  # Still running
            except OSError:
                self.caffeinate_pid = None

        try:
            # Start caffeinate in background
            proc = subprocess.Popen(
                ["caffeinate", "-dis"],  # Prevent display, idle, and system sleep
                start_new_session=True
            )
            self.caffeinate_pid = proc.pid
            logger.info(f"Started caffeinate (PID: {self.caffeinate_pid})")
        except Exception as e:
            logger.error(f"Failed to start caffeinate: {e}")

    def mac_cleanup(self):
        """Clean up Mac-specific resources."""
        if self.caffeinate_pid:
            try:
                os.kill(self.caffeinate_pid, signal.SIGTERM)
                logger.info("Stopped caffeinate")
            except Exception:
                pass

    # Vast-specific methods
    def vast_send_keepalive(self):
        """Send keepalive to prevent Vast.ai idle termination."""
        if self.node_type != NodeType.VAST:
            return

        try:
            # Write keepalive file (Vast monitors for activity)
            keepalive_file = Path("/tmp") / f"keepalive_{int(time.time())}"
            keepalive_file.write_text("alive")

            # Clean up old keepalive files
            for f in Path("/tmp").glob("keepalive_*"):
                try:
                    if f != keepalive_file and time.time() - f.stat().st_mtime > 300:
                        f.unlink()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Vast keepalive write failed: {e}")

    def run_health_check(self) -> dict[str, bool]:
        """Run all health checks and return status."""
        status = {
            "p2p_running": False,
            "p2p_healthy": False,
            "tailscale_ok": False,
            "network_ok": False,
        }

        # Check P2P
        status["p2p_running"] = self.is_p2p_running()
        if status["p2p_running"]:
            healthy, _ = self.check_p2p_health()
            status["p2p_healthy"] = healthy

        # Check Tailscale
        status["tailscale_ok"] = self.check_tailscale()

        # Check network
        status["network_ok"] = self.send_keepalive_ping()

        return status

    def run(self):
        """Main keepalive loop."""
        logger.info(f"Starting keepalive daemon (interval: {CHECK_INTERVAL}s)")

        consecutive_failures = 0
        was_p2p_healthy = True
        was_tailscale_ok = True
        was_network_ok = True

        while self.running:
            try:
                # Platform-specific pre-checks
                if self.node_type == NodeType.MAC:
                    self.mac_prevent_sleep()
                elif self.node_type == NodeType.VAST:
                    self.vast_send_keepalive()

                # Run health checks
                status = self.run_health_check()

                # Log status periodically
                if not all(status.values()):
                    logger.warning(f"Health check: {status}")
                else:
                    logger.debug(f"Health check: all OK")

                # Take corrective actions with notifications
                if not status["p2p_running"]:
                    logger.warning("P2P not running, starting...")
                    send_notification(
                        "p2p_down",
                        f"P2P orchestrator not running on {self.node_id}, attempting restart",
                        self.node_id,
                        severity="error"
                    )
                    self.start_p2p()
                    consecutive_failures += 1
                    was_p2p_healthy = False
                elif not status["p2p_healthy"] and consecutive_failures < 3:
                    # Give it a few cycles before restarting
                    consecutive_failures += 1
                    logger.warning(f"P2P unhealthy ({consecutive_failures}/3)")
                    was_p2p_healthy = False
                elif not status["p2p_healthy"]:
                    logger.warning("P2P unhealthy for too long, restarting...")
                    send_notification(
                        "p2p_unhealthy",
                        f"P2P orchestrator unhealthy on {self.node_id} for {consecutive_failures} checks, force restarting",
                        self.node_id,
                        severity="error"
                    )
                    subprocess.run(["pkill", "-f", "p2p_orchestrator"], timeout=5)
                    time.sleep(2)
                    self.start_p2p()
                    consecutive_failures = 0
                    was_p2p_healthy = False
                else:
                    # P2P is healthy - send recovery if it was down
                    if not was_p2p_healthy:
                        send_recovery_notification("p2p_down", self.node_id)
                    consecutive_failures = 0
                    was_p2p_healthy = True

                # Tailscale monitoring with notifications
                if not status["tailscale_ok"]:
                    if was_tailscale_ok:  # First failure
                        send_notification(
                            "tailscale_down",
                            f"Tailscale connectivity lost on {self.node_id}, attempting restart",
                            self.node_id,
                            severity="error"
                        )
                    self.restart_tailscale()
                    was_tailscale_ok = False
                else:
                    if not was_tailscale_ok:
                        send_recovery_notification("tailscale_down", self.node_id)
                    was_tailscale_ok = True

                # Network monitoring with notifications
                if not status["network_ok"]:
                    if was_network_ok:  # First failure
                        send_notification(
                            "network_unreachable",
                            f"Network connectivity issues on {self.node_id} - cannot reach peer nodes",
                            self.node_id,
                            severity="warning"
                        )
                    was_network_ok = False
                else:
                    if not was_network_ok:
                        send_recovery_notification("network_unreachable", self.node_id)
                    was_network_ok = True

            except Exception as e:
                logger.error(f"Health check error: {e}")
                send_notification(
                    "keepalive_error",
                    f"Keepalive daemon error on {self.node_id}: {str(e)}",
                    self.node_id,
                    severity="error"
                )

            # Sleep until next check
            time.sleep(CHECK_INTERVAL)

        # Cleanup
        if self.node_type == NodeType.MAC:
            self.mac_cleanup()

        logger.info("Keepalive daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="Universal Keepalive Daemon")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    args = parser.parse_args()

    global CHECK_INTERVAL
    CHECK_INTERVAL = args.interval

    keepalive = UniversalKeepalive(args.node_id, args.daemon)
    keepalive.run()


if __name__ == "__main__":
    main()

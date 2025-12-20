#!/usr/bin/env python3
"""Idle GPU Node Alert Script.

Monitors GPU-heavy nodes and sends Slack alerts when GPUs are idle for too long.
Designed to run via cron to ensure GPU resources are utilized.

Usage:
    # Single check
    python scripts/idle_gpu_alert.py

    # Run via cron every 30 minutes
    */30 * * * * cd ~/ringrift/ai-service && python3 scripts/idle_gpu_alert.py >> /tmp/idle_gpu_alert.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.logging_config import get_logger

logger = get_logger(__name__)

P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))

# Alert thresholds
GPU_IDLE_THRESHOLD = 10.0  # GPU is idle if < 10%
IDLE_DURATION_MINUTES = 30  # Alert after 30 minutes of idle
GPU_HEAVY_TAGS = ['gh200', 'h100', 'h200', 'a100', '4090', '5090']

# Persistence file for tracking idle duration
STATE_FILE = Path("/tmp/ringrift_idle_gpu_state.json")


@dataclass
class IdleGPUState:
    """Tracks how long a GPU node has been idle."""
    node_id: str
    first_idle_at: float
    last_check_at: float
    gpu_name: str
    gpu_percent: float
    cpu_percent: float
    external_work: dict[str, bool]


def http_get(url: str, timeout: int = 15) -> dict | None:
    """Make HTTP GET request and return JSON."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return None


def get_slack_webhook_url() -> str | None:
    """Get Slack webhook URL from environment or file."""
    # Try environment variable first
    url = os.environ.get("RINGRIFT_SLACK_WEBHOOK") or os.environ.get("SLACK_WEBHOOK_URL")
    if url:
        return url

    # Try file
    webhook_file = Path.home() / ".ringrift_slack_webhook"
    if webhook_file.exists():
        return webhook_file.read_text().strip()

    return None


def send_slack_alert(webhook_url: str, title: str, message: str, severity: str = "warning", details: dict[str, Any] = None) -> bool:
    """Send alert to Slack."""
    colors = {
        "info": "#2196F3",
        "warning": "#FFC107",
        "error": "#F44336",
        "critical": "#9C27B0",
    }

    fields = []
    if details:
        for k, v in list(details.items())[:6]:
            fields.append({"title": k, "value": str(v), "short": True})

    payload = {
        "attachments": [{
            "color": colors.get(severity, "#808080"),
            "title": title,
            "text": message,
            "ts": time.time(),
            "fields": fields,
            "footer": "RingRift Cluster Monitoring",
        }]
    }

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


def is_gpu_heavy(gpu_name: str) -> bool:
    """Check if this is a GPU-heavy node based on GPU name."""
    gpu_upper = (gpu_name or "").upper()
    return any(tag.upper() in gpu_upper for tag in GPU_HEAVY_TAGS)


def load_state() -> dict[str, IdleGPUState]:
    """Load idle state from file."""
    if not STATE_FILE.exists():
        return {}

    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return {
            node_id: IdleGPUState(
                node_id=node_id,
                first_idle_at=s["first_idle_at"],
                last_check_at=s["last_check_at"],
                gpu_name=s.get("gpu_name", ""),
                gpu_percent=s.get("gpu_percent", 0),
                cpu_percent=s.get("cpu_percent", 0),
                external_work=s.get("external_work", {}),
            )
            for node_id, s in data.items()
        }
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
        return {}


def save_state(state: dict[str, IdleGPUState]) -> None:
    """Save idle state to file."""
    try:
        data = {
            node_id: {
                "first_idle_at": s.first_idle_at,
                "last_check_at": s.last_check_at,
                "gpu_name": s.gpu_name,
                "gpu_percent": s.gpu_percent,
                "cpu_percent": s.cpu_percent,
                "external_work": s.external_work,
            }
            for node_id, s in state.items()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save state: {e}")


def check_cluster() -> list[dict]:
    """Check cluster for idle GPU-heavy nodes."""
    status = http_get(f"http://localhost:{P2P_PORT}/status")
    if not status:
        logger.error("Cannot reach local P2P orchestrator")
        return []

    idle_nodes = []
    # Check self
    self_info = status.get("self", {})
    if self_info.get("has_gpu") and is_gpu_heavy(self_info.get("gpu_name", "")):
        gpu_pct = float(self_info.get("gpu_percent", 0) or 0)
        if gpu_pct < GPU_IDLE_THRESHOLD:
            idle_nodes.append({
                "node_id": self_info.get("node_id", "self"),
                "gpu_name": self_info.get("gpu_name", ""),
                "gpu_percent": gpu_pct,
                "cpu_percent": float(self_info.get("cpu_percent", 0) or 0),
                "training_jobs": int(self_info.get("training_jobs", 0) or 0),
                "selfplay_jobs": int(self_info.get("selfplay_jobs", 0) or 0),
                "cmaes_running": self_info.get("cmaes_running", False),
                "gauntlet_running": self_info.get("gauntlet_running", False),
                "tournament_running": self_info.get("tournament_running", False),
            })

    # Check peers
    for node_id, peer in status.get("peers", {}).items():
        if not peer.get("has_gpu") or peer.get("retired"):
            continue

        if not is_gpu_heavy(peer.get("gpu_name", "")):
            continue

        gpu_pct = float(peer.get("gpu_percent", 0) or 0)
        if gpu_pct < GPU_IDLE_THRESHOLD:
            idle_nodes.append({
                "node_id": node_id,
                "gpu_name": peer.get("gpu_name", ""),
                "gpu_percent": gpu_pct,
                "cpu_percent": float(peer.get("cpu_percent", 0) or 0),
                "training_jobs": int(peer.get("training_jobs", 0) or 0),
                "selfplay_jobs": int(peer.get("selfplay_jobs", 0) or 0),
                "cmaes_running": peer.get("cmaes_running", False),
                "gauntlet_running": peer.get("gauntlet_running", False),
                "tournament_running": peer.get("tournament_running", False),
            })

    return idle_nodes


def main():
    parser = argparse.ArgumentParser(description="Idle GPU Node Alert")
    parser.add_argument("--threshold-minutes", type=int, default=IDLE_DURATION_MINUTES,
                        help="Minutes before alerting (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Don't send alerts")
    parser.add_argument("--test", action="store_true", help="Send test alert")
    args = parser.parse_args()

    webhook_url = get_slack_webhook_url()
    if not webhook_url:
        logger.warning("No Slack webhook configured. Set RINGRIFT_SLACK_WEBHOOK or run setup_slack_alerts.sh")
        if not args.dry_run:
            return

    # Test mode
    if args.test:
        logger.info("Sending test alert...")
        success = send_slack_alert(
            webhook_url,
            "Test Alert: Idle GPU Monitor Active",
            "This is a test alert from the idle GPU monitoring system.",
            severity="info",
            details={"Status": "OK", "Threshold": f"{args.threshold_minutes} min"}
        )
        logger.info(f"Test alert {'sent' if success else 'failed'}")
        return

    # Load previous state
    state = load_state()
    now = time.time()

    # Check cluster
    idle_nodes = check_cluster()
    logger.info(f"Found {len(idle_nodes)} idle GPU-heavy node(s)")

    # Update state and check for alerts
    alerts_to_send = []
    new_state = {}

    for node in idle_nodes:
        node_id = node["node_id"]

        if node_id in state:
            # Already tracked - check if we should alert
            s = state[node_id]
            idle_minutes = (now - s.first_idle_at) / 60

            if idle_minutes >= args.threshold_minutes and now - s.last_check_at > 7200:  # Check if we should alert (2 hours)
                alerts_to_send.append({
                    "node_id": node_id,
                    "idle_minutes": idle_minutes,
                    **node
                })

            # Update state
            new_state[node_id] = IdleGPUState(
                node_id=node_id,
                first_idle_at=s.first_idle_at,
                last_check_at=now,
                gpu_name=node["gpu_name"],
                gpu_percent=node["gpu_percent"],
                cpu_percent=node["cpu_percent"],
                external_work={
                    "cmaes": node.get("cmaes_running", False),
                    "gauntlet": node.get("gauntlet_running", False),
                    "tournament": node.get("tournament_running", False),
                }
            )
        else:
            # New idle node - start tracking
            new_state[node_id] = IdleGPUState(
                node_id=node_id,
                first_idle_at=now,
                last_check_at=now,
                gpu_name=node["gpu_name"],
                gpu_percent=node["gpu_percent"],
                cpu_percent=node["cpu_percent"],
                external_work={
                    "cmaes": node.get("cmaes_running", False),
                    "gauntlet": node.get("gauntlet_running", False),
                    "tournament": node.get("tournament_running", False),
                }
            )
            logger.info(f"Started tracking idle node: {node_id}")

    # Save updated state
    save_state(new_state)

    # Send alerts
    if alerts_to_send:
        logger.info(f"Sending alerts for {len(alerts_to_send)} node(s)")

        # Group message for multiple nodes
        if len(alerts_to_send) == 1:
            node = alerts_to_send[0]
            title = f"GPU Idle: {node['node_id']}"
            message = f"GPU has been idle for {node['idle_minutes']:.0f} minutes"
            details = {
                "GPU": node["gpu_name"],
                "GPU Util": f"{node['gpu_percent']:.0f}%",
                "CPU Util": f"{node['cpu_percent']:.0f}%",
                "Training Jobs": node["training_jobs"],
            }
            # Add external work info
            if node.get("cmaes_running"):
                details["CMA-ES"] = "Running (CPU)"
            if node.get("gauntlet_running"):
                details["Gauntlet"] = "Running (CPU)"
        else:
            title = f"GPU Idle: {len(alerts_to_send)} nodes"
            node_list = ", ".join(n["node_id"] for n in alerts_to_send[:5])
            if len(alerts_to_send) > 5:
                node_list += f" (+{len(alerts_to_send) - 5} more)"
            message = f"GPU-heavy nodes have been idle for 30+ minutes: {node_list}"
            details = {
                "Idle Nodes": len(alerts_to_send),
                "Longest Idle": f"{max(n['idle_minutes'] for n in alerts_to_send):.0f} min",
            }

        if args.dry_run:
            logger.info(f"[DRY RUN] Would send alert: {title}")
            logger.info(f"[DRY RUN] Message: {message}")
        else:
            success = send_slack_alert(webhook_url, title, message, severity="warning", details=details)
            if success:
                logger.info("Alert sent successfully")
            else:
                logger.error("Failed to send alert")
    else:
        logger.info("No alerts needed")


if __name__ == "__main__":
    main()

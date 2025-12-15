#!/usr/bin/env python3
"""Cluster Health Monitor - Alerts when nodes go offline or cluster issues detected.

Monitors P2P orchestrator health endpoints and sends alerts via:
- Console output (for cron/systemd)
- Webhook (Discord, Slack, etc.)
- Optional: email

Usage:
    # One-shot check (for cron):
    python scripts/cluster_health_monitor.py --once

    # Continuous monitoring:
    python scripts/cluster_health_monitor.py --interval 60

    # With webhook:
    python scripts/cluster_health_monitor.py --webhook https://discord.com/api/webhooks/...
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Known cluster nodes with their health endpoints
CLUSTER_NODES = {
    "lambda-gh200-i": "http://192.222.50.112:8770/health",
    "lambda-gh200-k": "http://192.222.51.150:8770/health",
    "lambda-gh200-l": "http://192.222.51.233:8770/health",
    "lambda-2xh100": "http://192.222.53.22:8770/health",
}

# Tailscale fallback addresses
TAILSCALE_FALLBACK = {
    "lambda-gh200-i": "http://100.99.27.56:8770/health",
    "lambda-gh200-k": "http://100.96.142.42:8770/health",
    "lambda-gh200-l": "http://100.76.145.60:8770/health",
    "lambda-2xh100": "http://100.78.101.123:8770/health",
}

# Alert thresholds
THRESHOLDS = {
    "leader_missing_seconds": 300,  # Alert if no leader seen for 5 minutes
    "min_active_peers": 2,          # Alert if fewer than 2 active peers
    "disk_percent_warning": 85,     # Alert if disk > 85%
    "disk_percent_critical": 95,    # Critical if disk > 95%
    "memory_percent_warning": 90,   # Alert if memory > 90%
}

# Track last alert times to avoid spam
last_alert_times: Dict[str, float] = {}
ALERT_COOLDOWN = 300  # 5 minutes between repeated alerts


def check_node_health(node_id: str, url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check health of a single node."""
    try:
        req = Request(url, headers={"User-Agent": "ClusterHealthMonitor/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return {
                "status": "healthy" if data.get("healthy") else "unhealthy",
                "data": data,
                "error": None,
            }
    except HTTPError as e:
        return {"status": "error", "data": None, "error": f"HTTP {e.code}"}
    except URLError as e:
        return {"status": "unreachable", "data": None, "error": str(e.reason)}
    except Exception as e:
        return {"status": "error", "data": None, "error": str(e)}


def check_node_with_fallback(node_id: str) -> Dict[str, Any]:
    """Check node health with Tailscale fallback."""
    primary_url = CLUSTER_NODES.get(node_id)
    if primary_url:
        result = check_node_health(node_id, primary_url)
        if result["status"] in ("healthy", "unhealthy"):
            return result

    # Try Tailscale fallback
    fallback_url = TAILSCALE_FALLBACK.get(node_id)
    if fallback_url:
        result = check_node_health(node_id, fallback_url)
        if result["status"] in ("healthy", "unhealthy"):
            result["via_tailscale"] = True
            return result

    return result


def should_alert(alert_key: str) -> bool:
    """Check if we should send an alert (respects cooldown)."""
    last_time = last_alert_times.get(alert_key, 0)
    if time.time() - last_time > ALERT_COOLDOWN:
        last_alert_times[alert_key] = time.time()
        return True
    return False


def send_webhook(webhook_url: str, message: str, level: str = "warning"):
    """Send alert to webhook (Discord/Slack compatible)."""
    if not webhook_url:
        return

    color = {"critical": 0xFF0000, "warning": 0xFFA500, "info": 0x00FF00}.get(level, 0x808080)

    payload = {
        "embeds": [{
            "title": f"RingRift Cluster Alert ({level.upper()})",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }

    try:
        req = Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=10):
            pass
    except Exception as e:
        print(f"[ALERT] Failed to send webhook: {e}", file=sys.stderr)


def check_cluster(webhook_url: Optional[str] = None, verbose: bool = False) -> List[str]:
    """Check all cluster nodes and return list of alerts."""
    alerts = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check each node
    node_results = {}
    for node_id in CLUSTER_NODES:
        result = check_node_with_fallback(node_id)
        node_results[node_id] = result

        if verbose:
            via = " (via Tailscale)" if result.get("via_tailscale") else ""
            print(f"[{ts}] {node_id}: {result['status']}{via}")

        # Node unreachable alert
        if result["status"] == "unreachable":
            alert_key = f"unreachable_{node_id}"
            if should_alert(alert_key):
                msg = f"Node {node_id} is unreachable: {result['error']}"
                alerts.append(msg)
                send_webhook(webhook_url, msg, "critical")

        # Node unhealthy alert
        elif result["status"] == "unhealthy":
            alert_key = f"unhealthy_{node_id}"
            if should_alert(alert_key):
                msg = f"Node {node_id} reports unhealthy status"
                alerts.append(msg)
                send_webhook(webhook_url, msg, "warning")

        # Check detailed health data
        elif result["data"]:
            data = result["data"]

            # Disk warning
            disk = data.get("disk_percent", 0)
            if disk >= THRESHOLDS["disk_percent_critical"]:
                alert_key = f"disk_critical_{node_id}"
                if should_alert(alert_key):
                    msg = f"CRITICAL: {node_id} disk at {disk:.1f}%"
                    alerts.append(msg)
                    send_webhook(webhook_url, msg, "critical")
            elif disk >= THRESHOLDS["disk_percent_warning"]:
                alert_key = f"disk_warning_{node_id}"
                if should_alert(alert_key):
                    msg = f"Warning: {node_id} disk at {disk:.1f}%"
                    alerts.append(msg)
                    send_webhook(webhook_url, msg, "warning")

            # Memory warning
            mem = data.get("memory_percent", 0)
            if mem >= THRESHOLDS["memory_percent_warning"]:
                alert_key = f"memory_{node_id}"
                if should_alert(alert_key):
                    msg = f"Warning: {node_id} memory at {mem:.1f}%"
                    alerts.append(msg)
                    send_webhook(webhook_url, msg, "warning")

            # Leader missing (check on one node only)
            if node_id == list(CLUSTER_NODES.keys())[0]:
                leader_last_seen = data.get("leader_last_seen_seconds")
                if leader_last_seen and leader_last_seen > THRESHOLDS["leader_missing_seconds"]:
                    alert_key = "leader_missing"
                    if should_alert(alert_key):
                        msg = f"Warning: No leader seen for {leader_last_seen:.0f}s (cluster may be leaderless)"
                        alerts.append(msg)
                        send_webhook(webhook_url, msg, "warning")

                # Low peer count
                active_peers = data.get("active_peers", 0)
                if active_peers < THRESHOLDS["min_active_peers"]:
                    alert_key = "low_peers"
                    if should_alert(alert_key):
                        msg = f"Warning: Only {active_peers} active peers in cluster"
                        alerts.append(msg)
                        send_webhook(webhook_url, msg, "warning")

    # Summary
    healthy_count = sum(1 for r in node_results.values() if r["status"] == "healthy")
    total_count = len(node_results)

    if verbose:
        print(f"[{ts}] Cluster status: {healthy_count}/{total_count} nodes healthy")

    if healthy_count == 0:
        alert_key = "cluster_down"
        if should_alert(alert_key):
            msg = "CRITICAL: All cluster nodes are unreachable!"
            alerts.append(msg)
            send_webhook(webhook_url, msg, "critical")

    return alerts


def main():
    parser = argparse.ArgumentParser(description="Monitor RingRift cluster health")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--webhook", help="Webhook URL for alerts (Discord/Slack)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"RingRift Cluster Health Monitor")
    print(f"Monitoring {len(CLUSTER_NODES)} nodes")
    print(f"Webhook: {'configured' if args.webhook else 'not configured'}")
    print("-" * 40)

    if args.once:
        alerts = check_cluster(args.webhook, verbose=True)
        if alerts:
            print(f"\nAlerts ({len(alerts)}):")
            for alert in alerts:
                print(f"  - {alert}")
            sys.exit(1)
        else:
            print("\nNo alerts - cluster healthy")
            sys.exit(0)
    else:
        print(f"Checking every {args.interval}s (Ctrl+C to stop)")
        while True:
            try:
                check_cluster(args.webhook, verbose=args.verbose)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopping monitor...")
                break


if __name__ == "__main__":
    main()

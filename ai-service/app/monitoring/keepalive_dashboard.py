"""Keepalive Dashboard & REST API for RingRift Cluster.

Provides real-time monitoring of cluster node health and keepalive status:
- Node online/offline status from P2P orchestrator
- Escalation tier per node (soft restart → service restart → tailscale → human)
- Predictive health warnings (disk, memory, GPU)
- Historical uptime tracking

Usage:
    # Start standalone server
    python -m app.monitoring.keepalive_dashboard --port 8771

    # Query API
    curl http://localhost:8771/api/cluster/keepalive

    # Or use programmatically
    from app.monitoring.keepalive_dashboard import (
        KeepaliveDashboard,
        get_cluster_keepalive_status,
    )

    status = await get_cluster_keepalive_status()
    print(f"Online nodes: {status['summary']['online_count']}")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

from app.utils.datetime_utils import iso_now, time_ago

logger = logging.getLogger(__name__)

# Configuration - use centralized port config
from app.config.ports import get_p2p_status_url

P2P_STATUS_URL = os.environ.get("P2P_STATUS_URL") or get_p2p_status_url()
ESCALATION_STATE_FILE = Path("/tmp/ringrift_escalation_state.json")
KEEPALIVE_DASHBOARD_PORT = 8771

# Node type detection patterns
NODE_TYPE_PATTERNS = {
    "lambda": ["lambda-", "gh200-", "h100"],
    "vast": ["vast-"],
    "hetzner": ["hetzner-", "cpu"],
    "mac": ["mac-", "macbook"],
}


@dataclass
class NodeKeepaliveStatus:
    """Keepalive status for a single node."""
    node_id: str
    node_type: str
    online: bool
    last_seen: float
    escalation_tier: int = 0
    escalation_attempts: int = 0
    health_warnings: list[str] = field(default_factory=list)
    uptime_percent: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "online": self.online,
            "last_seen": self.last_seen,
            "last_seen_ago": time_ago(self.last_seen) if self.last_seen > 0 else "never",
            "escalation_tier": self.escalation_tier,
            "escalation_tier_name": self._tier_name(),
            "escalation_attempts": self.escalation_attempts,
            "health_warnings": self.health_warnings,
            "uptime_percent": round(self.uptime_percent, 1),
        }

    def _tier_name(self) -> str:
        tier_names = {
            0: "none",
            1: "soft_restart",
            2: "service_restart",
            3: "tailscale_reset",
            4: "human_escalation",
        }
        return tier_names.get(self.escalation_tier, "unknown")


@dataclass
class ClusterKeepaliveStatus:
    """Aggregate keepalive status for the cluster."""
    nodes: list[NodeKeepaliveStatus]
    summary: dict[str, Any]
    timestamp: float
    p2p_leader: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "summary": self.summary,
            "p2p_leader": self.p2p_leader,
            "timestamp": self.timestamp,
            "timestamp_iso": iso_now(),
        }


def detect_node_type(node_id: str) -> str:
    """Detect node type from node ID."""
    node_lower = node_id.lower()
    for node_type, patterns in NODE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in node_lower:
                return node_type
    return "linux"


async def fetch_p2p_status() -> dict[str, Any]:
    """Fetch current P2P orchestrator status."""
    try:
        # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()

        def fetch():
            req = Request(P2P_STATUS_URL, headers={"Accept": "application/json"})
            with urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())

        return await loop.run_in_executor(None, fetch)
    except Exception as e:
        logger.warning(f"Failed to fetch P2P status: {e}")
        return {}


def load_escalation_state() -> dict[str, Any]:
    """Load escalation state from all nodes (local only for now)."""
    # Local escalation state
    if ESCALATION_STATE_FILE.exists():
        try:
            with open(ESCALATION_STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load escalation state: {e}")
    return {}


async def get_cluster_keepalive_status() -> ClusterKeepaliveStatus:
    """Get comprehensive cluster keepalive status.

    Combines:
    - P2P orchestrator status (online/offline)
    - Escalation manager state (tier, attempts)
    - Health warnings (predictive alerts)

    Returns:
        ClusterKeepaliveStatus with all nodes and summary
    """
    now = time.time()
    nodes: list[NodeKeepaliveStatus] = []

    # Get P2P status
    p2p_status = await fetch_p2p_status()
    p2p_peers = p2p_status.get("peers", {})
    alive_peers = set(p2p_status.get("alive_peers", []))
    leader_id = p2p_status.get("leader_id")

    # Get escalation state
    escalation_state = load_escalation_state()

    # Build node status list
    all_node_ids = set(p2p_peers.keys()) | set(escalation_state.keys())

    for node_id in sorted(all_node_ids):
        peer_info = p2p_peers.get(node_id, {})
        esc_info = escalation_state.get(node_id, {})

        # Determine online status
        is_online = node_id in alive_peers
        last_seen = peer_info.get("last_seen", 0)

        # Get escalation info
        escalation_tier = esc_info.get("current_tier", 0)
        tier_attempts = esc_info.get("tier_attempts", {})
        total_attempts = sum(tier_attempts.values()) if tier_attempts else 0

        # Collect health warnings
        warnings = []
        health_issues = esc_info.get("health_issues", {})
        if health_issues.get("disk_warning"):
            warnings.append(f"Disk usage high: {health_issues.get('disk_percent', 'N/A')}%")
        if health_issues.get("memory_warning"):
            warnings.append(f"Memory usage high: {health_issues.get('memory_percent', 'N/A')}%")
        if health_issues.get("gpu_ecc_errors"):
            warnings.append("GPU ECC errors detected")

        # Calculate uptime (simplified - based on escalation history)
        uptime = 100.0
        if escalation_tier > 0:
            # Reduce uptime score based on escalation tier
            uptime = max(0, 100 - (escalation_tier * 20))

        node_status = NodeKeepaliveStatus(
            node_id=node_id,
            node_type=detect_node_type(node_id),
            online=is_online,
            last_seen=last_seen,
            escalation_tier=escalation_tier,
            escalation_attempts=total_attempts,
            health_warnings=warnings,
            uptime_percent=uptime,
        )
        nodes.append(node_status)

    # Build summary
    online_count = sum(1 for n in nodes if n.online)
    offline_count = len(nodes) - online_count
    escalating_count = sum(1 for n in nodes if n.escalation_tier > 0)
    warning_count = sum(1 for n in nodes if n.health_warnings)

    # Group by type
    by_type: dict[str, dict[str, int]] = {}
    for node in nodes:
        if node.node_type not in by_type:
            by_type[node.node_type] = {"online": 0, "offline": 0, "total": 0}
        by_type[node.node_type]["total"] += 1
        if node.online:
            by_type[node.node_type]["online"] += 1
        else:
            by_type[node.node_type]["offline"] += 1

    summary = {
        "total_nodes": len(nodes),
        "online_count": online_count,
        "offline_count": offline_count,
        "online_percent": round(100 * online_count / len(nodes), 1) if nodes else 0,
        "escalating_count": escalating_count,
        "warning_count": warning_count,
        "by_type": by_type,
    }

    return ClusterKeepaliveStatus(
        nodes=nodes,
        summary=summary,
        timestamp=now,
        p2p_leader=leader_id,
    )


class KeepaliveDashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for keepalive dashboard."""

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/api/cluster/keepalive":
            self._handle_keepalive_status()
        elif self.path == "/api/cluster/keepalive/summary":
            self._handle_keepalive_summary()
        elif self.path == "/health":
            self._handle_health()
        elif self.path == "/":
            self._handle_dashboard()
        else:
            self._send_error(404, "Not found")

    def _handle_keepalive_status(self):
        """Return full keepalive status."""
        loop = asyncio.new_event_loop()
        try:
            status = loop.run_until_complete(get_cluster_keepalive_status())
            self._send_json(status.to_dict())
        except Exception as e:
            self._send_error(500, str(e))
        finally:
            loop.close()

    def _handle_keepalive_summary(self):
        """Return just the summary."""
        loop = asyncio.new_event_loop()
        try:
            status = loop.run_until_complete(get_cluster_keepalive_status())
            self._send_json({
                "summary": status.summary,
                "p2p_leader": status.p2p_leader,
                "timestamp": status.timestamp,
            })
        except Exception as e:
            self._send_error(500, str(e))
        finally:
            loop.close()

    def _handle_health(self):
        """Health check endpoint."""
        self._send_json({"status": "ok", "service": "keepalive-dashboard"})

    def _handle_dashboard(self):
        """Serve simple HTML dashboard."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>RingRift Keepalive Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; }
        .summary { display: flex; gap: 20px; margin-bottom: 20px; }
        .card { background: #16213e; padding: 15px; border-radius: 8px; min-width: 150px; }
        .card h3 { margin: 0 0 10px 0; color: #00d4ff; font-size: 14px; }
        .card .value { font-size: 36px; font-weight: bold; }
        .online { color: #00ff88; }
        .offline { color: #ff4444; }
        .warning { color: #ffaa00; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        th { background: #16213e; color: #00d4ff; }
        tr:hover { background: #1f2b4e; }
        .status-online { color: #00ff88; }
        .status-offline { color: #ff4444; }
        .tier-0 { color: #00ff88; }
        .tier-1 { color: #ffaa00; }
        .tier-2 { color: #ff6600; }
        .tier-3 { color: #ff4444; }
        .tier-4 { color: #ff0000; font-weight: bold; }
        .refresh { color: #888; font-size: 12px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🔄 RingRift Keepalive Dashboard</h1>
    <div id="summary" class="summary">Loading...</div>
    <div id="leader" style="margin-bottom: 20px;"></div>
    <table>
        <thead>
            <tr>
                <th>Node</th>
                <th>Type</th>
                <th>Status</th>
                <th>Last Seen</th>
                <th>Escalation</th>
                <th>Uptime</th>
                <th>Warnings</th>
            </tr>
        </thead>
        <tbody id="nodes"></tbody>
    </table>
    <div class="refresh" id="refresh"></div>

    <script>
        async function refresh() {
            try {
                const res = await fetch('/api/cluster/keepalive');
                const data = await res.json();

                // Update summary
                const s = data.summary;
                document.getElementById('summary').innerHTML = `
                    <div class="card"><h3>Total Nodes</h3><div class="value">${s.total_nodes}</div></div>
                    <div class="card"><h3>Online</h3><div class="value online">${s.online_count}</div></div>
                    <div class="card"><h3>Offline</h3><div class="value offline">${s.offline_count}</div></div>
                    <div class="card"><h3>Escalating</h3><div class="value warning">${s.escalating_count}</div></div>
                    <div class="card"><h3>Uptime</h3><div class="value">${s.online_percent}%</div></div>
                `;

                // Update leader
                document.getElementById('leader').innerHTML = `<strong>P2P Leader:</strong> ${data.p2p_leader || 'Unknown'}`;

                // Update nodes table
                const tbody = document.getElementById('nodes');
                tbody.innerHTML = data.nodes.map(n => `
                    <tr>
                        <td>${n.node_id}</td>
                        <td>${n.node_type}</td>
                        <td class="status-${n.online ? 'online' : 'offline'}">${n.online ? '● Online' : '○ Offline'}</td>
                        <td>${n.last_seen_ago}</td>
                        <td class="tier-${n.escalation_tier}">${n.escalation_tier_name} (${n.escalation_attempts} attempts)</td>
                        <td>${n.uptime_percent}%</td>
                        <td class="${n.health_warnings.length > 0 ? 'warning' : ''}">${n.health_warnings.join(', ') || '-'}</td>
                    </tr>
                `).join('');

                document.getElementById('refresh').textContent = `Last updated: ${new Date().toLocaleTimeString()} (auto-refresh every 10s)`;
            } catch (err) {
                document.getElementById('summary').innerHTML = `<div class="card"><h3>Error</h3><div class="value offline">${err.message}</div></div>`;
            }
        }

        refresh();
        setInterval(refresh, 10000);
    </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_json(self, data: dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())


class KeepaliveDashboard:
    """Keepalive dashboard server."""

    def __init__(self, port: int = KEEPALIVE_DASHBOARD_PORT):
        self.port = port
        self._server: Optional[HTTPServer] = None

    def run(self):
        """Run the dashboard server (blocking)."""
        self._server = HTTPServer(("0.0.0.0", self.port), KeepaliveDashboardHandler)
        logger.info(f"Keepalive dashboard running on http://localhost:{self.port}")
        logger.info(f"API endpoint: http://localhost:{self.port}/api/cluster/keepalive")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down keepalive dashboard")
            self._server.shutdown()

    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.shutdown()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RingRift Keepalive Dashboard")
    parser.add_argument("--port", type=int, default=KEEPALIVE_DASHBOARD_PORT,
                        help=f"Port to run on (default: {KEEPALIVE_DASHBOARD_PORT})")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON status and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    if args.json:
        # One-shot status output
        loop = asyncio.new_event_loop()
        status = loop.run_until_complete(get_cluster_keepalive_status())
        loop.close()
        print(json.dumps(status.to_dict(), indent=2))
    else:
        # Run server
        dashboard = KeepaliveDashboard(port=args.port)
        dashboard.run()


if __name__ == "__main__":
    main()

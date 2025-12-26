#!/usr/bin/env python3
"""Register this node with the P2P cluster.

Run this on startup to announce this node's current IP address to the cluster.
Useful for Vast.ai instances that get new IPs on restart.

Usage:
    # Register with a specific P2P coordinator
    python scripts/register_node.py --node-id <node-id> --coordinator http://<coordinator-ip>:8770

    # Auto-detect IP and register
    python scripts/register_node.py --node-id <node-id> --coordinator http://<coordinator-ip>:8770 --auto-ip

    # Include Vast instance ID for API-based updates
    python scripts/register_node.py --node-id <node-id> --coordinator http://<coordinator-ip>:8770 --vast-id <instance-id>

Example cron entry (register every 5 minutes):
    */5 * * * * python /path/to/ai-service/scripts/register_node.py --node-id <node-id> --coordinator http://<coordinator-ip>:8770 --auto-ip
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional


def _load_cluster_auth_token() -> str:
    token = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN") or "").strip()
    if token:
        return token
    token_file = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN_FILE") or "").strip()
    if token_file:
        try:
            return Path(token_file).read_text().strip()
        except (FileNotFoundError, OSError, PermissionError, IOError):
            return ""
    return ""


def get_public_ip() -> str | None:
    """Get this machine's public IP address."""
    # Try multiple services
    services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ifconfig.me/ip",
    ]

    for url in services:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                ip = response.read().decode().strip()
                if ip:
                    return ip
        except (ConnectionError, TimeoutError, urllib.error.URLError, OSError):
            continue

    return None


def get_tailscale_ip() -> str | None:
    """Get this machine's Tailscale IPv4 (100.x) if available."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        ip = (result.stdout or "").strip().splitlines()[0].strip()
        return ip or None
    except FileNotFoundError:
        return None
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
        return None


def get_local_ip() -> str | None:
    """Get this machine's local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except (OSError, ConnectionError, TimeoutError):
        return None


def get_ssh_port() -> int:
    """Get the SSH port this machine is listening on."""
    # Check for VAST SSH port environment variable
    vast_port = os.environ.get("SSH_PORT")
    if vast_port:
        return int(vast_port)

    # Default SSH port
    return 22


def register_with_coordinator(
    coordinator_url: str,
    node_id: str,
    host: str,
    port: int,
    vast_instance_id: str | None = None,
    tailscale_ip: str | None = None,
) -> bool:
    """Register this node with the P2P coordinator."""
    url = f"{coordinator_url.rstrip('/')}/register"

    payload = {
        "node_id": node_id,
        "host": host,
        "port": port,
    }
    if vast_instance_id:
        payload["vast_instance_id"] = vast_instance_id
    if tailscale_ip:
        payload["tailscale_ip"] = tailscale_ip

    try:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = _load_cluster_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
            if result.get("success"):
                print(f"Registered {node_id} at {host}:{port}")
                return True
            else:
                print(f"Registration failed: {result.get('error', 'Unknown error')}")
                return False

    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.read().decode()}")
        return False
    except Exception as e:
        print(f"Failed to register: {e}")
        return False


def register_with_any_coordinator(
    coordinator_urls: list[str],
    node_id: str,
    host: str,
    port: int,
    vast_instance_id: str | None = None,
    tailscale_ip: str | None = None,
) -> bool:
    for coordinator_url in coordinator_urls:
        if register_with_coordinator(
            coordinator_url=coordinator_url,
            node_id=node_id,
            host=host,
            port=port,
            vast_instance_id=vast_instance_id,
            tailscale_ip=tailscale_ip,
        ):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Register this node with the P2P cluster")
    parser.add_argument("--node-id", required=True, help="Node identifier (e.g., vast-5090-quad)")
    parser.add_argument(
        "--coordinator",
        required=True,
        help="Comma-separated seed coordinator URLs (e.g., http://host:8770,http://host2:8770)",
    )
    parser.add_argument("--host", help="IP address to register (auto-detect if not specified)")
    parser.add_argument("--port", type=int, help="SSH port to register (auto-detect if not specified)")
    parser.add_argument("--auto-ip", action="store_true", help="Auto-detect public IP")
    parser.add_argument("--local-ip", action="store_true", help="Use local IP instead of public")
    parser.add_argument("--vast-id", help="Vast.ai instance ID for API-based updates")

    args = parser.parse_args()

    tailscale_ip = get_tailscale_ip()

    # Determine IP
    if args.host:
        host = args.host
    elif args.local_ip:
        host = get_local_ip()
        if not host:
            print("Failed to detect local IP")
            sys.exit(1)
    elif args.auto_ip:
        host = get_public_ip() or tailscale_ip
        if not host:
            print("Failed to detect public IP")
            sys.exit(1)
    else:
        # Try public first, fall back to local
        host = get_public_ip() or tailscale_ip or get_local_ip()
        if not host:
            print("Failed to detect IP address")
            sys.exit(1)

    # Determine port
    port = args.port or get_ssh_port()

    coordinators = [c.strip() for c in (args.coordinator or "").split(",") if c.strip()]
    print(f"Registering {args.node_id} at {host}:{port} with {coordinators}")

    success = register_with_any_coordinator(
        coordinators,
        node_id=args.node_id,
        host=host,
        port=port,
        vast_instance_id=args.vast_id,
        tailscale_ip=tailscale_ip,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

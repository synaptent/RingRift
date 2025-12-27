#!/usr/bin/env python3
"""Vast.ai P2P Network Setup - Integrates SOCKS, aria2, and Cloudflare.

This module sets up comprehensive P2P networking on Vast instances using:
1. Tailscale userspace networking with SOCKS5 proxy (for containers without CAP_NET_ADMIN)
2. aria2 data servers for parallel multi-source model/data distribution
3. Cloudflare quick tunnels for NAT traversal (optional)

Usage:
    python scripts/vast_p2p_setup.py --setup-all        # Full setup on current host
    python scripts/vast_p2p_setup.py --deploy-to-vast   # Deploy to all Vast instances
    python scripts/vast_p2p_setup.py --check-status     # Check P2P network status

Architecture:
    Each Vast instance runs:
    - tailscaled in userspace mode with --socks5-server=localhost:1055
    - P2P orchestrator with RINGRIFT_SOCKS_PROXY=socks5://localhost:1055
    - aria2c RPC server for high-speed parallel downloads
    - (optional) cloudflared quick tunnel for NAT bypass

Integration with existing systems:
    - scripts/p2p/network.py: Uses SOCKS_PROXY env var for Tailscale mesh
    - app/distributed/aria2_transport.py: Multi-source parallel downloads
    - scripts/setup_cloudflare_tunnel.sh: Cloudflare tunnel setup
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add parent directory to path for imports
from app.core.ssh import run_vast_ssh_command as _run_vast_ssh


def run_vast_ssh_command(host: str, port: int, command: str, **kwargs) -> tuple[bool, str]:
    """Compatibility wrapper returning (success, output) tuple."""
    result = _run_vast_ssh(host, port, command, **kwargs)
    return result.success, result.stdout


# P2P Configuration
P2P_PORT = 8770
SOCKS_PORT = 1055
ARIA2_RPC_PORT = 6800
ARIA2_DATA_PORT = 8766

def _load_known_peers() -> list[str]:
    """Load bootstrap peers from config/distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"

    if not config_path.exists():
        print("[VastP2P] Warning: No config found, using empty peer list")
        return []

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        peers = []
        for _name, info in config.get("hosts", {}).items():
            # Use Tailscale IP for P2P mesh
            ip = info.get("tailscale_ip")
            if ip and info.get("status") == "ready":
                port = info.get("p2p_port", 8770)
                peers.append(f"{ip}:{port}")

        return peers[:5]  # Limit to 5 bootstrap peers
    except Exception as e:
        print(f"[VastP2P] Error loading config: {e}")
        return []

# Known peers for mesh bootstrap (loaded from config)
KNOWN_PEERS = _load_known_peers()


def run_local_command(cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run command locally.

    Uses shlex.split() to parse command string safely (no shell injection).
    """
    import shlex
    try:
        result = subprocess.run(
            shlex.split(cmd), capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def get_vast_instances() -> list[dict]:
    """Get running Vast instances from CLI."""
    try:
        vastai_paths = [
            "/Users/armand/.pyenv/versions/3.10.13/bin/vastai",
            "vastai",
            os.path.expanduser("~/.local/bin/vastai"),
        ]

        vastai_cmd = None
        for path in vastai_paths:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    vastai_cmd = path
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue

        if not vastai_cmd:
            return []

        result = subprocess.run(
            [vastai_cmd, "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode != 0:
            return []

        instances = []
        for inst in json.loads(result.stdout):
            if inst.get("actual_status") == "running":
                instances.append({
                    "id": str(inst.get("id")),
                    "host": inst.get("ssh_host"),
                    "port": inst.get("ssh_port"),
                    "name": f"vast-{inst.get('id', 'unknown')}",
                    "gpu": inst.get("gpu_name", "unknown"),
                    "num_gpus": inst.get("num_gpus", 1),
                })
        return instances
    except Exception as e:
        print(f"Error getting instances: {e}")
        return []


# =============================================================================
# SOCKS/Tailscale Setup
# =============================================================================

TAILSCALE_USERSPACE_SETUP = '''#!/bin/bash
set -e

# Install tailscale if not present
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Stop any existing tailscaled
pkill -f tailscaled || true
sleep 2

# Create state directories
mkdir -p /var/lib/tailscale /var/run/tailscale /var/log/ringrift

# Start tailscaled in userspace mode with SOCKS5 server
# This allows containers without CAP_NET_ADMIN to use Tailscale
nohup tailscaled \\
    --state=/var/lib/tailscale/tailscaled.state \\
    --socket=/var/run/tailscale/tailscaled.sock \\
    --tun=userspace-networking \\
    --socks5-server=localhost:{SOCKS_PORT} \\
    > /var/log/ringrift/tailscaled.log 2>&1 &

sleep 3

# Check if running
if pgrep -x tailscaled > /dev/null; then
    echo "tailscaled started with SOCKS5 on port {SOCKS_PORT}"
    # Try to get IP (may need auth)
    tailscale ip -4 2>/dev/null || echo "Need to authenticate: tailscale up"
else
    echo "Failed to start tailscaled"
    exit 1
fi
'''


def setup_tailscale_socks(host: str, port: int, name: str) -> tuple[str, bool, str]:
    """Setup Tailscale in userspace mode with SOCKS5 on a Vast instance."""
    script = TAILSCALE_USERSPACE_SETUP.replace("{SOCKS_PORT}", str(SOCKS_PORT))

    # Write and execute script
    success, output = run_vast_ssh_command(
        host, port,
        f"cat > /tmp/setup_tailscale.sh << 'EOFSCRIPT'\n{script}\nEOFSCRIPT\n"
        f"chmod +x /tmp/setup_tailscale.sh && /tmp/setup_tailscale.sh",
        timeout=60,
    )

    if success:
        return name, True, f"SOCKS5 on :{SOCKS_PORT}"
    return name, False, output


# =============================================================================
# aria2 Data Server Setup
# =============================================================================

ARIA2_SERVER_SETUP = '''#!/bin/bash
set -e

# Install aria2 if not present
if ! command -v aria2c &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq aria2 2>/dev/null
fi

# Stop existing aria2
pkill -f "aria2c.*--rpc" || true
sleep 1

# Create data directory
RINGRIFT_PATH=$(ls -d ~/ringrift/ai-service /root/ringrift/ai-service /root/RingRift/ai-service 2>/dev/null | head -1)
if [ -z "$RINGRIFT_PATH" ]; then
    echo "RingRift not found"
    exit 1
fi

DATA_DIR="$RINGRIFT_PATH/data"
mkdir -p "$DATA_DIR/aria2" /var/log/ringrift

# Start aria2c RPC server for parallel downloads
nohup aria2c \\
    --enable-rpc \\
    --rpc-listen-port={ARIA2_RPC_PORT} \\
    --rpc-listen-all=true \\
    --rpc-allow-origin-all \\
    --dir="$DATA_DIR" \\
    --max-concurrent-downloads=10 \\
    --split=16 \\
    --max-connection-per-server=16 \\
    --min-split-size=1M \\
    --continue=true \\
    --auto-file-renaming=false \\
    --allow-overwrite=true \\
    --check-certificate=false \\
    --daemon=true \\
    > /var/log/ringrift/aria2c.log 2>&1

# Also start simple HTTP server for data serving
cd "$DATA_DIR"
pkill -f "python3.*SimpleHTTPServer\\|python3.*http.server.*{ARIA2_DATA_PORT}" || true
nohup python3 -m http.server {ARIA2_DATA_PORT} --bind 0.0.0.0 > /var/log/ringrift/data_server.log 2>&1 &

sleep 2

# Verify
if curl -s "http://localhost:{ARIA2_RPC_PORT}/jsonrpc" -d '{"jsonrpc":"2.0","method":"aria2.getVersion","id":1}' > /dev/null 2>&1; then
    echo "aria2 RPC on :{ARIA2_RPC_PORT}, data server on :{ARIA2_DATA_PORT}"
else
    echo "aria2 RPC not responding"
    exit 1
fi
'''


def setup_aria2_server(host: str, port: int, name: str) -> tuple[str, bool, str]:
    """Setup aria2 RPC and data server on a Vast instance."""
    script = ARIA2_SERVER_SETUP.replace("{ARIA2_RPC_PORT}", str(ARIA2_RPC_PORT))
    script = script.replace("{ARIA2_DATA_PORT}", str(ARIA2_DATA_PORT))

    success, output = run_vast_ssh_command(
        host, port,
        f"cat > /tmp/setup_aria2.sh << 'EOFSCRIPT'\n{script}\nEOFSCRIPT\n"
        f"chmod +x /tmp/setup_aria2.sh && /tmp/setup_aria2.sh",
        timeout=60,
    )

    if success:
        return name, True, f"aria2 RPC:{ARIA2_RPC_PORT}, data:{ARIA2_DATA_PORT}"
    return name, False, output


# =============================================================================
# Cloudflare Tunnel Setup
# =============================================================================

CLOUDFLARE_TUNNEL_SETUP = '''#!/bin/bash
set -e

# Install cloudflared if not present
if ! command -v cloudflared &> /dev/null; then
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    dpkg -i /tmp/cloudflared.deb 2>/dev/null || apt-get install -f -y
    rm -f /tmp/cloudflared.deb
fi

# Stop existing tunnels
pkill -f "cloudflared.*tunnel" || true
sleep 2

mkdir -p /var/log/ringrift /etc/ringrift

# Start quick tunnel for P2P port
nohup cloudflared tunnel --url "http://127.0.0.1:{P2P_PORT}" > /var/log/ringrift/cloudflared.log 2>&1 &

# Wait for URL
for i in $(seq 1 20); do
    TUNNEL_URL=$(grep -o "https://[a-z0-9-]*\\.trycloudflare\\.com" /var/log/ringrift/cloudflared.log 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        echo "$TUNNEL_URL" > /etc/ringrift/tunnel_url
        echo "Tunnel: $TUNNEL_URL"
        exit 0
    fi
    sleep 1
done

echo "Tunnel setup timeout"
exit 1
'''


def setup_cloudflare_tunnel(host: str, port: int, name: str) -> tuple[str, bool, str]:
    """Setup Cloudflare quick tunnel on a Vast instance."""
    script = CLOUDFLARE_TUNNEL_SETUP.replace("{P2P_PORT}", str(P2P_PORT))

    success, output = run_vast_ssh_command(
        host, port,
        f"cat > /tmp/setup_cf.sh << 'EOFSCRIPT'\n{script}\nEOFSCRIPT\n"
        f"chmod +x /tmp/setup_cf.sh && /tmp/setup_cf.sh",
        timeout=90,
    )

    if success and "trycloudflare.com" in output:
        return name, True, output.split("Tunnel:")[-1].strip() if "Tunnel:" in output else output
    return name, False, output


# =============================================================================
# P2P Orchestrator with SOCKS
# =============================================================================

P2P_START_SCRIPT = '''#!/bin/bash
set -e

RINGRIFT_PATH=$(ls -d ~/ringrift/ai-service /root/ringrift/ai-service /root/RingRift/ai-service 2>/dev/null | head -1)
if [ -z "$RINGRIFT_PATH" ]; then
    echo "RingRift not found"
    exit 1
fi

cd "$RINGRIFT_PATH"
source venv/bin/activate 2>/dev/null || true
mkdir -p logs data/p2p_state

# Kill existing
pkill -f p2p_orchestrator || true
sleep 1

# Start P2P orchestrator with SOCKS proxy for Tailscale mesh
# Use -c to run as script to avoid __init__.py import issues
export RINGRIFT_SOCKS_PROXY="socks5://localhost:{SOCKS_PORT}"
export PYTHONPATH="$RINGRIFT_PATH"

nohup python3 -c "
import sys
sys.path.insert(0, '$RINGRIFT_PATH')
exec(open('$RINGRIFT_PATH/scripts/p2p_orchestrator.py').read())
" -- --node-id {NODE_ID} --port {P2P_PORT} --peers {PEERS} > logs/p2p_orchestrator.log 2>&1 &

sleep 3

if pgrep -f p2p_orchestrator > /dev/null; then
    echo "P2P started with SOCKS proxy"
    pgrep -f p2p_orchestrator | head -1
else
    # Try simpler direct execution
    nohup python3 "$RINGRIFT_PATH/scripts/p2p_orchestrator.py" \\
        --node-id {NODE_ID} \\
        --port {P2P_PORT} \\
        --peers {PEERS} \\
        > logs/p2p_orchestrator.log 2>&1 &
    sleep 2
    if pgrep -f p2p_orchestrator > /dev/null; then
        echo "P2P started (direct mode)"
        pgrep -f p2p_orchestrator | head -1
    else
        echo "Failed to start P2P"
        cat logs/p2p_orchestrator.log 2>/dev/null | tail -5
        exit 1
    fi
fi
'''


def start_p2p_with_socks(host: str, port: int, name: str) -> tuple[str, bool, str]:
    """Start P2P orchestrator with SOCKS proxy on a Vast instance."""
    peers_str = ",".join(KNOWN_PEERS)
    script = P2P_START_SCRIPT.replace("{SOCKS_PORT}", str(SOCKS_PORT))
    script = script.replace("{P2P_PORT}", str(P2P_PORT))
    script = script.replace("{NODE_ID}", name)
    script = script.replace("{PEERS}", peers_str)

    success, output = run_vast_ssh_command(
        host, port,
        f"cat > /tmp/start_p2p.sh << 'EOFSCRIPT'\n{script}\nEOFSCRIPT\n"
        f"chmod +x /tmp/start_p2p.sh && /tmp/start_p2p.sh",
        timeout=60,
    )

    if success and "started" in output.lower():
        return name, True, f"PID {output.split()[-1]}" if output.split() else "started"
    return name, False, output


# =============================================================================
# Status Checking
# =============================================================================

def check_instance_status(host: str, port: int, name: str) -> dict:
    """Check comprehensive status of a Vast instance."""
    status = {
        "name": name,
        "reachable": False,
        "tailscale_socks": False,
        "tailscale_ip": None,
        "aria2_rpc": False,
        "aria2_data": False,
        "p2p_running": False,
        "cloudflare_tunnel": None,
    }

    # Check reachability
    success, _ = run_vast_ssh_command(host, port, "echo ok", timeout=10)
    if not success:
        return status
    status["reachable"] = True

    # Check Tailscale SOCKS (test against first known peer if available)
    test_peer = KNOWN_PEERS[0] if KNOWN_PEERS else "localhost:8770"
    success, output = run_vast_ssh_command(
        host, port,
        f"curl -s --connect-timeout 3 --socks5 localhost:{SOCKS_PORT} http://{test_peer}/health 2>/dev/null && echo ok",
        timeout=15,
    )
    status["tailscale_socks"] = success and "ok" in output

    # Get Tailscale IP
    success, output = run_vast_ssh_command(host, port, "tailscale ip -4 2>/dev/null", timeout=10)
    if success and output.startswith("100."):
        status["tailscale_ip"] = output.strip()

    # Check aria2 RPC
    success, _ = run_vast_ssh_command(
        host, port,
        f"curl -s http://localhost:{ARIA2_RPC_PORT}/jsonrpc -d '{{\"jsonrpc\":\"2.0\",\"method\":\"aria2.getVersion\",\"id\":1}}' | grep -q version",
        timeout=10,
    )
    status["aria2_rpc"] = success

    # Check aria2 data server
    success, _ = run_vast_ssh_command(
        host, port, f"curl -s --connect-timeout 3 http://localhost:{ARIA2_DATA_PORT}/ | head -1", timeout=10
    )
    status["aria2_data"] = success

    # Check P2P
    success, output = run_vast_ssh_command(host, port, "pgrep -f p2p_orchestrator | head -1", timeout=10)
    status["p2p_running"] = success and bool(output.strip())

    # Check Cloudflare tunnel
    success, output = run_vast_ssh_command(host, port, "cat /etc/ringrift/tunnel_url 2>/dev/null", timeout=10)
    if success and "trycloudflare.com" in output:
        status["cloudflare_tunnel"] = output.strip()

    return status


# =============================================================================
# Main Commands
# =============================================================================

def setup_all_local():
    """Setup all components on local machine."""
    print("=" * 70)
    print("Setting up P2P networking components locally")
    print("=" * 70)

    # Tailscale SOCKS (local usually has proper permissions, skip userspace)
    print("\n1. Checking Tailscale...")
    success, output = run_local_command("tailscale ip -4")
    if success:
        print(f"   ✓ Tailscale IP: {output}")
    else:
        print("   ✗ Tailscale not configured")

    # aria2
    print("\n2. Checking aria2...")
    success, _ = run_local_command("which aria2c")
    if success:
        print("   ✓ aria2c available")
    else:
        print("   ✗ aria2c not installed (apt install aria2)")

    # Cloudflare
    print("\n3. Checking cloudflared...")
    success, _ = run_local_command("which cloudflared")
    if success:
        print("   ✓ cloudflared available")
    else:
        print("   ✗ cloudflared not installed")

    print("\n" + "=" * 70)


def deploy_to_all_vast(components: list[str] | None = None):
    """Deploy components to all Vast instances."""
    if components is None:
        components = ["tailscale", "aria2", "p2p"]

    print("=" * 70)
    print(f"Deploying to Vast instances: {', '.join(components)}")
    print("=" * 70)

    instances = get_vast_instances()
    if not instances:
        print("No Vast instances found")
        return

    print(f"Found {len(instances)} instances\n")

    for component in components:
        print(f"\n--- Deploying {component} ---")

        if component == "tailscale":
            setup_func = setup_tailscale_socks
        elif component == "aria2":
            setup_func = setup_aria2_server
        elif component == "cloudflare":
            setup_func = setup_cloudflare_tunnel
        elif component == "p2p":
            setup_func = start_p2p_with_socks
        else:
            print(f"Unknown component: {component}")
            continue

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(setup_func, inst["host"], inst["port"], inst["name"]): inst
                for inst in instances
            }
            for future in as_completed(futures):
                name, success, msg = future.result()
                status = "✓" if success else "✗"
                print(f"  {status} {name}: {msg}")


def check_all_status():
    """Check status of all Vast instances."""
    print("=" * 70)
    print("Vast P2P Network Status")
    print("=" * 70)

    instances = get_vast_instances()
    if not instances:
        print("No Vast instances found")
        return

    print(f"Checking {len(instances)} instances...\n")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(check_instance_status, inst["host"], inst["port"], inst["name"]): inst
            for inst in instances
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x["name"])

    # Print table
    print(f"{'Instance':<18} {'Reach':^5} {'SOCKS':^5} {'TS IP':^14} {'aria2':^5} {'P2P':^5} {'CF Tunnel'}")
    print("-" * 80)

    for r in results:
        reach = "✓" if r["reachable"] else "✗"
        socks = "✓" if r["tailscale_socks"] else "✗"
        ts_ip = r["tailscale_ip"][:14] if r["tailscale_ip"] else "-"
        aria2 = "✓" if r["aria2_rpc"] else "✗"
        p2p = "✓" if r["p2p_running"] else "✗"
        cf = r["cloudflare_tunnel"][:30] if r["cloudflare_tunnel"] else "-"

        print(f"{r['name']:<18} {reach:^5} {socks:^5} {ts_ip:<14} {aria2:^5} {p2p:^5} {cf}")

    # Summary
    total = len(results)
    reachable = sum(1 for r in results if r["reachable"])
    socks_ok = sum(1 for r in results if r["tailscale_socks"])
    p2p_ok = sum(1 for r in results if r["p2p_running"])

    print("-" * 80)
    print(f"Summary: {reachable}/{total} reachable, {socks_ok} SOCKS, {p2p_ok} P2P")


def main():
    parser = argparse.ArgumentParser(description="Vast P2P Network Setup")
    parser.add_argument("--setup-all", action="store_true", help="Setup all components locally")
    parser.add_argument("--deploy-to-vast", action="store_true", help="Deploy to all Vast instances")
    parser.add_argument("--check-status", action="store_true", help="Check network status")
    parser.add_argument("--components", nargs="+",
                        choices=["tailscale", "aria2", "cloudflare", "p2p"],
                        default=["tailscale", "aria2", "p2p"],
                        help="Components to deploy")
    args = parser.parse_args()

    if args.setup_all:
        setup_all_local()
    elif args.deploy_to_vast:
        deploy_to_all_vast(args.components)
    elif args.check_status:
        check_all_status()
    else:
        # Default to status check
        check_all_status()


if __name__ == "__main__":
    main()

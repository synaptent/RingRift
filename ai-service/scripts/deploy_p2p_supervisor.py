#!/usr/bin/env python3
"""Deploy P2P supervisor (cron-based) to cluster nodes for automatic restart on crash.

For Docker containers (Vast.ai, RunPod) that don't have systemd, this uses a
cron-based keepalive approach instead.

This script:
1. Creates a keepalive script in the ringrift directory
2. Adds cron entry to run it every minute
3. Starts P2P immediately

Usage:
    python scripts/deploy_p2p_supervisor.py                  # Deploy to all p2p_enabled nodes
    python scripts/deploy_p2p_supervisor.py --nodes vast-*   # Deploy to matching nodes
    python scripts/deploy_p2p_supervisor.py --dry-run        # Preview actions
    python scripts/deploy_p2p_supervisor.py --check          # Check P2P status
"""

import argparse
import asyncio
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

import yaml


KEEPALIVE_SCRIPT = '''#!/bin/bash
# RingRift P2P Keepalive Script
# Ensures P2P orchestrator is always running

NODE_ID="{node_id}"
RINGRIFT_PATH="{ringrift_path}"
P2P_PORT=8770
LOGFILE="$RINGRIFT_PATH/logs/p2p_keepalive.log"

# Ensure log directory exists
mkdir -p "$RINGRIFT_PATH/logs"

# Check if P2P is running
if curl -s --connect-timeout 5 http://localhost:$P2P_PORT/health > /dev/null 2>&1; then
    # Already running
    exit 0
fi

# Log restart
echo "[$(date)] P2P not running, starting..." >> "$LOGFILE"

# Kill any zombie process
pkill -9 -f "p2p_orchestrator.py" 2>/dev/null || true
sleep 1

# Find Python
if [ -f "$RINGRIFT_PATH/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/venv/bin/python"
else
    PYTHON="/usr/bin/python3"
fi

# Start P2P in background
cd "$RINGRIFT_PATH"
export PYTHONPATH="$RINGRIFT_PATH"
nohup $PYTHON scripts/p2p_orchestrator.py \\
    --node-id "$NODE_ID" \\
    --port $P2P_PORT \\
    --peers "https://p2p.ringrift.ai,http://100.78.101.123:8770,http://100.88.176.74:8770" \\
    --ringrift-path "${{RINGRIFT_PATH%/ai-service}}" \\
    >> "$RINGRIFT_PATH/logs/p2p.log" 2>&1 &

echo "[$(date)] P2P started with PID $!" >> "$LOGFILE"
'''


def load_hosts_config():
    """Load distributed_hosts.yaml configuration."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_ssh_cmd(host_config: dict) -> list[str]:
    """Build SSH command for a host."""
    ssh_host = host_config.get("ssh_host")
    ssh_port = host_config.get("ssh_port", 22)
    ssh_user = host_config.get("ssh_user", "root")
    ssh_key = host_config.get("ssh_key", "~/.ssh/id_ed25519")
    ssh_key = os.path.expanduser(ssh_key)

    return [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_port),
        "-i", ssh_key,
        f"{ssh_user}@{ssh_host}",
    ]


def get_keepalive_script(node_id: str, host_config: dict) -> str:
    """Generate keepalive script for a node."""
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    # Expand ~ for the script
    ringrift_path = ringrift_path.replace("~", "$HOME")
    return KEEPALIVE_SCRIPT.format(node_id=node_id, ringrift_path=ringrift_path)


async def deploy_to_node(
    node_id: str,
    host_config: dict,
    dry_run: bool = False,
) -> tuple[str, bool, str]:
    """Deploy keepalive supervisor to a single node."""
    ssh_cmd = build_ssh_cmd(host_config)
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    keepalive_script = get_keepalive_script(node_id, host_config)

    # Need sudo for nebius (ubuntu user)
    sudo_prefix = ""
    if host_config.get("ssh_user") == "ubuntu":
        sudo_prefix = "sudo "

    commands = f"""
set -e

# Expand home directory
RINGRIFT_PATH="{ringrift_path}"
RINGRIFT_PATH="${{RINGRIFT_PATH/#~/$HOME}}"

# Create directories
mkdir -p "$RINGRIFT_PATH/logs"
mkdir -p "$RINGRIFT_PATH/scripts"

# Write keepalive script
cat > "$RINGRIFT_PATH/scripts/p2p_keepalive.sh" << 'KEEPALIVE_EOF'
{keepalive_script}
KEEPALIVE_EOF

chmod +x "$RINGRIFT_PATH/scripts/p2p_keepalive.sh"

# Add cron entry (if not already present)
CRON_CMD="* * * * * $RINGRIFT_PATH/scripts/p2p_keepalive.sh"
(crontab -l 2>/dev/null | grep -v "p2p_keepalive" || true; echo "$CRON_CMD") | crontab -

# Run keepalive immediately to start P2P
$RINGRIFT_PATH/scripts/p2p_keepalive.sh

# Wait for P2P to start
sleep 5

# Check if running
if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
    echo "P2P_RUNNING"
else
    echo "P2P_NOT_RUNNING"
fi
"""

    if dry_run:
        return (node_id, True, f"Would deploy keepalive to {node_id}")

    try:
        result = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            commands,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=90)
        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        if "P2P_RUNNING" in stdout_text:
            return (node_id, True, "Deployed and running")
        elif "P2P_NOT_RUNNING" in stdout_text:
            # Cron installed but P2P not yet running (might take a moment)
            return (node_id, True, "Deployed, awaiting start")
        else:
            return (node_id, False, f"Exit {result.returncode}: {stderr_text[:200] or stdout_text[:200]}")
    except asyncio.TimeoutError:
        return (node_id, False, "Timeout")
    except Exception as e:
        return (node_id, False, str(e))


async def check_node_status(node_id: str, host_config: dict) -> tuple[str, str, str]:
    """Check P2P status on a node."""
    ssh_cmd = build_ssh_cmd(host_config)

    command = """
if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
    echo "active"
    curl -s http://localhost:8770/status 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('node_id','?'))" 2>/dev/null || echo "?"
else
    echo "inactive"
    echo "-"
fi
crontab -l 2>/dev/null | grep -q "p2p_keepalive" && echo "cron_ok" || echo "no_cron"
"""

    try:
        result = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(result.communicate(), timeout=20)
        lines = stdout.decode().strip().split('\n')
        status = lines[0] if lines else "unknown"
        node_reported = lines[1] if len(lines) > 1 else "?"
        has_cron = lines[2] if len(lines) > 2 else "?"
        return (node_id, status, f"cron:{has_cron}")
    except asyncio.TimeoutError:
        return (node_id, "timeout", "")
    except Exception as e:
        return (node_id, f"error: {e}", "")


async def main():
    parser = argparse.ArgumentParser(description="Deploy P2P supervisor (cron-based) to cluster nodes")
    parser.add_argument("--nodes", help="Node pattern to match (e.g., 'vast-*')")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("--check", action="store_true", help="Check P2P status on nodes")
    parser.add_argument("--skip-local", action="store_true", default=True, help="Skip local nodes (default: true)")
    args = parser.parse_args()

    config = load_hosts_config()
    hosts = config.get("hosts", {})

    # Filter to p2p_enabled nodes
    nodes_to_deploy = []
    for node_id, host_config in hosts.items():
        if not host_config.get("p2p_enabled", False):
            continue

        # Skip local nodes
        if args.skip_local:
            if node_id in ("local-mac", "mac-studio"):
                continue
            if host_config.get("ssh_host") == "localhost":
                continue

        # Pattern match if specified
        if args.nodes and not fnmatch(node_id, args.nodes):
            continue

        nodes_to_deploy.append((node_id, host_config))

    if not nodes_to_deploy:
        print("No nodes matched the criteria")
        sys.exit(1)

    print(f"Target nodes: {len(nodes_to_deploy)}")

    if args.check:
        # Check status mode
        print("\nChecking P2P status...\n")
        tasks = [check_node_status(nid, cfg) for nid, cfg in nodes_to_deploy]
        results = await asyncio.gather(*tasks)

        print(f"{'Node':<30} {'Status':<15} {'Supervisor'}")
        print("-" * 60)
        for node_id, status, supervisor in sorted(results):
            status_icon = "✓" if status == "active" else "✗" if status == "inactive" else "?"
            print(f"{node_id:<30} {status_icon} {status:<12} {supervisor}")

        active = sum(1 for _, s, _ in results if s == "active")
        print(f"\nActive: {active}/{len(results)}")
    else:
        # Deploy mode
        print("\nDeploying P2P keepalive supervisor...\n")

        # Deploy in batches of 10 for stability
        batch_size = 10
        all_results = []

        for i in range(0, len(nodes_to_deploy), batch_size):
            batch = nodes_to_deploy[i:i + batch_size]
            tasks = [deploy_to_node(nid, cfg, args.dry_run) for nid, cfg in batch]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            for node_id, success, message in results:
                icon = "✓" if success else "✗"
                print(f"  {icon} {node_id}: {message}")

        succeeded = sum(1 for _, s, _ in all_results if s)
        print(f"\nDeployed: {succeeded}/{len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())

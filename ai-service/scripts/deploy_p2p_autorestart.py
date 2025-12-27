#!/usr/bin/env python3
"""Deploy P2P with auto-restart to cluster nodes.

Uses a simple background supervisor script that monitors and restarts P2P.
Works on Docker containers without systemd or cron.

Usage:
    python scripts/deploy_p2p_autorestart.py                  # Deploy to all p2p_enabled nodes
    python scripts/deploy_p2p_autorestart.py --nodes vast-*   # Deploy to matching nodes
    python scripts/deploy_p2p_autorestart.py --check          # Check P2P status
"""

import argparse
import asyncio
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

import yaml


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


async def deploy_to_node(
    node_id: str,
    host_config: dict,
    dry_run: bool = False,
) -> tuple[str, bool, str]:
    """Deploy P2P with auto-restart to a single node."""
    ssh_cmd = build_ssh_cmd(host_config)
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")

    # Create a robust supervisor script
    commands = f'''
set -e

# Expand home directory
RINGRIFT_PATH="{ringrift_path}"
RINGRIFT_PATH="${{RINGRIFT_PATH/#\\~/\\$HOME}}"
eval RINGRIFT_PATH="$RINGRIFT_PATH"

NODE_ID="{node_id}"
P2P_PORT=8770

# Create directories
mkdir -p "$RINGRIFT_PATH/logs"

# Kill any existing P2P or supervisor
pkill -9 -f "p2p_orchestrator.py" 2>/dev/null || true
pkill -9 -f "p2p_supervisor_loop" 2>/dev/null || true
sleep 2

# Find Python
if [ -f "$RINGRIFT_PATH/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/venv/bin/python"
else
    PYTHON="/usr/bin/python3"
fi

# Create supervisor script
cat > "$RINGRIFT_PATH/scripts/p2p_supervisor_loop.sh" << 'SUPERVISOR_EOF'
#!/bin/bash
# P2P Supervisor Loop - keeps P2P running
NODE_ID="$1"
RINGRIFT_PATH="$2"
P2P_PORT="${{3:-8770}}"

PYTHON="$RINGRIFT_PATH/venv/bin/python"
if [ ! -x "$PYTHON" ]; then PYTHON="/usr/bin/python3"; fi

PARENT_PATH="${{RINGRIFT_PATH%/ai-service}}"
if [ "$PARENT_PATH" = "$RINGRIFT_PATH" ]; then
    PARENT_PATH="$RINGRIFT_PATH"
fi

cd "$RINGRIFT_PATH"
export PYTHONPATH="$RINGRIFT_PATH"

while true; do
    echo "[$(date)] Starting P2P orchestrator..."

    $PYTHON scripts/p2p_orchestrator.py \\
        --node-id "$NODE_ID" \\
        --port $P2P_PORT \\
        --peers "https://p2p.ringrift.ai,http://100.78.101.123:8770,http://100.88.176.74:8770" \\
        --ringrift-path "$PARENT_PATH" \\
        2>&1 | tee -a "$RINGRIFT_PATH/logs/p2p.log" || true

    echo "[$(date)] P2P exited, restarting in 10 seconds..."
    sleep 10
done
SUPERVISOR_EOF

chmod +x "$RINGRIFT_PATH/scripts/p2p_supervisor_loop.sh"

# Start supervisor in background with nohup
nohup "$RINGRIFT_PATH/scripts/p2p_supervisor_loop.sh" "$NODE_ID" "$RINGRIFT_PATH" "$P2P_PORT" \\
    >> "$RINGRIFT_PATH/logs/p2p_supervisor.log" 2>&1 &
SUPERVISOR_PID=$!

echo "Supervisor started with PID $SUPERVISOR_PID"

# Wait for P2P to start
sleep 8

# Check if running
if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
    echo "P2P_RUNNING"
else
    # Give it more time
    sleep 5
    if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
        echo "P2P_RUNNING"
    else
        echo "P2P_NOT_YET (supervisor running, waiting for startup)"
    fi
fi
'''

    if dry_run:
        return (node_id, True, f"Would deploy to {node_id}")

    try:
        result = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            commands,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        if "P2P_RUNNING" in stdout_text:
            return (node_id, True, "Running")
        elif "P2P_NOT_YET" in stdout_text or "Supervisor started" in stdout_text:
            return (node_id, True, "Supervisor started")
        else:
            # Even exit code 255 with successful output is OK
            if "Supervisor started with PID" in stdout_text:
                return (node_id, True, "Deployed")
            return (node_id, False, f"Exit {result.returncode}: {stderr_text[:150] or stdout_text[:150]}")
    except asyncio.TimeoutError:
        return (node_id, False, "Timeout")
    except Exception as e:
        return (node_id, False, str(e))


async def check_node_status(node_id: str, host_config: dict) -> tuple[str, str, int]:
    """Check P2P status on a node."""
    ssh_cmd = build_ssh_cmd(host_config)

    command = """
if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
    echo "active"
else
    echo "inactive"
fi
pgrep -c -f "p2p_supervisor_loop" 2>/dev/null || echo "0"
pgrep -c -f "p2p_orchestrator.py" 2>/dev/null || echo "0"
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
        supervisor_count = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else 0
        p2p_count = int(lines[2]) if len(lines) > 2 and lines[2].isdigit() else 0
        return (node_id, status, supervisor_count + p2p_count)
    except asyncio.TimeoutError:
        return (node_id, "timeout", 0)
    except Exception as e:
        return (node_id, f"error", 0)


async def main():
    parser = argparse.ArgumentParser(description="Deploy P2P with auto-restart to cluster nodes")
    parser.add_argument("--nodes", help="Node pattern to match (e.g., 'vast-*')")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("--check", action="store_true", help="Check P2P status on nodes")
    parser.add_argument("--skip-local", action="store_true", default=True, help="Skip local nodes")
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
        print("\nChecking P2P status...\n")
        tasks = [check_node_status(nid, cfg) for nid, cfg in nodes_to_deploy]
        results = await asyncio.gather(*tasks)

        print(f"{'Node':<30} {'Status':<12} {'Processes'}")
        print("-" * 55)
        for node_id, status, procs in sorted(results):
            icon = "✓" if status == "active" else "✗"
            print(f"{node_id:<30} {icon} {status:<10} {procs}")

        active = sum(1 for _, s, _ in results if s == "active")
        print(f"\nActive: {active}/{len(results)}")
    else:
        print("\nDeploying P2P with auto-restart...\n")

        # Deploy in batches
        batch_size = 8
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

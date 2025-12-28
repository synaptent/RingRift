#!/usr/bin/env python3
"""
Start P2P orchestrators on all cluster nodes using nohup (no systemd required).

Works on:
- Vast.ai (Docker, no systemd)
- RunPod (Docker, no systemd)
- Nebius/Vultr/Hetzner (systemd available but not required)

Usage:
    python scripts/start_p2p_cluster.py              # Start on all nodes
    python scripts/start_p2p_cluster.py --check      # Just check status
    python scripts/start_p2p_cluster.py --node vast  # Filter by name
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_hosts_config():
    """Load distributed_hosts.yaml configuration."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_ssh(host_name: str, host_config: dict, command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command on a host."""
    ssh_host = host_config.get("ssh_host", "")
    ssh_user = host_config.get("ssh_user", "ubuntu")
    ssh_key = os.path.expanduser(host_config.get("ssh_key", "~/.ssh/id_cluster"))
    ssh_port = host_config.get("ssh_port", 22)

    if not ssh_host:
        return False, "No SSH host"

    ssh_args = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes", "-i", ssh_key, "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}", command
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = (stdout.decode() + stderr.decode()).strip()
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)


async def check_p2p_status(host_name: str, host_config: dict) -> dict:
    """Check if P2P is running on a host."""
    result = {"host": host_name, "running": False, "leader": None, "peers": 0}

    # Check if p2p process is running
    ok, output = await run_ssh(host_name, host_config, "pgrep -f p2p_orchestrator || echo NONE")
    if ok and "NONE" not in output:
        result["running"] = True

        # Get status from health endpoint
        ok, output = await run_ssh(
            host_name, host_config,
            "curl -s http://localhost:8770/status 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"leader_id\",\"?\"), d.get(\"alive_peers\",0))' 2>/dev/null || echo '? 0'"
        )
        if ok:
            parts = output.split()
            if len(parts) >= 2:
                result["leader"] = parts[0]
                result["peers"] = int(parts[1]) if parts[1].isdigit() else 0

    return result


async def get_advertise_ip(host_name: str, host_config: dict) -> str:
    """Get the best IP to advertise for a node."""
    # Try Tailscale first
    ok, output = await run_ssh(host_name, host_config, "tailscale ip -4 2>/dev/null || echo NONE", timeout=10)
    if ok and "NONE" not in output and output.strip().startswith("100."):
        return output.strip().split()[0]

    # Fall back to the SSH host from config (what we used to reach it)
    tailscale_ip = host_config.get("tailscale_ip", "")
    if tailscale_ip:
        return tailscale_ip

    # Use SSH host as last resort
    return host_config.get("ssh_host", "")


async def start_p2p(host_name: str, host_config: dict, force_restart: bool = False) -> tuple[bool, str]:
    """Start P2P orchestrator on a host using nohup."""
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    venv_activate = host_config.get("venv_activate", f"source {ringrift_path}/venv/bin/activate")

    # First check if already running
    ok, output = await run_ssh(host_name, host_config, "pgrep -f p2p_orchestrator || echo NONE")
    was_running = ok and "NONE" not in output
    if was_running and not force_restart:
        return True, "Already running"

    # Kill any existing processes
    if was_running:
        await run_ssh(host_name, host_config, "pkill -f p2p_orchestrator 2>/dev/null || true", timeout=5)
        await asyncio.sleep(2)  # Wait a bit longer for graceful shutdown

    # Get the advertise IP for this node
    advertise_ip = await get_advertise_ip(host_name, host_config)
    if not advertise_ip:
        return False, "Could not determine advertise IP"

    # Start P2P with nohup and RINGRIFT_ADVERTISE_HOST set
    start_cmd = f"""
cd {ringrift_path} && \\
{venv_activate} && \\
mkdir -p logs && \\
export RINGRIFT_ADVERTISE_HOST={advertise_ip} && \\
setsid python scripts/p2p_orchestrator.py --node-id {host_name} > logs/p2p.log 2>&1 &
echo $!
"""
    ok, output = await run_ssh(host_name, host_config, start_cmd, timeout=30)
    if not ok:
        return False, f"Start failed: {output}"

    # Wait and verify
    await asyncio.sleep(3)
    ok, output = await run_ssh(host_name, host_config, "pgrep -f p2p_orchestrator || echo NONE")
    if ok and "NONE" not in output:
        action = "Restarted" if was_running else "Started"
        return True, f"{action} (PID {output.strip()}, advertise={advertise_ip})"

    # Get error from log
    ok, log = await run_ssh(host_name, host_config, f"tail -5 {ringrift_path}/logs/p2p.log 2>/dev/null || echo 'No log'")
    return False, f"Failed to start: {log}"


async def process_node(host_name: str, host_config: dict, check_only: bool, force_restart: bool = False) -> dict:
    """Process a single node."""
    result = {"host": host_name, "success": False, "message": ""}

    # Skip coordinator
    if host_config.get("role") == "coordinator":
        result["success"] = True
        result["message"] = "Skipped (coordinator)"
        return result

    if not host_config.get("ssh_host"):
        result["message"] = "No SSH configured"
        return result

    if check_only:
        status = await check_p2p_status(host_name, host_config)
        result["success"] = True
        if status["running"]:
            result["message"] = f"Running (leader={status['leader']}, peers={status['peers']})"
        else:
            result["message"] = "Not running"
        result["running"] = status["running"]
        return result

    # Start P2P
    ok, msg = await start_p2p(host_name, host_config, force_restart=force_restart)
    result["success"] = ok
    result["message"] = msg
    return result


async def main_async(check_only: bool, node_filter: str | None, concurrency: int, force_restart: bool = False):
    """Main async entry point."""
    config = load_hosts_config()
    hosts = config.get("hosts", {})

    if node_filter:
        hosts = {k: v for k, v in hosts.items() if node_filter.lower() in k.lower()}

    mode = "(check only)" if check_only else "(restart all)" if force_restart else "(starting P2P)"
    print(f"Processing {len(hosts)} hosts {mode}")
    print("-" * 60)

    sem = asyncio.Semaphore(concurrency)

    async def process_with_sem(name, cfg):
        async with sem:
            return await process_node(name, cfg, check_only, force_restart)

    tasks = [process_with_sem(name, cfg) for name, cfg in hosts.items()]
    results = await asyncio.gather(*tasks)

    # Print results
    running = sum(1 for r in results if r.get("running", False))
    started = sum(1 for r in results if r["success"] and "Started" in r["message"])
    restarted = sum(1 for r in results if r["success"] and "Restarted" in r.get("message", ""))
    failed = sum(1 for r in results if not r["success"])

    print("\nResults:")
    for r in sorted(results, key=lambda x: (not x.get("running", False), x["host"])):
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['host']}: {r['message']}")

    print(f"\nSummary: {running} running, {started} newly started, {restarted} restarted, {failed} failed")
    return results


def main():
    parser = argparse.ArgumentParser(description="Start P2P on cluster nodes")
    parser.add_argument("--check", action="store_true", help="Only check status")
    parser.add_argument("--restart", action="store_true", help="Force restart all nodes (fixes IP issues)")
    parser.add_argument("--node", type=str, help="Filter by node name")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent ops")
    args = parser.parse_args()

    asyncio.run(main_async(args.check, args.node, args.concurrency, args.restart))


if __name__ == "__main__":
    main()

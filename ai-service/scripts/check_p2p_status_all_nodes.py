#!/usr/bin/env python3
"""Check P2P daemon status on all cluster nodes.

Checks each P2P-enabled node in config/distributed_hosts.yaml for:
- P2P process running (pgrep for p2p_orchestrator.py)
- Port 8770 responding
- Disk usage
- NAT status and container type

Usage:
    python scripts/check_p2p_status_all_nodes.py

Output shows node status table with recommendations for deployment/restart.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import yaml


def parse_ssh_key(key_path: str) -> str:
    """Expand ssh key path."""
    if key_path.startswith("~"):
        return os.path.expanduser(key_path)
    return key_path


def has_p2p_process(pgrep_output: str) -> bool:
    for line in pgrep_output.splitlines():
        candidate = line.strip()
        if not candidate or "pgrep" in candidate:
            continue
        if "p2p_orchestrator.py" in candidate:
            return True
    return False


def check_node_p2p_sync(node_name: str, ssh_config: Dict) -> Dict:
    """
    Check P2P daemon status on a single node synchronously.
    Returns a dict with status information.
    """
    result = {
        "node": node_name,
        "p2p_running": False,
        "port_responding": False,
        "disk_usage": "N/A",
        "nat_status": "unknown",
        "container_type": "none",
        "recommendation": "investigate",
        "error": None,
        "ssh_host": ssh_config.get("ssh_host", "N/A"),
        "role": ssh_config.get("role", "unknown"),
        "gpu": ssh_config.get("gpu", "none"),
    }

    ssh_host = ssh_config.get("ssh_host")
    ssh_port = ssh_config.get("ssh_port", 22)
    ssh_user = ssh_config.get("ssh_user", "root")
    ssh_key = parse_ssh_key(ssh_config.get("ssh_key", "~/.ssh/id_cluster"))

    # Skip localhost checks via SSH
    if ssh_host == "localhost":
        try:
            # Check local P2P process
            pgrep_result = subprocess.run(
                ["pgrep", "-af", "p2p_orchestrator.py"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if pgrep_result.returncode == 0 and has_p2p_process(pgrep_result.stdout):
                result["p2p_running"] = True

            # Check local port
            curl_result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "2", "http://localhost:8770/status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if curl_result.returncode == 0 and curl_result.stdout.strip():
                result["port_responding"] = True
                result["p2p_running"] = True

            # Check disk usage
            df_result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if df_result.returncode == 0:
                lines = df_result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        result["disk_usage"] = parts[4]  # Use% column

            result["nat_status"] = "local"
            result["container_type"] = "none"

            # Determine recommendation
            if result["p2p_running"] and result["port_responding"]:
                result["recommendation"] = "OK"
            elif result["p2p_running"]:
                result["recommendation"] = "restart P2P"
            else:
                result["recommendation"] = "deploy P2P"

        except Exception as e:
            result["error"] = str(e)

        return result

    # Build SSH command
    ssh_cmd_base = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-i", ssh_key,
        "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}",
    ]

    try:
        # 1. Check if P2P daemon is running
        pgrep_cmd = ssh_cmd_base + ["pgrep -af p2p_orchestrator.py"]
        pgrep_result = subprocess.run(
            pgrep_cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if pgrep_result.returncode == 0 and has_p2p_process(pgrep_result.stdout):
            result["p2p_running"] = True

        # 2. Check P2P port status
        curl_cmd = ssh_cmd_base + ["curl -s --connect-timeout 5 http://localhost:8770/status | head -c 500"]
        curl_result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if curl_result.returncode == 0 and curl_result.stdout.strip():
            result["port_responding"] = True
            result["p2p_running"] = True

        # 3. Check disk usage
        df_cmd = ssh_cmd_base + ["df -h / | tail -1"]
        df_result = subprocess.run(
            df_cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if df_result.returncode == 0 and df_result.stdout.strip():
            parts = df_result.stdout.strip().split()
            if len(parts) >= 5:
                result["disk_usage"] = parts[4]  # Use% column

        # 4. Check NAT status (compare Tailscale IP vs public IP)
        # Simplified: just check if node is on vast.ai (which is NAT-blocked)
        if "vast" in node_name:
            result["nat_status"] = "NAT-blocked"
        elif "runpod" in node_name or "nebius" in node_name or "vultr" in node_name or "hetzner" in node_name:
            result["nat_status"] = "direct"
        else:
            result["nat_status"] = "unknown"

        # 5. Check container type (docker/podman)
        container_check_cmd = ssh_cmd_base + ["which docker podman 2>/dev/null"]
        container_result = subprocess.run(
            container_check_cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if container_result.returncode == 0:
            output = container_result.stdout.strip().lower()
            if "docker" in output:
                result["container_type"] = "docker"
            elif "podman" in output:
                result["container_type"] = "podman"

        # 6. Determine recommendation
        if result["p2p_running"] and result["port_responding"]:
            result["recommendation"] = "OK"
        elif result["p2p_running"] and not result["port_responding"]:
            result["recommendation"] = "restart P2P"
        elif not result["p2p_running"]:
            # Check if it's a good candidate for P2P
            if result["nat_status"] == "NAT-blocked":
                result["recommendation"] = "deploy P2P (NAT, limited utility)"
            elif result["container_type"] in ("docker", "podman"):
                result["recommendation"] = "deploy P2P (containerized)"
            else:
                result["recommendation"] = "deploy P2P"
        else:
            result["recommendation"] = "investigate"

    except subprocess.TimeoutExpired:
        result["error"] = "SSH timeout"
        result["recommendation"] = "investigate (timeout)"
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get all P2P-enabled nodes
    p2p_nodes = []
    for host_name, host_config in config.get("hosts", {}).items():
        if host_config.get("p2p_enabled", False):
            p2p_nodes.append((host_name, host_config))

    print(f"Checking P2P status on {len(p2p_nodes)} nodes...")
    print()

    # Check all nodes in parallel
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(check_node_p2p_sync, node_name, node_config)
            for node_name, node_config in p2p_nodes
        ]
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                print(f"Error checking node: {e}", file=sys.stderr)

    # Sort results by node name
    results.sort(key=lambda x: x["node"])

    # Print table header
    print(f"{'Node':<25} {'P2P':<8} {'Port':<8} {'Disk':<8} {'NAT':<15} {'Container':<12} {'Recommendation':<30}")
    print("=" * 140)

    # Print results
    for r in results:
        p2p_status = "running" if r["p2p_running"] else "stopped"
        port_status = "yes" if r["port_responding"] else "no"

        # Color coding (simplified for terminal)
        if r["recommendation"] == "OK":
            rec_display = r["recommendation"]
        else:
            rec_display = r["recommendation"]
            if r["error"]:
                rec_display += f" ({r['error'][:20]})"

        print(
            f"{r['node']:<25} "
            f"{p2p_status:<8} "
            f"{port_status:<8} "
            f"{r['disk_usage']:<8} "
            f"{r['nat_status']:<15} "
            f"{r['container_type']:<12} "
            f"{rec_display:<30}"
        )

    # Summary statistics
    print()
    print("=" * 140)
    total = len(results)
    running = sum(1 for r in results if r["p2p_running"])
    responding = sum(1 for r in results if r["port_responding"])
    ok = sum(1 for r in results if r["recommendation"] == "OK")
    need_deploy = sum(1 for r in results if "deploy" in r["recommendation"])
    need_restart = sum(1 for r in results if "restart" in r["recommendation"])

    print(f"Total nodes: {total}")
    print(f"P2P running: {running} ({running*100//total if total else 0}%)")
    print(f"Port responding: {responding} ({responding*100//total if total else 0}%)")
    print(f"OK: {ok}")
    print(f"Need deploy: {need_deploy}")
    print(f"Need restart: {need_restart}")

    # List nodes that need attention
    print()
    print("Nodes needing attention:")
    for r in sorted(results, key=lambda x: x["recommendation"]):
        if r["recommendation"] != "OK":
            print(f"  {r['node']}: {r['recommendation']}")


if __name__ == "__main__":
    main()

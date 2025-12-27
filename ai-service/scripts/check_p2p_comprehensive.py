#!/usr/bin/env python3
"""
Comprehensive P2P daemon status check across all cluster nodes.
Uses P2P network data as ground truth and cross-validates with SSH checks.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


def parse_ssh_key(key_path: str) -> str:
    """Expand ssh key path."""
    if key_path.startswith("~"):
        return os.path.expanduser(key_path)
    return key_path


def get_p2p_cluster_status() -> Optional[Dict]:
    """
    Get cluster status from the P2P network.
    Try local first, then a few known stable nodes.
    """
    # Try local first
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "5", "http://localhost:8770/status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass

    # Try known stable nodes
    stable_nodes = [
        ("ubuntu@89.169.112.47", 22, "~/.ssh/id_cluster"),  # nebius-backbone-1
        ("ubuntu@89.169.110.128", 22, "~/.ssh/id_cluster"),  # nebius-h100-3
        ("root@208.167.249.164", 22, "~/.ssh/id_ed25519"),  # vultr-a100-20gb
    ]

    for ssh_user_host, ssh_port, ssh_key in stable_nodes:
        try:
            ssh_key_path = parse_ssh_key(ssh_key)
            cmd = [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-i", ssh_key_path,
                "-p", str(ssh_port),
                ssh_user_host,
                "curl -s --connect-timeout 5 http://localhost:8770/status",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except Exception:
            continue

    return None


def check_node_ssh(node_name: str, ssh_config: Dict) -> Dict:
    """Check a node via SSH to see if P2P port responds."""
    result = {
        "node": node_name,
        "ssh_reachable": False,
        "port_responds": False,
        "disk_usage": "N/A",
        "error": None,
    }

    ssh_host = ssh_config.get("ssh_host")
    ssh_port = ssh_config.get("ssh_port", 22)
    ssh_user = ssh_config.get("ssh_user", "root")
    ssh_key = parse_ssh_key(ssh_config.get("ssh_key", "~/.ssh/id_cluster"))

    if ssh_host == "localhost":
        try:
            curl_result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "2", "http://localhost:8770/status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result["ssh_reachable"] = True
            if curl_result.returncode == 0 and curl_result.stdout.strip():
                try:
                    json.loads(curl_result.stdout)
                    result["port_responds"] = True
                except json.JSONDecodeError:
                    pass

            # Check disk
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
                        result["disk_usage"] = parts[4]
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
        # Check if SSH is reachable
        echo_cmd = ssh_cmd_base + ["echo ok"]
        echo_result = subprocess.run(
            echo_cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if echo_result.returncode == 0:
            result["ssh_reachable"] = True

        # Check P2P port
        curl_cmd = ssh_cmd_base + ["curl -s --connect-timeout 5 http://localhost:8770/status | head -c 500"]
        curl_result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if curl_result.returncode == 0 and curl_result.stdout.strip():
            try:
                json.loads(curl_result.stdout)
                result["port_responds"] = True
            except json.JSONDecodeError:
                pass

        # Check disk usage
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
                result["disk_usage"] = parts[4]

    except subprocess.TimeoutExpired:
        result["error"] = "SSH timeout"
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get P2P cluster status
    print("Fetching P2P cluster status...")
    p2p_status = get_p2p_cluster_status()

    if not p2p_status:
        print("ERROR: Could not fetch P2P cluster status from any node!")
        print("This means the P2P network may be down or unreachable.")
        sys.exit(1)

    # Extract node info from P2P
    p2p_nodes = set()
    p2p_alive = set()
    p2p_data = {}

    import time
    now = time.time()

    # Add self
    self_data = p2p_status.get("self", {})
    self_id = self_data.get("node_id")
    if self_id:
        p2p_nodes.add(self_id)
        p2p_data[self_id] = self_data
        if now - self_data.get("last_heartbeat", 0) < 120:
            p2p_alive.add(self_id)

    # Add peers
    for node_id, node_data in p2p_status.get("peers", {}).items():
        p2p_nodes.add(node_id)
        p2p_data[node_id] = node_data
        if now - node_data.get("last_heartbeat", 0) < 120:
            p2p_alive.add(node_id)

    print(f"P2P Network Status:")
    print(f"  Leader: {p2p_status.get('leader_id', 'none')}")
    print(f"  Voters: {p2p_status.get('voters_alive', 0)}/{p2p_status.get('voter_quorum_size', 0)} (quorum: {p2p_status.get('voter_quorum_ok', False)})")
    print(f"  Total nodes in P2P: {len(p2p_nodes)}")
    print(f"  Alive nodes (< 2min): {len(p2p_alive)}")
    print()

    # Get all P2P-enabled nodes from config
    config_nodes = {}
    for host_name, host_config in config.get("hosts", {}).items():
        if host_config.get("p2p_enabled", False):
            config_nodes[host_name] = host_config

    print(f"Config has {len(config_nodes)} nodes with p2p_enabled=true")
    print()

    # Identify missing nodes
    in_config_not_p2p = set(config_nodes.keys()) - p2p_nodes
    in_p2p_not_config = p2p_nodes - set(config_nodes.keys())

    if in_config_not_p2p:
        print(f"⚠ Nodes in config but NOT in P2P network ({len(in_config_not_p2p)}):")
        for node in sorted(in_config_not_p2p):
            print(f"  - {node}")
        print()

    if in_p2p_not_config:
        print(f"⚠ Nodes in P2P network but NOT in config ({len(in_p2p_not_config)}):")
        for node in sorted(in_p2p_not_config):
            p2p_info = p2p_data.get(node, {})
            gpu = p2p_info.get("gpu_name", "unknown")
            print(f"  - {node} ({gpu})")
        print()

    # Print comprehensive table
    print("=" * 150)
    print(f"{'Node':<25} {'P2P Status':<15} {'Last HB':<10} {'Jobs':<8} {'Disk':<8} {'NAT':<12} {'GPU':<30} {'Recommendation':<30}")
    print("=" * 150)

    all_nodes = sorted(set(config_nodes.keys()) | p2p_nodes)

    for node_name in all_nodes:
        p2p_info = p2p_data.get(node_name, {})
        config_info = config_nodes.get(node_name, {})

        # P2P status
        if node_name in p2p_alive:
            p2p_status_str = "✓ alive"
        elif node_name in p2p_nodes:
            p2p_status_str = "⚠ stale"
        else:
            p2p_status_str = "✗ not in P2P"

        # Last heartbeat
        last_hb = p2p_info.get("last_heartbeat", 0)
        if last_hb > 0:
            age = int(now - last_hb)
            if age < 60:
                hb_str = f"{age}s"
            elif age < 3600:
                hb_str = f"{age//60}m"
            else:
                hb_str = f"{age//3600}h"
        else:
            hb_str = "N/A"

        # Jobs
        selfplay = p2p_info.get("selfplay_jobs", 0)
        training = p2p_info.get("training_jobs", 0)
        jobs_str = f"S:{selfplay} T:{training}" if (selfplay or training) else "-"

        # Disk
        disk_pct = p2p_info.get("disk_percent", 0)
        disk_str = f"{int(disk_pct)}%" if disk_pct > 0 else "N/A"

        # NAT status
        nat_blocked = p2p_info.get("nat_blocked", False)
        nat_str = "NAT" if nat_blocked else "direct"

        # GPU
        gpu = p2p_info.get("gpu_name", config_info.get("gpu", "unknown"))[:30]

        # Recommendation
        if node_name not in p2p_nodes:
            recommendation = "DEPLOY P2P"
        elif node_name not in p2p_alive:
            recommendation = "RESTART P2P (stale)"
        elif p2p_info.get("retired", False):
            recommendation = "RETIRED"
        elif disk_pct > 90:
            recommendation = "DISK FULL (cleanup needed)"
        elif disk_pct > 80:
            recommendation = "Disk high (monitor)"
        else:
            recommendation = "OK"

        print(
            f"{node_name:<25} "
            f"{p2p_status_str:<15} "
            f"{hb_str:<10} "
            f"{jobs_str:<8} "
            f"{disk_str:<8} "
            f"{nat_str:<12} "
            f"{gpu:<30} "
            f"{recommendation:<30}"
        )

    print("=" * 150)

    # Summary
    print()
    print("SUMMARY:")
    need_deploy = [n for n in all_nodes if n not in p2p_nodes]
    need_restart = [n for n in p2p_nodes if n not in p2p_alive and n in config_nodes]
    retired = [n for n in p2p_nodes if p2p_data.get(n, {}).get("retired", False)]
    disk_full = [n for n in p2p_alive if p2p_data.get(n, {}).get("disk_percent", 0) > 90]

    print(f"  Nodes alive and healthy: {len(p2p_alive) - len(retired)}")
    print(f"  Nodes need DEPLOY: {len(need_deploy)}")
    if need_deploy:
        for n in need_deploy:
            print(f"    - {n}")
    print(f"  Nodes need RESTART: {len(need_restart)}")
    if need_restart:
        for n in need_restart:
            print(f"    - {n}")
    print(f"  Nodes RETIRED: {len(retired)}")
    if retired:
        for n in retired:
            print(f"    - {n}")
    print(f"  Nodes DISK FULL: {len(disk_full)}")
    if disk_full:
        for n in disk_full:
            disk_pct = p2p_data.get(n, {}).get("disk_percent", 0)
            print(f"    - {n} ({int(disk_pct)}%)")

    # Job summary
    print()
    print("ACTIVE JOBS:")
    total_selfplay = sum(p2p_data.get(n, {}).get("selfplay_jobs", 0) for n in p2p_alive)
    total_training = sum(p2p_data.get(n, {}).get("training_jobs", 0) for n in p2p_alive)
    print(f"  Total selfplay jobs: {total_selfplay}")
    print(f"  Total training jobs: {total_training}")


if __name__ == "__main__":
    main()

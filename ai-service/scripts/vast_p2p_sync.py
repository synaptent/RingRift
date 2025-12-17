#!/usr/bin/env python3
"""
Vast.ai P2P Sync - Synchronize Vast instance state with P2P network.

This script:
1. Gets active Vast instances from vastai CLI
2. Compares with P2P network retired nodes
3. Unretires nodes that match active Vast instances
4. Starts P2P orchestrator on nodes missing it
5. Updates distributed_hosts.yaml with current IPs
6. Can provision new instances based on demand
7. Can be run via cron for continuous sync

Usage:
    python scripts/vast_p2p_sync.py --check      # Check status only
    python scripts/vast_p2p_sync.py --sync       # Sync and unretire active instances
    python scripts/vast_p2p_sync.py --start-p2p  # Start P2P on instances missing it
    python scripts/vast_p2p_sync.py --full       # Full sync (check + sync + start)
    python scripts/vast_p2p_sync.py --update-config  # Update distributed_hosts.yaml
    python scripts/vast_p2p_sync.py --provision N    # Provision N new instances
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# P2P leader endpoint
P2P_LEADER = os.environ.get("P2P_LEADER", "http://100.88.176.74:8770")

# Vast instance ID to Tailscale IP mapping (discovered dynamically)
VAST_TAILSCALE_IPS: Dict[int, str] = {}


@dataclass
class VastInstance:
    """Vast.ai instance info."""
    id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    vcpus: float
    ram_gb: float
    ssh_host: str
    ssh_port: int
    status: str
    hourly_cost: float
    uptime_mins: float


@dataclass
class P2PNode:
    """P2P network node info."""
    node_id: str
    host: str
    retired: bool
    selfplay_jobs: int
    healthy: bool
    gpu_name: str


def get_vast_instances() -> List[VastInstance]:
    """Get active Vast instances from vastai CLI."""
    try:
        result = subprocess.run(
            ['vastai', 'show', 'instances', '--raw'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            logger.error(f"vastai CLI error: {result.stderr}")
            return []

        instances = json.loads(result.stdout)
        return [
            VastInstance(
                id=inst.get('id', 0),
                machine_id=inst.get('machine_id', 0),
                gpu_name=inst.get('gpu_name', 'Unknown'),
                num_gpus=inst.get('num_gpus', 0) or 1,
                vcpus=inst.get('cpu_cores_effective', 0) or 0,
                ram_gb=inst.get('cpu_ram', 0) / 1024 if inst.get('cpu_ram') else 0,
                ssh_host=inst.get('ssh_host', ''),
                ssh_port=inst.get('ssh_port', 22),
                status=inst.get('actual_status', 'unknown'),
                hourly_cost=inst.get('dph_total', 0) or 0,
                uptime_mins=inst.get('duration', 0) or 0,
            )
            for inst in instances
            if inst.get('actual_status') == 'running'
        ]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse vastai output: {e}")
        return []
    except subprocess.TimeoutExpired:
        logger.error("vastai CLI timeout")
        return []
    except FileNotFoundError:
        logger.error("vastai CLI not found - install with: pip install vastai")
        return []


def get_p2p_nodes() -> List[P2PNode]:
    """Get nodes from P2P network."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{P2P_LEADER}/status", timeout=10) as response:
            data = json.loads(response.read().decode())

        nodes = []
        peers = data.get('peers', {})
        for node_id, info in peers.items():
            nodes.append(P2PNode(
                node_id=node_id,
                host=info.get('reported_host') or info.get('host', ''),
                retired=info.get('retired', False),
                selfplay_jobs=info.get('selfplay_jobs', 0),
                healthy=not info.get('retired', False),
                gpu_name=info.get('gpu_name', ''),
            ))

        # Add self
        self_info = data.get('self', {})
        if self_info:
            nodes.append(P2PNode(
                node_id=self_info.get('node_id', ''),
                host=self_info.get('host', ''),
                retired=self_info.get('retired', False),
                selfplay_jobs=self_info.get('selfplay_jobs', 0),
                healthy=True,
                gpu_name=self_info.get('gpu_name', ''),
            ))

        return nodes
    except Exception as e:
        logger.error(f"Failed to get P2P nodes: {e}")
        return []


def get_vast_tailscale_ip(instance: VastInstance) -> Optional[str]:
    """Get Tailscale IP for a Vast instance via SSH."""
    if instance.id in VAST_TAILSCALE_IPS:
        return VAST_TAILSCALE_IPS[instance.id]

    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'tailscale ip -4 2>/dev/null || ip route get 1 | grep -oP "src \\K\\S+"'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            ip = result.stdout.strip().split('\n')[0]
            if ip.startswith('100.'):
                VAST_TAILSCALE_IPS[instance.id] = ip
                return ip
    except Exception as e:
        logger.debug(f"Failed to get Tailscale IP for instance {instance.id}: {e}")

    return None


def match_vast_to_p2p(vast_instances: List[VastInstance], p2p_nodes: List[P2PNode]) -> Dict[int, P2PNode]:
    """Match Vast instances to P2P nodes by various criteria."""
    matches: Dict[int, P2PNode] = {}

    for inst in vast_instances:
        # Try matching by Tailscale IP
        ts_ip = get_vast_tailscale_ip(inst)
        if ts_ip:
            for node in p2p_nodes:
                if node.host == ts_ip:
                    matches[inst.id] = node
                    break
            if inst.id in matches:
                continue

        # Try matching by node_id patterns
        for node in p2p_nodes:
            node_lower = node.node_id.lower()
            # Match patterns like vast-28844401, vast-4e19d4df2c83, etc.
            if f"vast-{inst.id}" in node_lower or f"vast{inst.id}" in node_lower:
                matches[inst.id] = node
                break
            # Match by machine ID
            if f"vast-{inst.machine_id}" in node_lower:
                matches[inst.id] = node
                break

    return matches


def unretire_node_via_api(node_id: str) -> bool:
    """Unretire a node by calling P2P API."""
    try:
        import urllib.request
        import urllib.parse

        url = f"{P2P_LEADER}/admin/unretire?node_id={urllib.parse.quote(node_id)}"
        req = urllib.request.Request(url, method='POST')
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        logger.warning(f"Failed to unretire {node_id} via API: {e}")
        return False


def start_p2p_on_instance(instance: VastInstance) -> bool:
    """Start P2P orchestrator on a Vast instance."""
    # Determine RingRift path
    ringrift_paths = ['/workspace/ringrift', '/root/ringrift']

    try:
        # Check which path exists
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'ls -d /workspace/ringrift 2>/dev/null || ls -d /root/ringrift 2>/dev/null'],
            capture_output=True, text=True, timeout=15
        )
        ringrift_path = result.stdout.strip() if result.returncode == 0 else '/root/ringrift'

        # Generate node ID
        node_id = f"vast-{instance.id}"

        # Start P2P
        cmd = f"""
cd {ringrift_path}/ai-service
mkdir -p logs
pkill -f p2p_orchestrator 2>/dev/null || true
nohup /opt/conda/bin/python3 scripts/p2p_orchestrator.py \\
    --node-id {node_id} \\
    --port 8770 \\
    --peers https://p2p.ringrift.ai \\
    --ringrift-path {ringrift_path} \\
    > logs/p2p.log 2>&1 &
sleep 2
curl -s http://localhost:8770/health | head -c 100
"""
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}', cmd],
            capture_output=True, text=True, timeout=30
        )

        if 'healthy' in result.stdout or 'node_id' in result.stdout:
            logger.info(f"Started P2P on instance {instance.id} ({instance.gpu_name})")
            return True
        else:
            logger.warning(f"P2P start on {instance.id} unclear: {result.stdout[:200]}")
            return False

    except Exception as e:
        logger.error(f"Failed to start P2P on instance {instance.id}: {e}")
        return False


def check_p2p_running(instance: VastInstance) -> Tuple[bool, int]:
    """Check if P2P is running on instance. Returns (is_running, selfplay_jobs)."""
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
             '-p', str(instance.ssh_port), f'root@{instance.ssh_host}',
             'curl -s http://localhost:8770/health 2>/dev/null'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            return True, data.get('selfplay_jobs', 0)
    except Exception:
        pass
    return False, 0


def main():
    parser = argparse.ArgumentParser(description="Sync Vast instances with P2P network")
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--sync', action='store_true', help='Sync and unretire active instances')
    parser.add_argument('--start-p2p', action='store_true', help='Start P2P on instances missing it')
    parser.add_argument('--full', action='store_true', help='Full sync (check + sync + start)')
    args = parser.parse_args()

    if not any([args.check, args.sync, args.start_p2p, args.full]):
        args.check = True  # Default to check

    # Get Vast instances
    logger.info("Getting Vast instances...")
    vast_instances = get_vast_instances()
    logger.info(f"Found {len(vast_instances)} active Vast instances")

    if not vast_instances:
        logger.warning("No active Vast instances found")
        return

    # Get P2P nodes
    logger.info("Getting P2P network nodes...")
    p2p_nodes = get_p2p_nodes()
    logger.info(f"Found {len(p2p_nodes)} P2P nodes")

    # Match Vast to P2P
    matches = match_vast_to_p2p(vast_instances, p2p_nodes)
    logger.info(f"Matched {len(matches)} Vast instances to P2P nodes")

    # Status report
    print("\n" + "=" * 80)
    print("VAST INSTANCE STATUS")
    print("=" * 80)
    print(f"{'ID':<10} {'GPU':<20} {'vCPUs':<8} {'P2P Node':<25} {'Status':<15}")
    print("-" * 80)

    for inst in vast_instances:
        p2p_node = matches.get(inst.id)
        if p2p_node:
            status = "RETIRED" if p2p_node.retired else f"OK ({p2p_node.selfplay_jobs} jobs)"
            node_str = p2p_node.node_id[:24]
        else:
            # Check if P2P is running directly
            running, jobs = check_p2p_running(inst)
            if running:
                status = f"P2P OK ({jobs} jobs)"
                node_str = "(not in network yet)"
            else:
                status = "NO P2P"
                node_str = "-"

        gpu_str = f"{inst.num_gpus}x{inst.gpu_name}"[:19]
        print(f"{inst.id:<10} {gpu_str:<20} {inst.vcpus:<8.0f} {node_str:<25} {status:<15}")

    print("=" * 80 + "\n")

    # Sync actions
    if args.sync or args.full:
        logger.info("Syncing retired nodes...")
        for inst_id, node in matches.items():
            if node.retired:
                logger.info(f"Unretiring {node.node_id} (Vast instance {inst_id})")
                if unretire_node_via_api(node.node_id):
                    logger.info(f"  -> Unretired successfully")
                else:
                    logger.warning(f"  -> Failed to unretire via API")

    # Start P2P actions
    if args.start_p2p or args.full:
        logger.info("Starting P2P on instances without it...")
        for inst in vast_instances:
            running, _ = check_p2p_running(inst)
            if not running:
                logger.info(f"Starting P2P on instance {inst.id} ({inst.gpu_name})")
                start_p2p_on_instance(inst)


if __name__ == "__main__":
    main()

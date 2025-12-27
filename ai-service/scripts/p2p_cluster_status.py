#!/usr/bin/env python3
"""
Comprehensive P2P cluster status checker.

Checks all nodes for:
- P2P status (running/not running)
- NAT blocking status
- Storage availability (>50GB free)
- Containerization (docker/bare-metal)
- GPU utilization
- Selfplay job counts
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class NodeHealth:
    """Health status for a single node."""

    name: str
    ssh_host: str
    ssh_port: int
    role: str
    gpu: str

    # Health checks
    reachable: bool = False
    p2p_running: bool = False
    p2p_port: int = 8770
    nat_blocked: bool = False

    # Storage
    storage_total_gb: Optional[float] = None
    storage_free_gb: Optional[float] = None
    storage_adequate: bool = False

    # Environment
    containerized: bool = False

    # GPU stats
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None

    # Selfplay
    selfplay_job_count: int = 0

    # Errors
    error_message: Optional[str] = None


class ClusterHealthChecker:
    """Check health of all cluster nodes."""

    def __init__(self, config_path: str = "config/distributed_hosts.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load distributed hosts config."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    async def check_node(self, name: str, info: dict) -> NodeHealth:
        """Check health of a single node."""
        health = NodeHealth(
            name=name,
            ssh_host=info.get('ssh_host', 'N/A'),
            ssh_port=info.get('ssh_port', 22),
            role=info.get('role', 'unknown'),
            gpu=info.get('gpu', 'none')
        )

        # Skip if not P2P enabled
        if not info.get('p2p_enabled', False):
            health.error_message = "P2P not enabled in config"
            return health

        # Skip if not ready
        if info.get('status') != 'ready':
            health.error_message = f"Status: {info.get('status', 'unknown')}"
            return health

        # Check reachability via SSH
        try:
            await self._check_reachability(health, info)
        except Exception as e:
            health.error_message = f"Reachability check failed: {e}"
            return health

        if not health.reachable:
            return health

        # Run all health checks in parallel
        await asyncio.gather(
            self._check_p2p_status(health, info),
            self._check_storage(health, info),
            self._check_containerization(health, info),
            self._check_gpu_stats(health, info),
            self._check_selfplay_jobs(health, info),
            return_exceptions=True
        )

        return health

    async def _check_reachability(self, health: NodeHealth, info: dict):
        """Check if node is SSH reachable."""
        ssh_user = info.get('ssh_user', 'root')
        ssh_key = info.get('ssh_key', '~/.ssh/id_cluster')

        # Expand tilde in ssh_key
        ssh_key = str(Path(ssh_key).expanduser())

        cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(health.ssh_port),
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{ssh_user}@{health.ssh_host}',
            'echo reachable'
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            if proc.returncode == 0 and b'reachable' in stdout:
                health.reachable = True
        except asyncio.TimeoutError:
            health.error_message = "SSH timeout"
        except Exception as e:
            health.error_message = f"SSH error: {e}"

    async def _check_p2p_status(self, health: NodeHealth, info: dict):
        """Check if P2P is running on the node."""
        if not health.reachable:
            return

        ssh_user = info.get('ssh_user', 'root')
        ssh_key = str(Path(info.get('ssh_key', '~/.ssh/id_cluster')).expanduser())

        # Try to curl the P2P status endpoint
        cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(health.ssh_port),
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{ssh_user}@{health.ssh_host}',
            f'curl -s --max-time 3 http://localhost:8770/status 2>/dev/null || echo "NOT_RUNNING"'
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

            output = stdout.decode('utf-8', errors='ignore').strip()

            if output and output != "NOT_RUNNING" and not output.startswith("curl:"):
                try:
                    status = json.loads(output)
                    health.p2p_running = True
                    # Check if NAT blocked
                    health.nat_blocked = status.get('nat_blocked', False)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            pass

    async def _check_storage(self, health: NodeHealth, info: dict):
        """Check storage availability."""
        if not health.reachable:
            return

        ssh_user = info.get('ssh_user', 'root')
        ssh_key = str(Path(info.get('ssh_key', '~/.ssh/id_cluster')).expanduser())
        ringrift_path = info.get('ringrift_path', '~/ringrift/ai-service')

        # Get df output for the ringrift directory
        cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(health.ssh_port),
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{ssh_user}@{health.ssh_host}',
            f'df -BG {ringrift_path} | tail -1'
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            output = stdout.decode('utf-8', errors='ignore').strip()
            parts = output.split()

            if len(parts) >= 4:
                total = parts[1].rstrip('G')
                available = parts[3].rstrip('G')

                health.storage_total_gb = float(total)
                health.storage_free_gb = float(available)
                health.storage_adequate = health.storage_free_gb > 50
        except Exception as e:
            pass

    async def _check_containerization(self, health: NodeHealth, info: dict):
        """Check if running in container."""
        if not health.reachable:
            return

        ssh_user = info.get('ssh_user', 'root')
        ssh_key = str(Path(info.get('ssh_key', '~/.ssh/id_cluster')).expanduser())

        # Check for /.dockerenv or cgroup
        cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(health.ssh_port),
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{ssh_user}@{health.ssh_host}',
            'test -f /.dockerenv && echo CONTAINER || grep -q docker /proc/1/cgroup 2>/dev/null && echo CONTAINER || echo BARE_METAL'
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            output = stdout.decode('utf-8', errors='ignore').strip()
            health.containerized = 'CONTAINER' in output
        except Exception as e:
            pass

    async def _check_gpu_stats(self, health: NodeHealth, info: dict):
        """Check GPU utilization and memory."""
        if not health.reachable or health.gpu == 'none':
            return

        ssh_user = info.get('ssh_user', 'root')
        ssh_key = str(Path(info.get('ssh_key', '~/.ssh/id_cluster')).expanduser())

        # Use nvidia-smi to get GPU stats
        cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(health.ssh_port),
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{ssh_user}@{health.ssh_host}',
            'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1'
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            output = stdout.decode('utf-8', errors='ignore').strip()
            parts = output.split(',')

            if len(parts) >= 3:
                health.gpu_utilization = float(parts[0].strip())
                health.gpu_memory_used_gb = float(parts[1].strip()) / 1024
                health.gpu_memory_total_gb = float(parts[2].strip()) / 1024
        except Exception as e:
            pass

    async def _check_selfplay_jobs(self, health: NodeHealth, info: dict):
        """Count running selfplay jobs."""
        if not health.reachable:
            return

        ssh_user = info.get('ssh_user', 'root')
        ssh_key = str(Path(info.get('ssh_key', '~/.ssh/id_cluster')).expanduser())

        # Count python selfplay processes
        cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(health.ssh_port),
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{ssh_user}@{health.ssh_host}',
            'ps aux | grep -E "selfplay|gpu_parallel" | grep -v grep | wc -l'
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            output = stdout.decode('utf-8', errors='ignore').strip()
            health.selfplay_job_count = int(output)
        except Exception as e:
            pass

    async def check_all_nodes(self) -> List[NodeHealth]:
        """Check health of all nodes in parallel."""
        tasks = []

        for name, info in self.config['hosts'].items():
            tasks.append(self.check_node(name, info))

        results = await asyncio.gather(*tasks)
        return results

    def print_report(self, results: List[NodeHealth]):
        """Print a comprehensive health report."""
        print("\n" + "=" * 140)
        print("P2P CLUSTER HEALTH REPORT")
        print("=" * 140)

        # Group by status
        p2p_running = []
        p2p_not_running = []
        unreachable = []
        not_enabled = []

        for health in results:
            if health.error_message and "not enabled" in health.error_message.lower():
                not_enabled.append(health)
            elif not health.reachable:
                unreachable.append(health)
            elif health.p2p_running:
                p2p_running.append(health)
            else:
                p2p_not_running.append(health)

        # Summary
        print(f"\nSUMMARY:")
        print(f"  Total nodes in config: {len(results)}")
        print(f"  P2P running: {len(p2p_running)}")
        print(f"  P2P not running (but reachable): {len(p2p_not_running)}")
        print(f"  Unreachable: {len(unreachable)}")
        print(f"  P2P not enabled: {len(not_enabled)}")

        # Detailed P2P running nodes
        if p2p_running:
            print("\n" + "-" * 140)
            print("NODES WITH P2P RUNNING:")
            print("-" * 140)
            print(f"{'Name':<25} | {'Host':<20} | {'NAT':<5} | {'Storage':<15} | {'Container':<10} | {'GPU Util':<10} | {'GPU Mem':<15} | {'Jobs':<5}")
            print("-" * 140)

            for health in sorted(p2p_running, key=lambda x: x.name):
                nat_status = "YES" if health.nat_blocked else "NO"
                storage_info = f"{health.storage_free_gb:.1f}GB free" if health.storage_free_gb else "N/A"
                container_status = "Yes" if health.containerized else "No"
                gpu_util = f"{health.gpu_utilization:.1f}%" if health.gpu_utilization is not None else "N/A"

                if health.gpu_memory_used_gb and health.gpu_memory_total_gb:
                    gpu_mem = f"{health.gpu_memory_used_gb:.1f}/{health.gpu_memory_total_gb:.1f}GB"
                else:
                    gpu_mem = "N/A"

                print(f"{health.name:<25} | {health.ssh_host:<20} | {nat_status:<5} | {storage_info:<15} | {container_status:<10} | {gpu_util:<10} | {gpu_mem:<15} | {health.selfplay_job_count:<5}")

        # Nodes without P2P
        if p2p_not_running:
            print("\n" + "-" * 140)
            print("NODES WITHOUT P2P (but reachable and P2P enabled):")
            print("-" * 140)
            print(f"{'Name':<25} | {'Host':<20} | {'Storage':<15} | {'Container':<10} | {'GPU':<40}")
            print("-" * 140)

            for health in sorted(p2p_not_running, key=lambda x: x.name):
                storage_info = f"{health.storage_free_gb:.1f}GB free" if health.storage_free_gb else "N/A"
                container_status = "Yes" if health.containerized else "No"

                print(f"{health.name:<25} | {health.ssh_host:<20} | {storage_info:<15} | {container_status:<10} | {health.gpu:<40}")

        # Unreachable nodes
        if unreachable:
            print("\n" + "-" * 140)
            print("UNREACHABLE NODES:")
            print("-" * 140)
            print(f"{'Name':<25} | {'Host':<20} | {'Error':<80}")
            print("-" * 140)

            for health in sorted(unreachable, key=lambda x: x.name):
                error = health.error_message or "Unknown error"
                print(f"{health.name:<25} | {health.ssh_host:<20} | {error:<80}")

        # Recommendations
        print("\n" + "=" * 140)
        print("RECOMMENDATIONS:")
        print("=" * 140)

        # Nodes needing P2P deployment
        needs_p2p = [h for h in p2p_not_running if h.storage_adequate]
        if needs_p2p:
            print(f"\nNodes ready for P2P deployment ({len(needs_p2p)}):")
            for health in needs_p2p:
                print(f"  - {health.name} ({health.gpu})")

        # Nodes needing P2P deployment but low storage
        needs_p2p_low_storage = [h for h in p2p_not_running if not h.storage_adequate and h.storage_free_gb]
        if needs_p2p_low_storage:
            print(f"\nNodes needing P2P but low storage ({len(needs_p2p_low_storage)}):")
            for health in needs_p2p_low_storage:
                print(f"  - {health.name}: {health.storage_free_gb:.1f}GB free (need >50GB)")

        # Nodes with low storage
        low_storage = [h for h in p2p_running if h.storage_free_gb and h.storage_free_gb < 50]
        if low_storage:
            print(f"\nP2P nodes with low storage (<50GB, {len(low_storage)}):")
            for health in sorted(low_storage, key=lambda x: x.storage_free_gb or 0):
                print(f"  - {health.name}: {health.storage_free_gb:.1f}GB free")

        # NAT-blocked nodes
        nat_blocked = [h for h in p2p_running if h.nat_blocked]
        if nat_blocked:
            print(f"\nNAT-blocked nodes ({len(nat_blocked)}):")
            for health in nat_blocked:
                print(f"  - {health.name}")

        # Idle GPUs
        idle_gpus = [h for h in p2p_running if h.gpu_utilization is not None and h.gpu_utilization < 10 and h.selfplay_job_count == 0]
        if idle_gpus:
            print(f"\nIdle GPUs available for selfplay ({len(idle_gpus)}):")
            for health in idle_gpus:
                print(f"  - {health.name} ({health.gpu})")

        # P2P deployment instructions
        if needs_p2p or needs_p2p_low_storage:
            print(f"\nTo deploy P2P on a node:")
            print(f"  cd /Users/armand/Development/RingRift/ai-service")
            print(f"  python scripts/deploy_p2p.py --node <node-name>")

        print("\n" + "=" * 140)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Check P2P cluster health')
    parser.add_argument('--config', default='config/distributed_hosts.yaml',
                       help='Path to distributed_hosts.yaml config')
    args = parser.parse_args()

    checker = ClusterHealthChecker(args.config)

    print("Checking P2P cluster health...")
    print("This may take 1-2 minutes for all nodes...")

    results = await checker.check_all_nodes()
    checker.print_report(results)


if __name__ == '__main__':
    asyncio.run(main())

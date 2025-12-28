#!/usr/bin/env python3
"""Master Cluster Automation Script for RingRift AI Training.

Provides comprehensive cluster management:
1. Sets up new nodes (venv, dependencies)
2. Starts P2P orchestrator on all nodes
3. Assigns appropriate work based on capabilities
4. Monitors and self-heals

Usage:
    # Full cluster setup
    python scripts/master_cluster_automation.py --setup-all

    # Setup specific node
    python scripts/master_cluster_automation.py --setup-node vast-28925166

    # Start P2P on all ready nodes
    python scripts/master_cluster_automation.py --start-p2p

    # Start selfplay on GPU nodes
    python scripts/master_cluster_automation.py --start-selfplay

    # Daemon mode (continuous monitoring and self-healing)
    python scripts/master_cluster_automation.py --daemon

December 2025: Created for automated cluster orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("master_automation")


@dataclass
class NodeCapabilities:
    """Detected node capabilities."""

    node_id: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "root"
    ssh_key: str = "~/.ssh/id_cluster"
    has_gpu: bool = False
    gpu_type: str = "CPU"
    gpu_memory_gb: int = 0
    cpu_cores: int = 0
    memory_gb: int = 0
    disk_free_gb: int = 0
    venv_ready: bool = False
    p2p_running: bool = False
    ringrift_path: str = "~/ringrift/ai-service"
    suitable_for: list[str] = field(default_factory=list)


class MasterClusterAutomation:
    """Master automation controller for cluster management."""

    def __init__(self, config_path: str | None = None):
        """Initialize the automation controller.

        Args:
            config_path: Path to distributed_hosts.yaml
        """
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
            )
        self.config_path = config_path
        self.hosts: dict[str, dict] = {}
        self._load_hosts()

    def _load_hosts(self) -> None:
        """Load host configuration from YAML."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            self.hosts = config.get("hosts", {})
            logger.info(f"Loaded {len(self.hosts)} hosts from config")
        except Exception as e:
            logger.error(f"Failed to load hosts config: {e}")
            self.hosts = {}

    def get_ready_hosts(self) -> list[str]:
        """Get list of hosts with status='ready'."""
        return [
            node_id
            for node_id, info in self.hosts.items()
            if info.get("status") == "ready"
        ]

    def get_gpu_hosts(self) -> list[str]:
        """Get list of hosts with GPUs."""
        return [
            node_id
            for node_id, info in self.hosts.items()
            if info.get("status") == "ready" and info.get("gpu")
        ]

    def _get_ssh_command(self, node_id: str) -> list[str]:
        """Get SSH command for a node."""
        info = self.hosts.get(node_id, {})
        ssh_host = info.get("ssh_host", "")
        ssh_port = info.get("ssh_port", 22)
        ssh_user = info.get("ssh_user", "root")
        ssh_key = os.path.expanduser(info.get("ssh_key", "~/.ssh/id_cluster"))

        return [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            ssh_key,
            "-p",
            str(ssh_port),
            f"{ssh_user}@{ssh_host}",
        ]

    def _run_ssh_command(
        self, node_id: str, command: str, timeout: int = 60
    ) -> tuple[bool, str]:
        """Run SSH command on a node.

        Returns:
            Tuple of (success, output)
        """
        ssh_cmd = self._get_ssh_command(node_id)
        try:
            result = subprocess.run(
                ssh_cmd + [command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    async def detect_capabilities(self, node_id: str) -> NodeCapabilities | None:
        """Detect capabilities of a node."""
        info = self.hosts.get(node_id, {})
        if not info:
            logger.error(f"Node {node_id} not found in config")
            return None

        caps = NodeCapabilities(
            node_id=node_id,
            ssh_host=info.get("ssh_host", ""),
            ssh_port=info.get("ssh_port", 22),
            ssh_user=info.get("ssh_user", "root"),
            ssh_key=info.get("ssh_key", "~/.ssh/id_cluster"),
            ringrift_path=info.get("ringrift_path", "~/ringrift/ai-service"),
            memory_gb=info.get("memory_gb", 0),
            cpu_cores=info.get("cpus", 0),
        )

        # Check GPU
        gpu_info = info.get("gpu", "")
        if gpu_info and "CPU" not in gpu_info.upper():
            caps.has_gpu = True
            caps.gpu_type = gpu_info
            # Parse GPU memory from string like "NVIDIA GH200 (96GB)"
            if "(" in gpu_info and "GB" in gpu_info:
                try:
                    mem_str = gpu_info.split("(")[1].split("GB")[0]
                    caps.gpu_memory_gb = int(mem_str)
                except (IndexError, ValueError):
                    pass

        # Check venv
        success, output = self._run_ssh_command(
            node_id, f"test -f {caps.ringrift_path}/venv/bin/python && echo 'yes'"
        )
        caps.venv_ready = success and "yes" in output

        # Check P2P
        success, output = self._run_ssh_command(
            node_id, "pgrep -f p2p_orchestrator || echo 'not running'"
        )
        caps.p2p_running = success and "not running" not in output

        # Determine suitability
        if caps.has_gpu:
            caps.suitable_for.append("selfplay")
            if caps.gpu_memory_gb >= 48:
                caps.suitable_for.append("training")
        else:
            caps.suitable_for.append("cpu_selfplay")

        return caps

    async def setup_node(self, node_id: str) -> bool:
        """Setup a single node with venv and dependencies."""
        info = self.hosts.get(node_id, {})
        if not info:
            logger.error(f"Node {node_id} not found in config")
            return False

        ringrift_path = info.get("ringrift_path", "~/ringrift/ai-service")
        logger.info(f"Setting up {node_id}...")

        # 1. Check if code exists
        success, output = self._run_ssh_command(
            node_id, f"test -d {ringrift_path} && echo 'exists'"
        )
        if not success or "exists" not in output:
            logger.error(f"RingRift code not found on {node_id}")
            return False

        # 2. Create venv if needed
        success, output = self._run_ssh_command(
            node_id,
            f"cd {ringrift_path} && test -d venv || python3 -m venv venv",
            timeout=60,
        )
        if not success:
            logger.error(f"Failed to create venv on {node_id}: {output}")
            return False

        # 3. Install core dependencies
        logger.info(f"Installing dependencies on {node_id}...")
        install_cmd = f"""
cd {ringrift_path}
source venv/bin/activate
pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || pip install torch -q
pip install numpy aiohttp pyyaml psutil sqlalchemy requests aiofiles tqdm rich -q
echo 'Core packages installed'
"""
        success, output = self._run_ssh_command(node_id, install_cmd, timeout=300)
        if not success:
            logger.error(f"Failed to install dependencies on {node_id}: {output}")
            return False

        logger.info(f"Setup complete for {node_id}")
        return True

    async def start_p2p(self, node_id: str, peer_host: str = "46.62.147.150") -> bool:
        """Start P2P orchestrator on a node."""
        info = self.hosts.get(node_id, {})
        if not info:
            logger.error(f"Node {node_id} not found in config")
            return False

        ringrift_path = info.get("ringrift_path", "~/ringrift/ai-service")
        advertise_host = info.get("ssh_host", "")
        advertise_port = info.get("ssh_port", 8770)

        # For Vast.ai, use SSH host:port for advertising
        if "vast" in node_id.lower():
            advertise_port = info.get("ssh_port", 8770)

        cmd = f"""
cd {ringrift_path}
source venv/bin/activate
mkdir -p logs
pkill -f p2p_orchestrator 2>/dev/null || true
sleep 1
setsid python scripts/p2p_orchestrator.py \\
    --node-id {node_id} \\
    --advertise-host {advertise_host} \\
    --peers {peer_host}:8770 \\
    > logs/p2p.log 2>&1 &
sleep 2
pgrep -f p2p_orchestrator && echo 'P2P started'
"""
        success, output = self._run_ssh_command(node_id, cmd, timeout=30)
        if success and "P2P started" in output:
            logger.info(f"P2P started on {node_id}")
            return True
        else:
            logger.error(f"Failed to start P2P on {node_id}: {output}")
            return False

    async def start_selfplay(
        self,
        node_id: str,
        board_type: str = "hex8",
        num_players: int = 2,
        num_games: int = 1000,
    ) -> bool:
        """Start selfplay on a GPU node."""
        info = self.hosts.get(node_id, {})
        if not info:
            logger.error(f"Node {node_id} not found in config")
            return False

        ringrift_path = info.get("ringrift_path", "~/ringrift/ai-service")

        cmd = f"""
cd {ringrift_path}
source venv/bin/activate
mkdir -p data/games logs
nohup python scripts/selfplay.py \\
    --board {board_type} --num-players {num_players} \\
    --engine gumbel --num-games {num_games} \\
    --output-dir data/games/selfplay_{board_type}_{num_players}p \\
    > logs/selfplay_{board_type}_{num_players}p.log 2>&1 &
sleep 2
pgrep -f 'selfplay.py' && echo 'Selfplay started'
"""
        success, output = self._run_ssh_command(node_id, cmd, timeout=30)
        if success and "Selfplay started" in output:
            logger.info(f"Selfplay started on {node_id}")
            return True
        else:
            logger.error(f"Failed to start selfplay on {node_id}: {output}")
            return False

    async def health_check(self, node_id: str) -> dict[str, Any]:
        """Check health of a node."""
        health = {
            "node_id": node_id,
            "reachable": False,
            "p2p_running": False,
            "selfplay_running": False,
            "gpu_available": False,
        }

        # Check reachability
        success, _ = self._run_ssh_command(node_id, "hostname", timeout=5)
        health["reachable"] = success

        if not success:
            return health

        # Check P2P
        success, output = self._run_ssh_command(
            node_id, "pgrep -f p2p_orchestrator || echo 'not running'"
        )
        health["p2p_running"] = success and "not running" not in output

        # Check selfplay
        success, output = self._run_ssh_command(
            node_id, "pgrep -f selfplay.py || echo 'not running'"
        )
        health["selfplay_running"] = success and "not running" not in output

        # Check GPU
        success, output = self._run_ssh_command(
            node_id, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null || echo 'no gpu'"
        )
        health["gpu_available"] = success and "no gpu" not in output

        return health

    async def setup_all(self) -> dict[str, bool]:
        """Setup all ready nodes."""
        results = {}
        for node_id in self.get_ready_hosts():
            results[node_id] = await self.setup_node(node_id)
        return results

    async def start_p2p_all(self) -> dict[str, bool]:
        """Start P2P on all ready nodes."""
        results = {}
        for node_id in self.get_ready_hosts():
            results[node_id] = await self.start_p2p(node_id)
        return results

    async def start_selfplay_all(self) -> dict[str, bool]:
        """Start selfplay on all GPU nodes."""
        results = {}
        for node_id in self.get_gpu_hosts():
            results[node_id] = await self.start_selfplay(node_id)
        return results

    async def daemon_loop(self, interval: int = 300) -> None:
        """Continuous health check and self-healing loop."""
        logger.info("Starting daemon loop...")
        while True:
            for node_id in self.get_ready_hosts():
                try:
                    health = await self.health_check(node_id)

                    if not health["reachable"]:
                        logger.warning(f"Node {node_id} unreachable")
                        continue

                    if not health["p2p_running"]:
                        logger.info(f"Restarting P2P on {node_id}")
                        await self.start_p2p(node_id)

                    # Start selfplay on GPU nodes that aren't running any
                    if health["gpu_available"] and not health["selfplay_running"]:
                        logger.info(f"Starting selfplay on idle GPU node {node_id}")
                        await self.start_selfplay(node_id)

                except Exception as e:
                    logger.error(f"Health check failed for {node_id}: {e}")

            logger.info(f"Health check complete. Sleeping {interval}s...")
            await asyncio.sleep(interval)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Master Cluster Automation")
    parser.add_argument("--setup-all", action="store_true", help="Setup all ready nodes")
    parser.add_argument("--setup-node", type=str, help="Setup specific node")
    parser.add_argument("--start-p2p", action="store_true", help="Start P2P on all nodes")
    parser.add_argument("--start-p2p-node", type=str, help="Start P2P on specific node")
    parser.add_argument("--start-selfplay", action="store_true", help="Start selfplay on GPU nodes")
    parser.add_argument("--health-check", action="store_true", help="Check health of all nodes")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--interval", type=int, default=300, help="Daemon check interval")
    parser.add_argument("--config", type=str, help="Path to distributed_hosts.yaml")

    args = parser.parse_args()

    automation = MasterClusterAutomation(config_path=args.config)

    if args.setup_all:
        results = asyncio.run(automation.setup_all())
        for node_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {node_id}")

    elif args.setup_node:
        success = asyncio.run(automation.setup_node(args.setup_node))
        print(f"{'✅' if success else '❌'} {args.setup_node}")

    elif args.start_p2p:
        results = asyncio.run(automation.start_p2p_all())
        for node_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {node_id}")

    elif args.start_p2p_node:
        success = asyncio.run(automation.start_p2p(args.start_p2p_node))
        print(f"{'✅' if success else '❌'} {args.start_p2p_node}")

    elif args.start_selfplay:
        results = asyncio.run(automation.start_selfplay_all())
        for node_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {node_id}")

    elif args.health_check:
        for node_id in automation.get_ready_hosts():
            health = asyncio.run(automation.health_check(node_id))
            reachable = "✅" if health["reachable"] else "❌"
            p2p = "🟢" if health["p2p_running"] else "🔴"
            selfplay = "🟢" if health["selfplay_running"] else "⚪"
            gpu = "🖥️" if health["gpu_available"] else ""
            print(f"{reachable} {node_id}: P2P={p2p} Selfplay={selfplay} {gpu}")

    elif args.daemon:
        asyncio.run(automation.daemon_loop(args.interval))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

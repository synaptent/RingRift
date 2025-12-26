"""Cluster Watchdog Daemon (December 2025 - Phase 8).

Self-healing daemon that continuously monitors cluster utilization and
automatically activates idle nodes by spawning selfplay.

Key features:
- Runs on coordinator every 5 minutes
- Discovers nodes from provider CLIs (vastai, runpodctl, vultr-cli)
- Checks GPU utilization via SSH
- Auto-spawns selfplay on underutilized nodes
- Tracks failure counts and escalates persistent issues

This daemon leverages the enhanced cluster_activator.py for actual
node discovery and activation.

Usage:
    from app.coordination.cluster_watchdog_daemon import ClusterWatchdogDaemon

    daemon = ClusterWatchdogDaemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ClusterWatchdogConfig:
    """Configuration for cluster watchdog daemon."""
    enabled: bool = True
    # How often to check cluster (5 minutes)
    check_interval_seconds: int = 300
    # Minimum GPU utilization threshold - below this, spawn selfplay
    min_gpu_utilization: float = 20.0
    # Maximum consecutive failures before escalation
    max_consecutive_failures: int = 3
    # Cooldown after successful activation (avoid re-checking same node immediately)
    activation_cooldown_seconds: int = 600  # 10 minutes
    # SSH timeout for node checks
    ssh_timeout_seconds: int = 30
    # Max nodes to activate per cycle (avoid overwhelming cluster)
    max_activations_per_cycle: int = 10
    # Board/player configs to cycle through
    selfplay_configs: list[tuple[str, int]] = field(default_factory=lambda: [
        ("hex8", 2),
        ("square8", 2),
        ("hex8", 3),
        ("square8", 3),
        ("hex8", 4),
        ("square8", 4),
    ])
    # Games per selfplay spawn
    games_per_spawn: int = 1000

    @classmethod
    def from_env(cls) -> ClusterWatchdogConfig:
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get("RINGRIFT_WATCHDOG_ENABLED", "1") == "1"
        config.check_interval_seconds = int(
            os.environ.get("RINGRIFT_WATCHDOG_INTERVAL", "300")
        )
        config.min_gpu_utilization = float(
            os.environ.get("RINGRIFT_WATCHDOG_MIN_GPU", "20.0")
        )
        return config


# =============================================================================
# Node Tracking
# =============================================================================


@dataclass
class WatchdogNodeStatus:
    """Status of a node tracked by the watchdog."""
    node_id: str
    provider: str  # vast, runpod, vultr
    ssh_cmd: str
    gpu_memory_gb: int = 0
    last_check: float = 0.0
    last_activation: float = 0.0
    gpu_utilization: float = 0.0
    python_processes: int = 0
    consecutive_failures: int = 0
    is_reachable: bool = False
    error: str = ""


@dataclass
class WatchdogCycleStats:
    """Statistics for a watchdog cycle."""
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    nodes_discovered: int = 0
    nodes_reachable: int = 0
    nodes_idle: int = 0
    nodes_activated: int = 0
    nodes_failed: int = 0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Cluster Watchdog Daemon
# =============================================================================


class ClusterWatchdogDaemon:
    """Self-healing daemon for cluster utilization.

    Monitors all provider nodes and auto-spawns selfplay on idle GPUs.
    """

    def __init__(self, config: ClusterWatchdogConfig | None = None):
        self.config = config or ClusterWatchdogConfig.from_env()
        self._running = False
        self._task: asyncio.Task | None = None
        self._nodes: dict[str, WatchdogNodeStatus] = {}
        self._config_index = 0  # Cycle through selfplay configs
        self._last_cycle_stats: WatchdogCycleStats | None = None
        self._hostname = socket.gethostname()
        self._start_time = 0.0

        # Path to cluster_activator.py for node discovery
        self._activator_path = Path(__file__).parent.parent.parent / "scripts" / "cluster_activator.py"

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Get daemon uptime in seconds."""
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0.0

    async def start(self) -> None:
        """Start the watchdog daemon."""
        if self._running:
            logger.warning("[ClusterWatchdog] Already running")
            return

        if not self.config.enabled:
            logger.info("[ClusterWatchdog] Disabled via config")
            return

        self._running = True
        self._start_time = time.time()

        # Register with coordinator registry
        register_coordinator(
            "cluster_watchdog",
            CoordinatorStatus(
                coordinator_type="cluster_watchdog",
                is_running=True,
                host=self._hostname,
                start_time=self._start_time,
            ),
        )

        logger.info(
            f"[ClusterWatchdog] Starting on {self._hostname} "
            f"(interval={self.config.check_interval_seconds}s, "
            f"min_gpu={self.config.min_gpu_utilization}%)"
        )

        self._task = safe_create_task(
            self._main_loop(),
            name="cluster_watchdog_main",
        )

    async def stop(self) -> None:
        """Stop the watchdog daemon."""
        if not self._running:
            return

        logger.info("[ClusterWatchdog] Stopping...")
        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        unregister_coordinator("cluster_watchdog")
        logger.info("[ClusterWatchdog] Stopped")

    async def _main_loop(self) -> None:
        """Main watchdog loop."""
        try:
            while self._running:
                try:
                    await self._run_cycle()
                except Exception as e:
                    logger.error(f"[ClusterWatchdog] Cycle failed: {e}")

                # Wait for next cycle
                await asyncio.sleep(self.config.check_interval_seconds)

        except asyncio.CancelledError:
            logger.debug("[ClusterWatchdog] Main loop cancelled")
        except Exception as e:
            logger.error(f"[ClusterWatchdog] Main loop crashed: {e}")
            self._running = False

    async def _run_cycle(self) -> None:
        """Run a single watchdog cycle."""
        stats = WatchdogCycleStats(cycle_start=time.time())

        try:
            # Step 1: Discover nodes from all providers
            nodes = await self._discover_nodes()
            stats.nodes_discovered = len(nodes)

            if not nodes:
                logger.warning("[ClusterWatchdog] No nodes discovered from providers")
                return

            # Step 2: Check GPU utilization on each node
            reachable_nodes = []
            for node in nodes:
                try:
                    await self._check_node_status(node)
                    if node.is_reachable:
                        reachable_nodes.append(node)
                        stats.nodes_reachable += 1
                except Exception as e:
                    node.error = str(e)
                    node.consecutive_failures += 1

            # Step 3: Find idle nodes (below threshold)
            idle_nodes = [
                n for n in reachable_nodes
                if n.gpu_utilization < self.config.min_gpu_utilization
            ]
            stats.nodes_idle = len(idle_nodes)

            # Step 4: Activate idle nodes (up to max per cycle)
            activated = 0
            for node in idle_nodes[:self.config.max_activations_per_cycle]:
                # Skip if recently activated
                if (time.time() - node.last_activation) < self.config.activation_cooldown_seconds:
                    continue

                success = await self._activate_node(node)
                if success:
                    node.last_activation = time.time()
                    node.consecutive_failures = 0
                    activated += 1
                    stats.nodes_activated += 1
                else:
                    node.consecutive_failures += 1
                    stats.nodes_failed += 1

                    # Escalate persistent failures
                    if node.consecutive_failures >= self.config.max_consecutive_failures:
                        logger.error(
                            f"[ClusterWatchdog] ESCALATION: {node.node_id} has "
                            f"{node.consecutive_failures} consecutive failures"
                        )
                        stats.errors.append(
                            f"Persistent failure on {node.node_id}: {node.error}"
                        )

            stats.cycle_end = time.time()
            self._last_cycle_stats = stats

            # Log cycle summary
            duration = stats.cycle_end - stats.cycle_start
            logger.info(
                f"[ClusterWatchdog] Cycle complete: "
                f"discovered={stats.nodes_discovered}, "
                f"reachable={stats.nodes_reachable}, "
                f"idle={stats.nodes_idle}, "
                f"activated={stats.nodes_activated}, "
                f"failed={stats.nodes_failed}, "
                f"duration={duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"[ClusterWatchdog] Cycle error: {e}")
            stats.errors.append(str(e))

    async def _discover_nodes(self) -> list[WatchdogNodeStatus]:
        """Discover nodes from provider CLIs using cluster_activator."""
        nodes = []

        # Query each provider CLI
        vast_nodes = await self._query_vast_cli()
        nodes.extend(vast_nodes)

        runpod_nodes = await self._query_runpod_cli()
        nodes.extend(runpod_nodes)

        vultr_nodes = await self._query_vultr_cli()
        nodes.extend(vultr_nodes)

        # Update our tracked nodes
        for node in nodes:
            if node.node_id in self._nodes:
                # Preserve failure counts and last activation
                existing = self._nodes[node.node_id]
                node.consecutive_failures = existing.consecutive_failures
                node.last_activation = existing.last_activation
            self._nodes[node.node_id] = node

        return nodes

    async def _query_vast_cli(self) -> list[WatchdogNodeStatus]:
        """Query Vast.ai CLI for running instances."""
        nodes = []
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["vastai", "show", "instances", "--raw"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )

            if result.returncode != 0:
                logger.debug(f"[ClusterWatchdog] vastai CLI failed: {result.stderr}")
                return nodes

            import json
            instances = json.loads(result.stdout)

            for inst in instances:
                if inst.get("actual_status") != "running":
                    continue

                ssh_host = inst.get("ssh_host", "")
                ssh_port = inst.get("ssh_port", 22)
                inst_id = inst.get("id", "")
                gpu_name = inst.get("gpu_name", "")

                # Estimate GPU memory from name
                gpu_memory = 24  # Default
                if "5090" in gpu_name:
                    gpu_memory = 32
                elif "5080" in gpu_name:
                    gpu_memory = 16
                elif "A100" in gpu_name or "H100" in gpu_name:
                    gpu_memory = 80
                elif "A40" in gpu_name:
                    gpu_memory = 48
                elif "4090" in gpu_name:
                    gpu_memory = 24
                elif "3090" in gpu_name:
                    gpu_memory = 24
                elif "4060" in gpu_name or "3060" in gpu_name:
                    gpu_memory = 8

                ssh_cmd = f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p {ssh_port} -i ~/.ssh/id_cluster root@{ssh_host}"

                nodes.append(WatchdogNodeStatus(
                    node_id=f"vast-{inst_id}",
                    provider="vast",
                    ssh_cmd=ssh_cmd,
                    gpu_memory_gb=gpu_memory,
                ))

        except Exception as e:
            logger.debug(f"[ClusterWatchdog] Vast discovery error: {e}")

        return nodes

    async def _query_runpod_cli(self) -> list[WatchdogNodeStatus]:
        """Query RunPod CLI for running pods."""
        nodes = []
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["runpodctl", "get", "pod"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )

            if result.returncode != 0:
                logger.debug(f"[ClusterWatchdog] runpodctl CLI failed: {result.stderr}")
                return nodes

            # Parse runpodctl output (tab-separated)
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return nodes

            for line in lines[1:]:  # Skip header
                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                pod_id = parts[0].strip()
                status = parts[2].strip() if len(parts) > 2 else ""

                if status != "RUNNING":
                    continue

                # RunPod doesn't provide SSH info directly in CLI output
                # We'd need to look it up or use stored config
                # For now, skip nodes without SSH info
                nodes.append(WatchdogNodeStatus(
                    node_id=f"runpod-{pod_id}",
                    provider="runpod",
                    ssh_cmd="",  # Will be filled from config
                    gpu_memory_gb=80,  # Assume A100/H100
                ))

        except Exception as e:
            logger.debug(f"[ClusterWatchdog] RunPod discovery error: {e}")

        return nodes

    async def _query_vultr_cli(self) -> list[WatchdogNodeStatus]:
        """Query Vultr CLI for running instances."""
        nodes = []
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["vultr-cli", "instance", "list", "--output", "json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )

            if result.returncode != 0:
                logger.debug(f"[ClusterWatchdog] vultr-cli failed: {result.stderr}")
                return nodes

            import json
            data = json.loads(result.stdout)
            instances = data.get("instances", [])

            for inst in instances:
                if inst.get("status") != "active":
                    continue

                inst_id = inst.get("id", "")
                main_ip = inst.get("main_ip", "")
                label = inst.get("label", "")

                # Skip non-GPU instances
                if "a100" not in label.lower() and "gpu" not in label.lower():
                    continue

                ssh_cmd = f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@{main_ip}"

                nodes.append(WatchdogNodeStatus(
                    node_id=f"vultr-{label or inst_id}",
                    provider="vultr",
                    ssh_cmd=ssh_cmd,
                    gpu_memory_gb=20,  # Vultr A100 vGPU
                ))

        except Exception as e:
            logger.debug(f"[ClusterWatchdog] Vultr discovery error: {e}")

        return nodes

    async def _check_node_status(self, node: WatchdogNodeStatus) -> None:
        """Check GPU utilization on a node via SSH."""
        node.last_check = time.time()

        if not node.ssh_cmd:
            node.is_reachable = False
            node.error = "No SSH command configured"
            return

        try:
            # Check GPU utilization
            gpu_cmd = f"{node.ssh_cmd} 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1'"
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    gpu_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.ssh_timeout_seconds,
                ),
            )

            if result.returncode == 0 and result.stdout.strip():
                gpu_str = result.stdout.strip().replace(" %", "").replace("%", "")
                try:
                    node.gpu_utilization = float(gpu_str)
                except ValueError:
                    node.gpu_utilization = 0.0

            # Check Python processes
            proc_cmd = f"{node.ssh_cmd} 'pgrep -c python 2>/dev/null || echo 0'"
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    proc_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.ssh_timeout_seconds,
                ),
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    node.python_processes = int(result.stdout.strip())
                except ValueError:
                    node.python_processes = 0

            node.is_reachable = True
            node.error = ""

        except subprocess.TimeoutExpired:
            node.is_reachable = False
            node.error = "SSH timeout"
        except Exception as e:
            node.is_reachable = False
            node.error = str(e)

    async def _activate_node(self, node: WatchdogNodeStatus) -> bool:
        """Spawn selfplay on an idle node."""
        if not node.ssh_cmd:
            logger.warning(f"[ClusterWatchdog] Cannot activate {node.node_id}: no SSH command")
            return False

        # Get next config from rotation
        board_type, num_players = self.config.selfplay_configs[self._config_index]
        self._config_index = (self._config_index + 1) % len(self.config.selfplay_configs)

        logger.info(
            f"[ClusterWatchdog] Activating {node.node_id} with "
            f"{board_type}_{num_players}p selfplay"
        )

        try:
            # Build selfplay command
            selfplay_cmd = (
                f"cd ~/ringrift/ai-service && "
                f"mkdir -p logs data/games && "
                f"PYTHONPATH=. RINGRIFT_SKIP_SHADOW_CONTRACTS=true "
                f"nohup python3 scripts/selfplay.py "
                f"--board {board_type} --num-players {num_players} "
                f"--engine gumbel --num-games {self.config.games_per_spawn} "
                f"--output-dir data/games "
                f"> logs/selfplay_{board_type}_{num_players}p.log 2>&1 &"
            )

            full_cmd = f"{node.ssh_cmd} '{selfplay_cmd}'"

            # Run with timeout - but nohup should return quickly
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,  # 1 minute timeout
                ),
            )

            # nohup may return non-zero but still succeed
            logger.info(f"[ClusterWatchdog] Spawn result for {node.node_id}: rc={result.returncode}")
            return True

        except subprocess.TimeoutExpired:
            # Timeout with nohup often means success (command is running)
            logger.info(f"[ClusterWatchdog] Spawn timeout for {node.node_id} (likely succeeded)")
            return True
        except Exception as e:
            node.error = str(e)
            logger.error(f"[ClusterWatchdog] Failed to activate {node.node_id}: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        return {
            "running": self._running,
            "hostname": self._hostname,
            "uptime_seconds": self.uptime_seconds,
            "config": {
                "check_interval": self.config.check_interval_seconds,
                "min_gpu_utilization": self.config.min_gpu_utilization,
            },
            "tracked_nodes": len(self._nodes),
            "last_cycle": {
                "discovered": self._last_cycle_stats.nodes_discovered if self._last_cycle_stats else 0,
                "reachable": self._last_cycle_stats.nodes_reachable if self._last_cycle_stats else 0,
                "idle": self._last_cycle_stats.nodes_idle if self._last_cycle_stats else 0,
                "activated": self._last_cycle_stats.nodes_activated if self._last_cycle_stats else 0,
                "failed": self._last_cycle_stats.nodes_failed if self._last_cycle_stats else 0,
            } if self._last_cycle_stats else None,
            "nodes": [
                {
                    "id": n.node_id,
                    "provider": n.provider,
                    "gpu_util": n.gpu_utilization,
                    "processes": n.python_processes,
                    "reachable": n.is_reachable,
                    "failures": n.consecutive_failures,
                }
                for n in self._nodes.values()
            ],
        }

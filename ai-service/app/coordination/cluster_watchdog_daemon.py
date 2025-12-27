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

Dec 2025: Refactored to use BaseDaemon base class (~80 LOC reduction).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.base_daemon import BaseDaemon, DaemonConfig

logger = logging.getLogger(__name__)

# Health event emission (uses safe fallbacks internally)
from app.coordination.event_emitters import (
    emit_health_check_failed,
    emit_health_check_passed,
    emit_node_unhealthy,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ClusterWatchdogConfig(DaemonConfig):
    """Configuration for cluster watchdog daemon.

    Extends DaemonConfig with watchdog-specific settings.
    """

    # Watchdog-specific settings
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
    def from_env(cls, prefix: str = "RINGRIFT_WATCHDOG") -> "ClusterWatchdogConfig":
        """Load configuration from environment variables."""
        config = cls()
        # Load base config
        config.enabled = os.environ.get(f"{prefix}_ENABLED", "1") == "1"
        if os.environ.get(f"{prefix}_INTERVAL"):
            config.check_interval_seconds = int(os.environ.get(f"{prefix}_INTERVAL", "300"))
        # Load watchdog-specific
        if os.environ.get(f"{prefix}_MIN_GPU"):
            config.min_gpu_utilization = float(os.environ.get(f"{prefix}_MIN_GPU", "20.0"))
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


class ClusterWatchdogDaemon(BaseDaemon[ClusterWatchdogConfig]):
    """Self-healing daemon for cluster utilization.

    Monitors all provider nodes and auto-spawns selfplay on idle GPUs.

    Inherits from BaseDaemon which provides:
    - Lifecycle management (start/stop)
    - Coordinator protocol registration
    - Protected main loop with error handling
    - Health check interface
    """

    def __init__(self, config: ClusterWatchdogConfig | None = None):
        super().__init__(config)
        self._nodes: dict[str, WatchdogNodeStatus] = {}
        self._config_index = 0  # Cycle through selfplay configs
        self._last_cycle_stats: WatchdogCycleStats | None = None
        # December 2025: Track cluster health for spawn decisions
        self._cluster_healthy: bool = True

        # Path to cluster_activator.py for node discovery
        self._activator_path = Path(__file__).parent.parent.parent / "scripts" / "cluster_activator.py"

    @staticmethod
    def _get_default_config() -> ClusterWatchdogConfig:
        """Return default configuration."""
        return ClusterWatchdogConfig.from_env()

    async def _on_start(self) -> None:
        """Log watchdog-specific startup info."""
        logger.info(
            f"[{self._get_daemon_name()}] Config: "
            f"interval={self.config.check_interval_seconds}s, "
            f"min_gpu={self.config.min_gpu_utilization}%"
        )
        # Subscribe to cluster events (December 2025)
        await self._subscribe_to_events()

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant cluster events."""
        try:
            from app.coordination.event_router import DataEventType, get_event_router

            router = get_event_router()
            router.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)
            router.subscribe(DataEventType.HOST_ONLINE, self._on_host_online)
            # December 2025: Subscribe to cluster health for spawn pause
            router.subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy)
            router.subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy)
            logger.info(f"[{self._get_daemon_name()}] Subscribed to cluster events")
        except ImportError:
            logger.debug(f"[{self._get_daemon_name()}] Event router not available")
        except Exception as e:
            logger.warning(f"[{self._get_daemon_name()}] Failed to subscribe: {e}")

    async def _on_host_offline(self, event) -> None:
        """Handle host going offline."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", "unknown")
            logger.info(f"[{self._get_daemon_name()}] Host offline: {host}")
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Error handling host offline: {e}")

    async def _on_host_online(self, event) -> None:
        """Handle host coming online."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", "unknown")
            logger.info(f"[{self._get_daemon_name()}] Host online: {host}")
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Error handling host online: {e}")

    async def _on_cluster_unhealthy(self, event) -> None:
        """Handle cluster becoming unhealthy - pause spawning."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            reason = payload.get("reason", "unknown")
            logger.warning(f"[{self._get_daemon_name()}] Cluster unhealthy: {reason} - pausing spawning")
            self._cluster_healthy = False
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Error handling cluster unhealthy: {e}")

    async def _on_cluster_healthy(self, event) -> None:
        """Handle cluster becoming healthy - resume spawning."""
        try:
            logger.info(f"[{self._get_daemon_name()}] Cluster healthy - resuming spawning")
            self._cluster_healthy = True
        except Exception as e:
            logger.debug(f"[{self._get_daemon_name()}] Error handling cluster healthy: {e}")

    async def _run_cycle(self) -> None:
        """Run a single watchdog cycle."""
        stats = WatchdogCycleStats(cycle_start=time.time())

        # December 2025: Skip spawning when cluster is unhealthy
        if not self._cluster_healthy:
            logger.info("[ClusterWatchdog] Skipping cycle - cluster unhealthy")
            return

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
                        # Emit NODE_UNHEALTHY event
                        await emit_node_unhealthy(
                            node_id=node.node_id,
                            reason=f"Persistent failures: {node.error}",
                            gpu_utilization=node.gpu_utilization,
                            consecutive_failures=node.consecutive_failures,
                            source="cluster_watchdog_daemon",
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
            result = await asyncio.get_running_loop().run_in_executor(
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
            result = await asyncio.get_running_loop().run_in_executor(
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
            result = await asyncio.get_running_loop().run_in_executor(
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
            # Check GPU utilization - use shlex to avoid shell=True
            ssh_parts = shlex.split(node.ssh_cmd)
            gpu_remote_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1"
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_parts + [gpu_remote_cmd],
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

            # Check Python processes - use shlex to avoid shell=True
            proc_remote_cmd = "pgrep -c python 2>/dev/null || echo 0"
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_parts + [proc_remote_cmd],
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

            # Emit health check passed event
            await emit_health_check_passed(
                node_id=node.node_id,
                check_type="ssh_gpu",
                source="cluster_watchdog_daemon",
            )

        except subprocess.TimeoutExpired:
            node.is_reachable = False
            node.error = "SSH timeout"
            await self._emit_health_failure(node, "SSH timeout")
        except Exception as e:
            node.is_reachable = False
            node.error = str(e)
            await self._emit_health_failure(node, str(e))

    async def _emit_health_failure(self, node: WatchdogNodeStatus, reason: str) -> None:
        """Emit health check failure event."""
        await emit_health_check_failed(
            node_id=node.node_id,
            reason=reason,
            check_type="ssh_gpu",
            error=node.error,
            source="cluster_watchdog_daemon",
        )

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
            # Build selfplay command - use shlex to avoid shell=True
            ssh_parts = shlex.split(node.ssh_cmd)
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

            # Run with timeout - but nohup should return quickly
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_parts + [selfplay_cmd],
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

    async def health_check(self) -> bool:
        """Check if the daemon is healthy.

        Returns True if the daemon is running and functioning properly.
        Used by DaemonManager for crash detection and auto-restart.

        Dec 2025: Added for DaemonManager health monitoring integration.
        """
        # Check if daemon is running
        if not self._running:
            return False

        # Check if we have recent cycle data (within 2x check interval)
        if self._last_cycle_stats is not None:
            max_age = self.config.check_interval_seconds * 2
            age = time.time() - self._last_cycle_stats.cycle_end
            if age > max_age:
                logger.warning(
                    f"[{self._get_daemon_name()}] Health check: stale cycle data "
                    f"(age={age:.0f}s, max={max_age}s)"
                )
                return False

            # Check for excessive errors in last cycle
            if len(self._last_cycle_stats.errors) > 5:
                logger.warning(
                    f"[{self._get_daemon_name()}] Health check: too many errors "
                    f"({len(self._last_cycle_stats.errors)})"
                )
                return False

        return True

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring.

        Extends base class status with watchdog-specific fields.
        """
        status = super().get_status()

        # Add watchdog-specific config
        status["config"]["min_gpu_utilization"] = self.config.min_gpu_utilization

        # Add watchdog-specific stats
        status["tracked_nodes"] = len(self._nodes)
        status["last_cycle"] = {
            "discovered": self._last_cycle_stats.nodes_discovered if self._last_cycle_stats else 0,
            "reachable": self._last_cycle_stats.nodes_reachable if self._last_cycle_stats else 0,
            "idle": self._last_cycle_stats.nodes_idle if self._last_cycle_stats else 0,
            "activated": self._last_cycle_stats.nodes_activated if self._last_cycle_stats else 0,
            "failed": self._last_cycle_stats.nodes_failed if self._last_cycle_stats else 0,
        } if self._last_cycle_stats else None
        status["nodes"] = [
            {
                "id": n.node_id,
                "provider": n.provider,
                "gpu_util": n.gpu_utilization,
                "processes": n.python_processes,
                "reachable": n.is_reachable,
                "failures": n.consecutive_failures,
            }
            for n in self._nodes.values()
        ]

        return status


# =============================================================================
# Module-level Singleton Factory (December 2025)
# =============================================================================

_cluster_watchdog_daemon: ClusterWatchdogDaemon | None = None


def get_cluster_watchdog_daemon() -> ClusterWatchdogDaemon:
    """Get the singleton ClusterWatchdogDaemon instance.

    December 2025: Added for consistency with other daemon factories.
    """
    global _cluster_watchdog_daemon
    if _cluster_watchdog_daemon is None:
        _cluster_watchdog_daemon = ClusterWatchdogDaemon()
    return _cluster_watchdog_daemon


def reset_cluster_watchdog_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _cluster_watchdog_daemon
    _cluster_watchdog_daemon = None

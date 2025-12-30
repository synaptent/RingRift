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

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus

logger = logging.getLogger(__name__)

# Health event emission (uses safe fallbacks internally)
from app.coordination.event_emitters import (
    emit_health_check_failed,
    emit_health_check_passed,
    emit_node_activated,
    emit_node_unhealthy,
)

# Import centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import (
        ClusterWatchdogDefaults,
        build_ssh_options,  # Dec 30, 2025: Centralized SSH config
    )
    HAS_CENTRALIZED_DEFAULTS = True
except ImportError:
    HAS_CENTRALIZED_DEFAULTS = False
    build_ssh_options = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ClusterWatchdogConfig:
    """Configuration for cluster watchdog daemon.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.
    """

    # Check interval (passed to HandlerBase as cycle_interval)
    check_interval_seconds: int = 300  # 5 minutes

    # Daemon control
    enabled: bool = True

    # Watchdog-specific settings (from centralized defaults)
    # Minimum GPU utilization threshold - below this, spawn selfplay
    min_gpu_utilization: float = (
        ClusterWatchdogDefaults.MIN_GPU_UTILIZATION if HAS_CENTRALIZED_DEFAULTS else 20.0
    )
    # Maximum consecutive failures before escalation
    max_consecutive_failures: int = (
        ClusterWatchdogDefaults.MAX_CONSECUTIVE_FAILURES if HAS_CENTRALIZED_DEFAULTS else 3
    )
    # Cooldown after successful activation (avoid re-checking same node immediately)
    activation_cooldown_seconds: int = (
        ClusterWatchdogDefaults.ACTIVATION_COOLDOWN if HAS_CENTRALIZED_DEFAULTS else 600
    )
    # SSH timeout for node checks
    ssh_timeout_seconds: int = (
        ClusterWatchdogDefaults.SSH_TIMEOUT if HAS_CENTRALIZED_DEFAULTS else 30
    )
    # Max nodes to activate per cycle (avoid overwhelming cluster)
    max_activations_per_cycle: int = (
        ClusterWatchdogDefaults.MAX_ACTIVATIONS_PER_CYCLE if HAS_CENTRALIZED_DEFAULTS else 10
    )
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
        config.enabled = os.environ.get(f"{prefix}_ENABLED", "1") == "1"
        if os.environ.get(f"{prefix}_INTERVAL"):
            config.check_interval_seconds = int(os.environ.get(f"{prefix}_INTERVAL", "300"))
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


class ClusterWatchdogDaemon(HandlerBase):
    """Self-healing daemon for cluster utilization.

    Monitors all provider nodes and auto-spawns selfplay on idle GPUs.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking
    - Uses _get_event_subscriptions() for event wiring
    """

    def __init__(self, config: ClusterWatchdogConfig | None = None):
        self._daemon_config = config or ClusterWatchdogConfig.from_env()

        super().__init__(
            name="ClusterWatchdogDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._nodes: dict[str, WatchdogNodeStatus] = {}
        self._config_index = 0  # Cycle through selfplay configs
        self._last_cycle_stats: WatchdogCycleStats | None = None
        # December 2025: Track cluster health for spawn decisions
        self._cluster_healthy: bool = True

        # Path to cluster_activator.py for node discovery
        self._activator_path = Path(__file__).parent.parent.parent / "scripts" / "cluster_activator.py"

    @property
    def config(self) -> ClusterWatchdogConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions for HandlerBase.

        December 2025: Migrated from _subscribe_to_events() to dict-based pattern.
        """
        return {
            "HOST_OFFLINE": self._on_host_offline,
            "HOST_ONLINE": self._on_host_online,
            "P2P_CLUSTER_UNHEALTHY": self._on_cluster_unhealthy,
            "P2P_CLUSTER_HEALTHY": self._on_cluster_healthy,
        }

    async def _on_start(self) -> None:
        """Log watchdog-specific startup info."""
        logger.info(
            f"[{self.name}] Config: "
            f"interval={self.config.check_interval_seconds}s, "
            f"min_gpu={self.config.min_gpu_utilization}%"
        )

    async def _on_host_offline(self, event) -> None:
        """Handle host going offline."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", "unknown")
            logger.info(f"[{self.name}] Host offline: {host}")
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"[{self.name}] Error handling host offline: {e}")

    async def _on_host_online(self, event) -> None:
        """Handle host coming online."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", "unknown")
            logger.info(f"[{self.name}] Host online: {host}")
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"[{self.name}] Error handling host online: {e}")

    async def _on_cluster_unhealthy(self, event) -> None:
        """Handle cluster becoming unhealthy - pause spawning."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            reason = payload.get("reason", "unknown")
            logger.warning(f"[{self.name}] Cluster unhealthy: {reason} - pausing spawning")
            self._cluster_healthy = False
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"[{self.name}] Error handling cluster unhealthy: {e}")

    async def _on_cluster_healthy(self, event) -> None:
        """Handle cluster becoming healthy - resume spawning."""
        try:
            logger.info(f"[{self.name}] Cluster healthy - resuming spawning")
            self._cluster_healthy = True
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"[{self.name}] Error handling cluster healthy: {e}")

    async def _on_stop(self) -> None:
        """Graceful shutdown handler.

        December 2025: Added for clean daemon shutdown with event emission.
        """
        try:
            # Emit shutdown event
            from app.coordination.event_emitters import emit_coordinator_shutdown

            await emit_coordinator_shutdown(
                coordinator_name=self.name,
                reason="graceful",
                remaining_tasks=0,
                state_snapshot={
                    "uptime_seconds": self.uptime_seconds,
                    "cycles_completed": getattr(self, "_cycles_completed", 0),
                    "cluster_healthy": self._cluster_healthy,
                },
            )
            logger.info(f"[{self.name}] Graceful shutdown complete")
        except ImportError:
            logger.debug(f"[{self.name}] Event emitters not available for shutdown")
        except Exception as e:
            logger.warning(f"[{self.name}] Error during shutdown: {e}")

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

                # Dec 30, 2025: Use centralized SSH config for consistent timeouts
                if build_ssh_options:
                    ssh_cmd = build_ssh_options(
                        key_path="~/.ssh/id_cluster",
                        provider="vast",
                        port=ssh_port,
                        include_keepalive=True,
                    )
                    ssh_cmd = f"{ssh_cmd} root@{ssh_host}"
                else:
                    ssh_cmd = f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p {ssh_port} -i ~/.ssh/id_cluster root@{ssh_host}"

                nodes.append(WatchdogNodeStatus(
                    node_id=f"vast-{inst_id}",
                    provider="vast",
                    ssh_cmd=ssh_cmd,
                    gpu_memory_gb=gpu_memory,
                ))

        except FileNotFoundError:
            # vastai CLI not installed - expected on non-coordinator nodes
            logger.debug("[ClusterWatchdog] Vast CLI not installed")
        except (subprocess.SubprocessError, ValueError, KeyError) as e:
            logger.info(f"[ClusterWatchdog] Vast discovery error: {e}")

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

        except FileNotFoundError:
            # runpodctl CLI not installed - expected on non-coordinator nodes
            logger.debug("[ClusterWatchdog] RunPod CLI not installed")
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            logger.info(f"[ClusterWatchdog] RunPod discovery error: {e}")

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

                # Dec 30, 2025: Use centralized SSH config for consistent timeouts
                if build_ssh_options:
                    ssh_cmd = build_ssh_options(
                        key_path="~/.ssh/id_ed25519",
                        provider="vultr",
                        include_keepalive=True,
                    )
                    ssh_cmd = f"{ssh_cmd} root@{main_ip}"
                else:
                    ssh_cmd = f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@{main_ip}"

                nodes.append(WatchdogNodeStatus(
                    node_id=f"vultr-{label or inst_id}",
                    provider="vultr",
                    ssh_cmd=ssh_cmd,
                    gpu_memory_gb=20,  # Vultr A100 vGPU
                ))

        except FileNotFoundError:
            # vultr-cli not installed - expected on non-coordinator nodes
            logger.debug("[ClusterWatchdog] Vultr CLI not installed")
        except (subprocess.SubprocessError, ValueError, KeyError) as e:
            logger.info(f"[ClusterWatchdog] Vultr discovery error: {e}")

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
            # December 2025: Emit NODE_ACTIVATED event for cluster coordination
            config_key = f"{board_type}_{num_players}p"
            asyncio.create_task(
                emit_node_activated(
                    node_id=node.node_id,
                    activation_type="selfplay",
                    config_key=config_key,
                )
            )
            return True

        except subprocess.TimeoutExpired:
            # Timeout with nohup often means success (command is running)
            logger.info(f"[ClusterWatchdog] Spawn timeout for {node.node_id} (likely succeeded)")
            # December 2025: Emit NODE_ACTIVATED event for cluster coordination
            config_key = f"{board_type}_{num_players}p"
            asyncio.create_task(
                emit_node_activated(
                    node_id=node.node_id,
                    activation_type="selfplay",
                    config_key=config_key,
                )
            )
            return True
        except Exception as e:
            node.error = str(e)
            logger.error(f"[ClusterWatchdog] Failed to activate {node.node_id}: {e}")
            return False

    def health_check(self) -> HealthCheckResult:
        """Check if the daemon is healthy.

        Returns HealthCheckResult for protocol compliance.
        Used by DaemonManager for crash detection and auto-restart.

        Dec 2025: Fixed to return HealthCheckResult instead of bool.
        Dec 2025: Converted from async to sync for DaemonManager compatibility.
        """
        is_healthy = True
        message = "Cluster watchdog daemon running"
        status = CoordinatorStatus.RUNNING

        # Check if daemon is running
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Cluster watchdog daemon not running",
                details={"running": False},
            )

        # Check if we have recent cycle data (within 2x check interval)
        if self._last_cycle_stats is not None:
            max_age = self.config.check_interval_seconds * 2
            age = time.time() - self._last_cycle_stats.cycle_end
            if age > max_age:
                is_healthy = False
                status = CoordinatorStatus.DEGRADED
                message = f"Stale cycle data (age={age:.0f}s, max={max_age}s)"
                logger.warning(f"[{self.name}] Health check: {message}")

            # Check for excessive errors in last cycle
            elif len(self._last_cycle_stats.errors) > 5:
                is_healthy = False
                status = CoordinatorStatus.DEGRADED
                message = f"Too many errors ({len(self._last_cycle_stats.errors)})"
                logger.warning(f"[{self.name}] Health check: {message}")

        cycle_stats = {}
        if self._last_cycle_stats:
            cycle_stats = {
                "nodes_discovered": self._last_cycle_stats.nodes_discovered,
                "nodes_activated": self._last_cycle_stats.nodes_activated,
                "cycle_errors": len(self._last_cycle_stats.errors),
            }

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details={
                "running": self._running,
                "cycles_completed": self._stats.cycles_completed,
                "errors_count": self._stats.errors_count,
                "cluster_healthy": self._cluster_healthy,
                **cycle_stats,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        from datetime import datetime

        # Convert last_activity timestamp to ISO string
        last_activity_iso = None
        if self._stats.last_activity > 0:
            last_activity_iso = datetime.fromtimestamp(self._stats.last_activity).isoformat()

        return {
            "running": self._running,
            "config": {
                "enabled": self.config.enabled,
                "check_interval_seconds": self.config.check_interval_seconds,
                "min_gpu_utilization": self.config.min_gpu_utilization,
                "max_activations_per_cycle": self.config.max_activations_per_cycle,
            },
            "stats": {
                "cycles_completed": self._stats.cycles_completed,
                "last_activity": last_activity_iso,
                "errors_count": self._stats.errors_count,
                "events_processed": self._stats.events_processed,
            },
            "cluster_healthy": self._cluster_healthy,
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


# =============================================================================
# Singleton Accessors (using HandlerBase class methods)
# =============================================================================


def get_cluster_watchdog_daemon() -> ClusterWatchdogDaemon:
    """Get or create the singleton ClusterWatchdogDaemon instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return ClusterWatchdogDaemon.get_instance()


def reset_cluster_watchdog_daemon() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    ClusterWatchdogDaemon.reset_instance()

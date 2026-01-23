"""Cluster Healing Loop for P2P Orchestrator.

January 2026: Automates cluster membership healing by proactively
connecting missing nodes from distributed_hosts.yaml.

Problem: Nodes may fail to join the P2P cluster due to:
- Wrong bootstrap peers configured
- P2P process crashed and didn't restart
- Network partitions isolating sub-clusters
- New nodes added to YAML but never contacted

Solution: Periodically scan distributed_hosts.yaml, identify nodes missing
from P2P peers, and proactively reach out via SSH to restart P2P with
correct bootstrap peers.

Usage:
    from scripts.p2p.loops import ClusterHealingLoop, ClusterHealingConfig

    healing_loop = ClusterHealingLoop(
        get_current_peers=lambda: orchestrator.get_peer_ids(),
        get_alive_peer_addresses=lambda: orchestrator.get_alive_peer_addresses(),
        on_node_joined=lambda node_id: logger.info(f"Healed: {node_id}"),
        config=ClusterHealingConfig(
            check_interval_seconds=300,
            max_heal_per_cycle=5,
        ),
    )
    await healing_loop.run_forever()

Events:
    NODE_HEALED: Emitted when a missing node is successfully restarted
    NODE_HEAL_FAILED: Emitted when healing attempt fails
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml

from .base import BaseLoop

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ClusterHealingConfig:
    """Configuration for cluster healing loop."""

    # Interval between healing cycles (seconds)
    # Default: 5 minutes - balance between responsiveness and not hammering nodes
    check_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_CLUSTER_HEALING_INTERVAL", "300")
        )
    )

    # Maximum nodes to attempt healing per cycle
    # Limits SSH load and allows gradual recovery
    # January 2026 Sprint 10: Reduced from 10 to 5 to prevent cascading restarts
    # January 5, 2026 (Phase 7.3): Now dynamically scaled based on cluster size.
    # This is the BASE value - actual value is computed by _get_dynamic_max_heal_per_cycle()
    # For large clusters (25+ nodes), this can be scaled up to 8 per cycle.
    max_heal_per_cycle: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_HEAL_PER_CYCLE", "3")
        )
    )

    # Phase 7.3: Enable dynamic scaling of max_heal_per_cycle based on cluster size
    # If True, max_heal_per_cycle scales: ≤10 peers=3, ≤25 peers=5, 25+ peers=8
    # If False, uses the static max_heal_per_cycle value
    dynamic_heal_rate: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_CLUSTER_HEALING_DYNAMIC_RATE", "true"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # SSH timeout for connecting to nodes (seconds)
    ssh_timeout_seconds: float = 30.0

    # Time to wait for P2P to start before checking status (seconds)
    p2p_startup_wait_seconds: float = 15.0

    # Path to distributed_hosts.yaml
    hosts_yaml_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "RINGRIFT_HOSTS_YAML",
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "distributed_hosts.yaml",
            )
        )
    )

    # Number of bootstrap peers to use when restarting P2P
    num_bootstrap_peers: int = 4

    # Backoff after failed healing attempt (seconds)
    backoff_after_failure_seconds: float = 600.0  # 10 minutes

    # Whether to emit events
    emit_events: bool = True

    # Whether the loop is enabled
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_CLUSTER_HEALING_ENABLED", "true"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Dry run mode (log actions without executing)
    dry_run: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_CLUSTER_HEALING_DRY_RUN", "false"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Time-based rate limiting (Jan 2026)
    # Maximum heals allowed in any rolling time window.
    # This provides hard protection against heal storms regardless of cycle timing.
    # January 5, 2026 (Phase 7.3): Increased from 5 to 12 for faster partition recovery.
    # With dynamic_heal_rate=True and 25+ nodes, can heal 8 nodes per cycle.
    # 12 per window (60s) allows 2 full cycles without rate limiting.
    max_heals_per_window: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_CLUSTER_HEALING_MAX_PER_WINDOW", "12")
        )
    )

    # Rolling window for rate limiting (seconds)
    rate_limit_window_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_CLUSTER_HEALING_RATE_WINDOW", "60")
        )
    )


# =============================================================================
# Host Info
# =============================================================================


@dataclass
class HostInfo:
    """Information about a cluster host from distributed_hosts.yaml."""

    name: str
    tailscale_ip: str
    ssh_user: str = "root"
    ssh_port: int = 22
    p2p_port: int = 8770
    enabled: bool = True
    role: str = "gpu_selfplay"

    @property
    def ssh_target(self) -> str:
        """SSH target string."""
        if self.ssh_port == 22:
            return f"{self.ssh_user}@{self.tailscale_ip}"
        return f"{self.ssh_user}@{self.tailscale_ip} -p {self.ssh_port}"

    @property
    def p2p_url(self) -> str:
        """P2P HTTP URL."""
        return f"http://{self.tailscale_ip}:{self.p2p_port}"


# =============================================================================
# Cluster Healing Loop
# =============================================================================


class ClusterHealingLoop(BaseLoop):
    """Background loop that heals cluster by connecting missing nodes.

    Key features:
    - Reads expected membership from distributed_hosts.yaml
    - Identifies nodes missing from P2P peers
    - Reaches out via SSH to restart P2P with correct bootstrap peers
    - Tracks healing attempts with backoff for repeated failures
    - Emits events for monitoring
    """

    def __init__(
        self,
        get_current_peers: Callable[[], set[str]],
        get_alive_peer_addresses: Callable[[], list[str]],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        on_node_joined: Callable[[str], None] | None = None,
        config: ClusterHealingConfig | None = None,
    ):
        """Initialize cluster healing loop.

        Args:
            get_current_peers: Callback returning set of known peer node_ids
            get_alive_peer_addresses: Callback returning list of alive peer
                HTTP addresses (e.g., "http://100.x.x.x:8770")
            emit_event: Optional callback to emit events
            on_node_joined: Optional callback when a node is successfully healed
            config: Healing configuration
        """
        self.config = config or ClusterHealingConfig()
        super().__init__(
            name="cluster_healing",
            interval=self.config.check_interval_seconds,
            enabled=self.config.enabled,
        )

        self._get_current_peers = get_current_peers
        self._get_alive_peer_addresses = get_alive_peer_addresses
        self._emit_event = emit_event
        self._on_node_joined = on_node_joined

        # Track healing attempts with backoff
        self._heal_failures: dict[str, int] = {}  # node_id -> failure count
        self._heal_next_attempt: dict[str, float] = {}  # node_id -> next attempt time

        # Time-based rate limiting (Jan 2026)
        # Track timestamps of successful heals for rolling window rate limit
        self._heal_timestamps: list[float] = []

        # Cache hosts config
        self._hosts_cache: dict[str, HostInfo] = {}
        self._hosts_cache_time: float = 0.0

        # Statistics
        self._stats_heal_attempts = 0
        self._stats_heal_successes = 0
        self._stats_heal_failures = 0
        self._stats_rate_limited = 0  # Jan 2026: track rate-limited attempts

    def _load_hosts_config(self) -> dict[str, HostInfo]:
        """Load and cache hosts from distributed_hosts.yaml."""
        # Cache for 5 minutes
        now = time.time()
        if self._hosts_cache and (now - self._hosts_cache_time) < 300:
            return self._hosts_cache

        try:
            with open(self.config.hosts_yaml_path) as f:
                config = yaml.safe_load(f)

            # Jan 12, 2026: Defensive check - YAML can return None/str if malformed
            if not isinstance(config, dict):
                logger.warning(
                    f"[ClusterHealing] YAML returned {type(config).__name__}, expected dict"
                )
                return self._hosts_cache or {}

            hosts = {}
            # Jan 2026: hosts is a dict {name: {props}}, not a list
            hosts_config = config.get("hosts", {})
            if not isinstance(hosts_config, dict):
                logger.warning(f"[ClusterHealing] hosts is {type(hosts_config).__name__}, expected dict")
                return self._hosts_cache or {}

            for name, host_data in hosts_config.items():
                if not name or not isinstance(host_data, dict):
                    continue

                # Skip disabled hosts
                if not host_data.get("enabled", True):
                    continue

                # Extract SSH info
                tailscale_ip = host_data.get("tailscale_ip", "")
                if not tailscale_ip:
                    continue

                ssh_user = host_data.get("ssh_user", "root")
                ssh_port = host_data.get("ssh_port", 22)
                p2p_port = host_data.get("p2p_port", 8770)
                role = host_data.get("role", "gpu_selfplay")

                hosts[name] = HostInfo(
                    name=name,
                    tailscale_ip=tailscale_ip,
                    ssh_user=ssh_user,
                    ssh_port=ssh_port,
                    p2p_port=p2p_port,
                    role=role,
                )

            self._hosts_cache = hosts
            self._hosts_cache_time = now
            logger.debug(f"[ClusterHealing] Loaded {len(hosts)} hosts from YAML")
            return hosts

        except Exception as e:
            logger.warning(f"[ClusterHealing] Failed to load hosts YAML: {e}")
            return self._hosts_cache or {}

    def _get_dynamic_max_heal_per_cycle(self) -> int:
        """Get dynamic healing rate based on cluster size.

        January 5, 2026 (Phase 7.3): Scales healing rate with cluster size
        to reduce partition recovery time from 1+ hour to 15-20 minutes.

        If dynamic_heal_rate is disabled, returns the static config value.

        Returns:
            Number of nodes to attempt healing per cycle:
            - ≤10 peers: 3 nodes (small cluster, conservative)
            - ≤25 peers: 5 nodes (medium cluster)
            - 25+ peers: 8 nodes (large cluster, aggressive healing)
        """
        if not self.config.dynamic_heal_rate:
            return self.config.max_heal_per_cycle

        # Count current P2P peers to determine cluster size
        try:
            current_peers = self._get_current_peers()
            peer_count = len(current_peers)
        except Exception:
            # Fallback to static config if peer count unavailable
            return self.config.max_heal_per_cycle

        if peer_count <= 10:
            return 3
        elif peer_count <= 25:
            return 5
        else:  # 25+ nodes - large cluster needs aggressive healing
            return 8

    def _get_missing_nodes(self) -> list[HostInfo]:
        """Get list of nodes in YAML but not in P2P peers."""
        hosts = self._load_hosts_config()
        current_peers = self._get_current_peers()
        now = time.time()

        # CRITICAL: Get local node_id to skip self-healing
        # Jan 21, 2026: This is ROOT CAUSE #1 of cluster instability - the healing
        # loop was restarting the local P2P process, killing the orchestrator itself.
        try:
            from app.config.node_identity import get_node_id_safe
            local_node_id = get_node_id_safe()
        except Exception:
            # Fallback: try to get from environment
            import socket
            local_node_id = os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())

        missing = []
        for name, host in hosts.items():
            # CRITICAL: Never heal the local node - it IS the orchestrator
            if name == local_node_id:
                logger.debug(f"[ClusterHealing] Skipping {name}: is local orchestrator")
                continue

            if name in current_peers:
                continue

            # Check backoff
            next_attempt = self._heal_next_attempt.get(name, 0)
            if now < next_attempt:
                remaining = next_attempt - now
                logger.debug(
                    f"[ClusterHealing] Skipping {name}: backoff ({remaining:.0f}s remaining)"
                )
                continue

            missing.append(host)

        return missing

    def _select_bootstrap_peers(self) -> list[str]:
        """Select bootstrap peers for P2P restart."""
        alive_addresses = self._get_alive_peer_addresses()
        if not alive_addresses:
            return []

        # Select random subset
        count = min(self.config.num_bootstrap_peers, len(alive_addresses))
        return random.sample(alive_addresses, count)

    async def _check_p2p_status(self, host: HostInfo) -> dict[str, Any] | None:
        """Check P2P status on a remote node via SSH + curl."""
        cmd = f"curl -s --connect-timeout 5 http://localhost:{host.p2p_port}/status"
        ssh_cmd = self._build_ssh_command(host, cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.ssh_timeout_seconds
            )

            if proc.returncode == 0 and stdout:
                import json

                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError:
                    pass
            return None

        except asyncio.TimeoutError:
            logger.debug(f"[ClusterHealing] SSH timeout checking {host.name}")
            return None
        except Exception as e:
            logger.debug(f"[ClusterHealing] Error checking {host.name}: {e}")
            return None

    async def _restart_p2p(
        self, host: HostInfo, bootstrap_peers: list[str]
    ) -> bool:
        """Restart P2P on a remote node with correct bootstrap peers."""
        peers_arg = ",".join(bootstrap_peers)

        # Build restart command
        restart_cmd = f"""
pkill -9 -f p2p_orchestrator || true
sleep 2
cd ~/ringrift/ai-service && \\
nohup python3 scripts/p2p_orchestrator.py --peers "{peers_arg}" \\
  > logs/p2p_orchestrator.log 2>&1 &
sleep 3
echo "P2P restarted"
"""
        ssh_cmd = self._build_ssh_command(host, restart_cmd)

        if self.config.dry_run:
            logger.info(f"[ClusterHealing] DRY RUN: Would restart P2P on {host.name}")
            logger.debug(f"  Bootstrap peers: {peers_arg}")
            return True

        try:
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.ssh_timeout_seconds + 10
            )

            if proc.returncode == 0:
                logger.info(f"[ClusterHealing] Restarted P2P on {host.name}")
                return True
            else:
                logger.warning(
                    f"[ClusterHealing] Failed to restart P2P on {host.name}: "
                    f"exit={proc.returncode}"
                )
                return False

        except asyncio.TimeoutError:
            logger.warning(f"[ClusterHealing] SSH timeout restarting P2P on {host.name}")
            return False
        except Exception as e:
            logger.warning(f"[ClusterHealing] Error restarting P2P on {host.name}: {e}")
            return False

    def _build_ssh_command(self, host: HostInfo, remote_cmd: str) -> str:
        """Build SSH command for a host."""
        ssh_opts = (
            "-o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null "
            "-o ConnectTimeout=10 "
            "-o BatchMode=yes"
        )
        if host.ssh_port != 22:
            ssh_opts += f" -p {host.ssh_port}"

        return f"ssh {ssh_opts} {host.ssh_user}@{host.tailscale_ip} '{remote_cmd}'"

    def _is_rate_limited(self) -> bool:
        """Check if we've hit the rate limit for heals.

        Jan 2026: Added to prevent heal storms. Uses a rolling time window
        to ensure we never exceed max_heals_per_window in any rate_limit_window_seconds
        period.

        Returns:
            True if rate limited (should not heal), False if OK to heal
        """
        now = time.time()
        window = self.config.rate_limit_window_seconds
        max_heals = self.config.max_heals_per_window

        # Prune timestamps outside the window
        cutoff = now - window
        self._heal_timestamps = [ts for ts in self._heal_timestamps if ts > cutoff]

        # Check if at limit
        return len(self._heal_timestamps) >= max_heals

    def _record_heal_timestamp(self) -> None:
        """Record a successful heal for rate limiting."""
        self._heal_timestamps.append(time.time())

    def _get_rate_limit_remaining(self) -> int:
        """Get number of heals remaining in current window."""
        now = time.time()
        cutoff = now - self.config.rate_limit_window_seconds
        recent = [ts for ts in self._heal_timestamps if ts > cutoff]
        return max(0, self.config.max_heals_per_window - len(recent))

    async def _heal_node(self, host: HostInfo, bootstrap_peers: list[str]) -> bool:
        """Attempt to heal a single node."""
        # CRITICAL: Double-check we're not healing ourselves
        try:
            from app.config.node_identity import get_node_id_safe
            local_node_id = get_node_id_safe()
        except Exception:
            import socket
            local_node_id = os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())

        if host.name == local_node_id:
            logger.warning(f"[ClusterHealing] BLOCKED self-healing attempt for {host.name}")
            return False

        self._stats_heal_attempts += 1
        logger.info(f"[ClusterHealing] Attempting to heal {host.name}")

        # First check if P2P is already running
        status = await self._check_p2p_status(host)
        if status:
            # P2P is running but not connected to us - might be in another partition
            leader = status.get("leader_id")
            peers = status.get("alive_peers", 0)

            # Jan 23, 2026: Don't restart nodes that have healthy P2P
            # A node is considered healthy if it has:
            # - A leader (any leader) OR
            # - At least 2 alive peers (could be in election)
            # Only restart truly isolated nodes (0-1 peers, no leader)
            if leader or peers >= 2:
                logger.info(
                    f"[ClusterHealing] {host.name} has healthy P2P "
                    f"(leader={leader}, peers={peers}), skipping restart - not isolated"
                )
                return True  # Consider it "healed" - it's in a partition but not dead

            logger.info(
                f"[ClusterHealing] {host.name} has isolated P2P "
                f"(leader={leader}, peers={peers}), restarting with correct bootstrap"
            )

        # Restart P2P with our bootstrap peers
        success = await self._restart_p2p(host, bootstrap_peers)

        if success:
            # Wait for P2P to start
            await asyncio.sleep(self.config.p2p_startup_wait_seconds)

            # Verify it joined
            status = await self._check_p2p_status(host)
            if status:
                self._stats_heal_successes += 1
                self._heal_failures.pop(host.name, None)
                self._heal_next_attempt.pop(host.name, None)

                if self._on_node_joined:
                    self._on_node_joined(host.name)

                if self._emit_event and self.config.emit_events:
                    self._emit_event(
                        "NODE_HEALED",
                        {
                            "node_id": host.name,
                            "tailscale_ip": host.tailscale_ip,
                            "leader_id": status.get("leader_id"),
                            "timestamp": time.time(),
                        },
                    )

                logger.info(f"[ClusterHealing] Successfully healed {host.name}")
                return True

        # Record failure with backoff
        self._record_heal_failure(host.name, "restart_failed")
        return False

    def _record_heal_failure(self, node_id: str, reason: str) -> None:
        """Record a healing failure and apply backoff."""
        self._stats_heal_failures += 1
        failures = self._heal_failures.get(node_id, 0) + 1
        self._heal_failures[node_id] = failures

        # Exponential backoff: 10min, 20min, 40min, max 1 hour
        backoff = self.config.backoff_after_failure_seconds * (2 ** (failures - 1))
        backoff = min(backoff, 3600)  # Cap at 1 hour
        self._heal_next_attempt[node_id] = time.time() + backoff

        logger.warning(
            f"[ClusterHealing] Failed to heal {node_id}: {reason} "
            f"(failures: {failures}, backoff: {backoff:.0f}s)"
        )

        if self._emit_event and self.config.emit_events:
            self._emit_event(
                "NODE_HEAL_FAILED",
                {
                    "node_id": node_id,
                    "reason": reason,
                    "consecutive_failures": failures,
                    "next_attempt_in_seconds": backoff,
                    "timestamp": time.time(),
                },
            )

    async def _run_once(self) -> None:
        """Execute one healing cycle."""
        if not self.config.enabled:
            return

        # Get missing nodes
        missing = self._get_missing_nodes()
        if not missing:
            logger.debug("[ClusterHealing] No missing nodes to heal")
            return

        logger.info(f"[ClusterHealing] Found {len(missing)} missing nodes")

        # Get bootstrap peers
        bootstrap_peers = self._select_bootstrap_peers()
        if not bootstrap_peers:
            logger.warning("[ClusterHealing] No alive peers for bootstrap, skipping")
            return

        # Limit healing attempts per cycle (Phase 7.3: use dynamic rate)
        max_heal = self._get_dynamic_max_heal_per_cycle()
        to_heal = missing[:max_heal]
        logger.info(
            f"[ClusterHealing] Attempting to heal {len(to_heal)}/{len(missing)} nodes "
            f"(max_heal={max_heal}, dynamic={self.config.dynamic_heal_rate})"
        )

        # Heal each node with rate limiting (Jan 2026)
        healed = 0
        rate_limited = 0
        for host in to_heal:
            # Check time-based rate limit before each heal
            if self._is_rate_limited():
                remaining = self._get_rate_limit_remaining()
                logger.info(
                    f"[ClusterHealing] Rate limited, skipping {host.name} "
                    f"({remaining} heals remaining in window)"
                )
                self._stats_rate_limited += 1
                rate_limited += 1
                continue

            success = await self._heal_node(host, bootstrap_peers)
            if success:
                healed += 1
                self._record_heal_timestamp()  # Track for rate limiting

        if healed > 0 or rate_limited > 0:
            logger.info(
                f"[ClusterHealing] Cycle complete: healed={healed}, "
                f"rate_limited={rate_limited}, attempted={len(to_heal)}"
            )

    def get_healing_stats(self) -> dict[str, Any]:
        """Get healing statistics."""
        return {
            "heal_attempts": self._stats_heal_attempts,
            "heal_successes": self._stats_heal_successes,
            "heal_failures": self._stats_heal_failures,
            "rate_limited": self._stats_rate_limited,  # Jan 2026
            "nodes_in_backoff": len(self._heal_next_attempt),
            "success_rate": (
                self._stats_heal_successes / max(1, self._stats_heal_attempts)
            ),
            # Jan 2026: Rate limit status
            "rate_limit_remaining": self._get_rate_limit_remaining(),
            "rate_limit_window_seconds": self.config.rate_limit_window_seconds,
            "max_heals_per_window": self.config.max_heals_per_window,
        }

    def reset_backoff(self, node_id: str) -> None:
        """Reset backoff for a specific node."""
        self._heal_failures.pop(node_id, None)
        self._heal_next_attempt.pop(node_id, None)
        logger.debug(f"[ClusterHealing] Reset backoff for {node_id}")

    def reset_all_backoffs(self) -> None:
        """Reset backoff for all nodes."""
        self._heal_failures.clear()
        self._heal_next_attempt.clear()
        logger.info("[ClusterHealing] Reset all node backoffs")

    def health_check(self) -> Any:
        """Check cluster healing loop health for DaemonManager integration.

        Jan 2026: Added specialized health check for cluster healing monitoring.

        Returns:
            HealthCheckResult with healing-specific metrics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"ClusterHealingLoop {'running' if self._running else 'stopped'}",
                "details": self.get_healing_stats(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="ClusterHealingLoop is stopped",
            )

        if not self.config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.IDLE,
                message="ClusterHealingLoop disabled",
            )

        # Check base loop health first
        base_health = super().health_check()
        if not base_health.healthy:
            return base_health

        # Check for high failure rate
        success_rate = (
            self._stats_heal_successes / max(1, self._stats_heal_attempts)
        )
        if self._stats_heal_attempts > 5 and success_rate < 0.3:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Low healing success rate: {success_rate:.1%}",
                details={
                    "heal_attempts": self._stats_heal_attempts,
                    "heal_successes": self._stats_heal_successes,
                    "heal_failures": self._stats_heal_failures,
                    "success_rate": f"{success_rate:.1%}",
                    "nodes_in_backoff": len(self._heal_next_attempt),
                },
            )

        # Check if rate limited
        rate_limit_remaining = self._get_rate_limit_remaining()
        if rate_limit_remaining == 0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message="ClusterHealingLoop rate limited",
                details={
                    "rate_limited": True,
                    "max_heals_per_window": self.config.max_heals_per_window,
                    "window_seconds": self.config.rate_limit_window_seconds,
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="ClusterHealingLoop operational",
            details={
                "heal_attempts": self._stats_heal_attempts,
                "heal_successes": self._stats_heal_successes,
                "heal_failures": self._stats_heal_failures,
                "success_rate": f"{success_rate:.1%}",
                "nodes_in_backoff": len(self._heal_next_attempt),
                "rate_limit_remaining": rate_limit_remaining,
                "dry_run": self.config.dry_run,
                "total_runs": self.stats.total_runs,
            },
        )


__all__ = [
    "ClusterHealingConfig",
    "ClusterHealingLoop",
    "HostInfo",
]

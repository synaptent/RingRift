"""Remote P2P Recovery Loop for P2P Orchestrator.

December 2025: Automatically starts P2P on cluster nodes that should be running it.

Problem: Cluster nodes may have P2P stopped due to:
1. Node reboot
2. Process crash
3. Manual intervention
4. OOM kills

This caused 0/11 Lambda GH200 nodes to be in the mesh even though they were
reachable via Tailscale.

Solution: Periodically check which configured nodes are NOT in the P2P mesh
and use paramiko to SSH in and start P2P on them.

Usage:
    from scripts.p2p.loops import RemoteP2PRecoveryLoop, RemoteP2PRecoveryConfig

    recovery_loop = RemoteP2PRecoveryLoop(
        get_alive_peer_ids=lambda: orchestrator.get_alive_peer_ids(),
        emit_event=orchestrator._emit_event,
    )
    await recovery_loop.run_forever()

Events:
    REMOTE_P2P_STARTED: Emitted when P2P is started on a remote node
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import BaseLoop

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RemoteP2PRecoveryConfig:
    """Configuration for remote P2P recovery."""

    # Interval between recovery cycles (seconds) - default 5 minutes
    check_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_REMOTE_P2P_RECOVERY_INTERVAL", "300")
        )
    )

    # Maximum nodes to recover per cycle (prevent thundering herd)
    max_nodes_per_cycle: int = 5

    # SSH timeout per node (seconds)
    ssh_timeout_seconds: float = 30.0

    # Whether the loop is enabled
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_REMOTE_P2P_RECOVERY_ENABLED", "1"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Dry run mode - log what would be done without actually doing it
    dry_run: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_REMOTE_P2P_RECOVERY_DRY_RUN", "0"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # SSH key path
    ssh_key_path: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_SSH_KEY", os.path.expanduser("~/.ssh/id_cluster")
        )
    )

    # Minimum time since last attempt for a node before retrying (seconds)
    retry_cooldown_seconds: float = 600.0  # 10 minutes

    # Whether to emit events on recovery
    emit_events: bool = True


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class RemoteP2PRecoveryStats:
    """Statistics for remote P2P recovery operations."""

    nodes_recovered: int = 0
    nodes_failed: int = 0
    last_recovery_time: float = 0.0
    cycles_run: int = 0
    nodes_skipped_cooldown: int = 0

    def to_dict(self) -> dict:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "nodes_recovered": self.nodes_recovered,
            "nodes_failed": self.nodes_failed,
            "last_recovery_time": self.last_recovery_time,
            "cycles_run": self.cycles_run,
            "nodes_skipped_cooldown": self.nodes_skipped_cooldown,
        }


# =============================================================================
# Recovery Loop
# =============================================================================


class RemoteP2PRecoveryLoop(BaseLoop):
    """Background loop that automatically starts P2P on nodes that aren't running it.

    Key features:
    - Uses paramiko for SSH connections (works through NAT/firewalls)
    - Prefers Tailscale IPs for connectivity
    - Respects cooldown periods to avoid hammering failed nodes
    - Limits recoveries per cycle to prevent thundering herd
    - Emits REMOTE_P2P_STARTED event for monitoring
    """

    def __init__(
        self,
        get_alive_peer_ids: Callable[[], list[str]],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        config: RemoteP2PRecoveryConfig | None = None,
    ):
        """Initialize remote P2P recovery loop.

        Args:
            get_alive_peer_ids: Callback returning list of alive peer node IDs
            emit_event: Optional callback to emit events (event_name, event_data)
            config: Recovery configuration
        """
        self.config = config or RemoteP2PRecoveryConfig()
        super().__init__(
            name="remote_p2p_recovery",
            interval=self.config.check_interval_seconds,
            enabled=self.config.enabled,
        )

        # Callbacks
        self._get_alive_peer_ids = get_alive_peer_ids
        self._emit_event = emit_event

        # Statistics
        self._stats = RemoteP2PRecoveryStats()

        # Track last attempt time per node to implement cooldown
        self._last_attempt: dict[str, float] = {}

        # Paramiko client (lazy loaded)
        self._paramiko: Any = None

    def _get_paramiko(self) -> Any:
        """Lazy-load paramiko module."""
        if self._paramiko is None:
            try:
                import paramiko
                self._paramiko = paramiko
            except ImportError:
                logger.warning("[RemoteP2PRecovery] paramiko not installed, SSH recovery disabled")
                return None
        return self._paramiko

    async def _run_once(self) -> None:
        """Execute one recovery cycle."""
        if not self.config.enabled:
            return

        paramiko = self._get_paramiko()
        if paramiko is None:
            return

        now = time.time()
        self._stats.cycles_run += 1

        # Get current alive peers
        alive_peer_ids = set(self._get_alive_peer_ids())
        if not alive_peer_ids and not self.config.dry_run:
            logger.debug("[RemoteP2PRecovery] No alive peers yet, skipping")
            return

        # Get configured nodes that should be running P2P
        configured_nodes = self._get_configured_nodes()
        if not configured_nodes:
            logger.debug("[RemoteP2PRecovery] No configured nodes found")
            return

        # Find nodes that should be running P2P but aren't
        nodes_to_recover = []
        for node_id, node_info in configured_nodes.items():
            # Skip if already alive
            if node_id in alive_peer_ids:
                continue

            # Skip if retired or not ready
            if node_info.get("status") not in ("ready", "active"):
                continue

            # Skip if no SSH access configured
            if not node_info.get("tailscale_ip") and not node_info.get("ssh_host"):
                continue

            # Skip if in cooldown
            last_attempt = self._last_attempt.get(node_id, 0)
            if now - last_attempt < self.config.retry_cooldown_seconds:
                self._stats.nodes_skipped_cooldown += 1
                continue

            nodes_to_recover.append((node_id, node_info))

        if not nodes_to_recover:
            logger.debug(
                f"[RemoteP2PRecovery] All configured nodes are alive "
                f"({len(alive_peer_ids)} alive, {len(configured_nodes)} configured)"
            )
            return

        # Limit recoveries per cycle
        nodes_this_cycle = nodes_to_recover[: self.config.max_nodes_per_cycle]

        if self.config.dry_run:
            logger.info(
                f"[RemoteP2PRecovery] DRY RUN: Would recover {len(nodes_this_cycle)} nodes: "
                f"{[n[0] for n in nodes_this_cycle]}"
            )
            return

        logger.info(
            f"[RemoteP2PRecovery] Recovering {len(nodes_this_cycle)}/{len(nodes_to_recover)} nodes"
        )

        # Recover nodes in parallel (up to max_nodes_per_cycle concurrent)
        async def recover_with_semaphore(node_id: str, node_info: dict) -> bool:
            return await self._recover_node(node_id, node_info, paramiko)

        tasks = [
            recover_with_semaphore(node_id, node_info)
            for node_id, node_info in nodes_this_cycle
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes/failures
        for (node_id, _), result in zip(nodes_this_cycle, results):
            self._last_attempt[node_id] = now
            if isinstance(result, Exception):
                logger.warning(f"[RemoteP2PRecovery] Error recovering {node_id}: {result}")
                self._stats.nodes_failed += 1
            elif result:
                self._stats.nodes_recovered += 1
            else:
                self._stats.nodes_failed += 1

        self._stats.last_recovery_time = now

    def _get_configured_nodes(self) -> dict[str, dict[str, Any]]:
        """Get configured nodes from distributed_hosts.yaml."""
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config",
                "distributed_hosts.yaml",
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("hosts", {})
        except Exception as e:
            logger.warning(f"[RemoteP2PRecovery] Failed to load config: {e}")
            return {}

    async def _recover_node(
        self, node_id: str, node_info: dict, paramiko: Any
    ) -> bool:
        """Attempt to start P2P on a remote node.

        Args:
            node_id: Node identifier
            node_info: Node configuration from distributed_hosts.yaml
            paramiko: paramiko module

        Returns:
            True if P2P was started successfully
        """
        # Prefer Tailscale IP, fall back to SSH host
        host = node_info.get("tailscale_ip") or node_info.get("ssh_host")
        user = node_info.get("user", "ubuntu")
        port = node_info.get("ssh_port", 22)

        if not host:
            logger.warning(f"[RemoteP2PRecovery] No host configured for {node_id}")
            return False

        logger.info(f"[RemoteP2PRecovery] Starting P2P on {node_id} ({host})")

        try:
            # Run SSH operations in thread pool to not block event loop
            result = await asyncio.to_thread(
                self._ssh_start_p2p, node_id, host, port, user, paramiko
            )
            return result
        except Exception as e:
            logger.warning(f"[RemoteP2PRecovery] SSH failed for {node_id}: {e}")
            return False

    def _ssh_start_p2p(
        self, node_id: str, host: str, port: int, user: str, paramiko: Any
    ) -> bool:
        """Execute SSH commands to start P2P on a node (runs in thread).

        Args:
            node_id: Node identifier
            host: SSH host/IP
            port: SSH port
            user: SSH user
            paramiko: paramiko module

        Returns:
            True if P2P was started successfully
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                host,
                port=port,
                username=user,
                key_filename=self.config.ssh_key_path,
                timeout=self.config.ssh_timeout_seconds,
                banner_timeout=self.config.ssh_timeout_seconds,
            )

            # First kill any existing P2P process
            client.exec_command("pkill -f p2p_orchestrator || true")
            time.sleep(1)

            # Pull latest code and start P2P
            # Use python3 explicitly since 'python' may not exist on all nodes
            cmd = f"""cd ~/ringrift/ai-service && \
git pull origin main 2>/dev/null || true && \
PYTHONPATH=. nohup python3 scripts/p2p_orchestrator.py --node-id {node_id} --port 8770 > logs/p2p.log 2>&1 &"""

            stdin, stdout, stderr = client.exec_command(cmd)
            stdout.channel.recv_exit_status()  # Wait for completion

            # Wait for process to start
            time.sleep(3)

            # Verify it started
            _, stdout2, _ = client.exec_command("pgrep -f p2p_orchestrator")
            pid = stdout2.read().decode().strip()

            client.close()

            if pid:
                logger.info(f"[RemoteP2PRecovery] Started P2P on {node_id} (PID: {pid})")

                # Emit event
                if self._emit_event and self.config.emit_events:
                    self._emit_event(
                        "REMOTE_P2P_STARTED",
                        {
                            "node_id": node_id,
                            "host": host,
                            "pid": pid,
                            "timestamp": time.time(),
                        },
                    )
                return True
            else:
                logger.warning(f"[RemoteP2PRecovery] P2P not running after start on {node_id}")
                return False

        except Exception as e:
            logger.warning(f"[RemoteP2PRecovery] SSH error for {node_id}: {e}")
            return False
        finally:
            try:
                client.close()
            except Exception:
                pass

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        return {
            **self._stats.to_dict(),
            "config": {
                "interval_seconds": self.config.check_interval_seconds,
                "max_per_cycle": self.config.max_nodes_per_cycle,
                "retry_cooldown_seconds": self.config.retry_cooldown_seconds,
                "enabled": self.config.enabled,
                "dry_run": self.config.dry_run,
            },
        }

    def reset_stats(self) -> None:
        """Reset recovery statistics."""
        self._stats = RemoteP2PRecoveryStats()
        self._last_attempt.clear()
        logger.info("[RemoteP2PRecovery] Statistics reset")

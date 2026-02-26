"""ConnectivityRecoveryCoordinator - Unified event-driven connectivity recovery.

This coordinator handles all connectivity-related events and triggers appropriate
recovery actions. It bridges the TailscaleHealthDaemon, NodeAvailabilityDaemon,
and P2P orchestrator to provide unified connectivity management.

Key Events Handled:
- TAILSCALE_DISCONNECTED: Node's Tailscale went offline
- TAILSCALE_RECOVERED: Node's Tailscale came back online
- P2P_NODE_DEAD: P2P peer marked as dead
- HOST_OFFLINE: Host marked offline in P2P mesh
- HOST_ONLINE: Host came back online

Recovery Actions:
- SSH-based Tailscale restart for masked failures
- P2P mesh updates when nodes recover
- Cluster health recalculation
- Alert generation for persistent failures

December 2025 - Created as part of P2P stability improvements.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.coordination.contracts import HealthCheckResult
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_router import get_event_payload
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for connectivity issues."""

    TAILSCALE_UP = "tailscale_up"  # Try tailscale up command
    TAILSCALE_RESTART = "tailscale_restart"  # Restart tailscaled daemon
    SSH_RECOVERY = "ssh_recovery"  # SSH into node and recover Tailscale
    P2P_RECONNECT = "p2p_reconnect"  # Force P2P reconnection
    NODE_RESTART = "node_restart"  # Restart the entire node (cloud provider)
    ALERT = "alert"  # Generate alert for human intervention


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    node_name: str
    strategy: RecoveryStrategy
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class NodeConnectivityState:
    """Connectivity state for a single node."""

    node_name: str
    tailscale_connected: bool = False
    p2p_connected: bool = False
    last_seen: float = 0.0
    recovery_attempts: int = 0
    last_recovery_time: float = 0.0
    consecutive_failures: int = 0
    pending_recovery: bool = False


@dataclass
class RecoveryConfig:
    """Configuration for connectivity recovery."""

    # Recovery timing
    recovery_cooldown_seconds: float = 300.0  # 5 minutes between recovery attempts
    max_recovery_attempts: int = 5
    escalation_threshold: int = 3  # Escalate after N failed attempts

    # SSH recovery settings
    ssh_recovery_enabled: bool = True
    ssh_timeout_seconds: float = 30.0

    # Alerting
    alert_after_failures: int = 5
    slack_webhook: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RecoveryConfig":
        """Load config from environment variables."""
        return cls(
            recovery_cooldown_seconds=float(
                os.environ.get("RINGRIFT_RECOVERY_COOLDOWN", "300")
            ),
            max_recovery_attempts=int(
                os.environ.get("RINGRIFT_MAX_RECOVERY_ATTEMPTS", "5")
            ),
            ssh_recovery_enabled=os.environ.get(
                "RINGRIFT_SSH_RECOVERY_ENABLED", "true"
            ).lower() == "true",
            slack_webhook=os.environ.get("RINGRIFT_SLACK_WEBHOOK"),
        )


class ConnectivityRecoveryCoordinator(HandlerBase):
    """Coordinates recovery across all connectivity issues.

    Subscribes to connectivity-related events and triggers appropriate
    recovery actions based on the type and severity of the issue.

    Usage:
        coordinator = ConnectivityRecoveryCoordinator.get_instance()
        await coordinator.start()

    The coordinator:
    1. Tracks connectivity state for all nodes
    2. Handles Tailscale disconnection events
    3. Handles P2P node failure events
    4. Escalates persistent failures
    5. Generates alerts for human intervention
    """

    def __init__(self, config: Optional[RecoveryConfig] = None):
        """Initialize the coordinator."""
        self._config = config or RecoveryConfig.from_env()
        super().__init__(
            name="connectivity_recovery",
            config=self._config,
            cycle_interval=60.0,  # Check every minute
        )
        self._node_states: dict[str, NodeConnectivityState] = {}
        self._recovery_history: list[RecoveryAttempt] = []
        self._pending_recoveries: set[str] = set()

    async def _run_cycle(self) -> None:
        """Main cycle - check for nodes needing recovery."""
        try:
            # Check for nodes that need periodic recovery checks
            await self._check_pending_recoveries()
            self._stats.cycles_completed += 1
            self._stats.last_activity = time.time()
        except Exception as e:
            self._stats.errors_count += 1
            self._stats.last_error = str(e)
            logger.error(f"Recovery cycle error: {e}")

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to connectivity events."""
        return {
            "TAILSCALE_DISCONNECTED": self._on_tailscale_disconnected,
            "TAILSCALE_RECOVERED": self._on_tailscale_recovered,
            "TAILSCALE_RECOVERY_FAILED": self._on_tailscale_recovery_failed,
            "P2P_NODE_DEAD": self._on_p2p_node_dead,
            "HOST_OFFLINE": self._on_host_offline,
            "HOST_ONLINE": self._on_host_online,
        }

    def health_check(self) -> HealthCheckResult:
        """Return health check result."""
        pending_count = len(self._pending_recoveries)
        recent_failures = sum(
            1 for s in self._node_states.values()
            if s.consecutive_failures > 0
        )

        is_healthy = pending_count < 5 and recent_failures < 10

        return HealthCheckResult(
            healthy=is_healthy,
            message=f"{len(self._node_states)} nodes tracked, {pending_count} pending recoveries",
            details={
                "nodes_tracked": len(self._node_states),
                "pending_recoveries": pending_count,
                "recent_failures": recent_failures,
                "recovery_history_size": len(self._recovery_history),
                "cycles_completed": self._stats.cycles_completed,
            },
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_tailscale_disconnected(self, event: Any) -> None:
        """Handle Tailscale disconnection event."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        node_name = payload.get("node_name") or payload.get("hostname")
        if not node_name:
            logger.warning("TAILSCALE_DISCONNECTED event missing node_name")
            return

        logger.info(f"Tailscale disconnected: {node_name}")

        # Update state
        state = self._get_or_create_state(node_name)
        state.tailscale_connected = False
        state.consecutive_failures += 1

        # Check if local TailscaleHealthDaemon should handle this
        if payload.get("local_daemon_handling"):
            logger.debug(f"Local daemon handling recovery for {node_name}")
            return

        # Queue for SSH-based recovery if enabled
        if self._config.ssh_recovery_enabled and node_name not in self._pending_recoveries:
            self._pending_recoveries.add(node_name)
            asyncio.create_task(self._attempt_ssh_recovery(node_name))

    async def _on_tailscale_recovered(self, event: Any) -> None:
        """Handle Tailscale recovery event."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        node_name = payload.get("node_name") or payload.get("hostname")
        if not node_name:
            return

        logger.info(f"Tailscale recovered: {node_name}")

        state = self._get_or_create_state(node_name)
        state.tailscale_connected = True
        state.consecutive_failures = 0
        state.last_seen = time.time()
        self._pending_recoveries.discard(node_name)

    async def _on_tailscale_recovery_failed(self, event: Any) -> None:
        """Handle Tailscale recovery failure event."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        node_name = payload.get("node_name") or payload.get("hostname")
        if not node_name:
            return

        logger.warning(f"Tailscale recovery failed: {node_name}")

        state = self._get_or_create_state(node_name)
        state.recovery_attempts += 1
        state.last_recovery_time = time.time()

        # Check if we should escalate
        if state.recovery_attempts >= self._config.escalation_threshold:
            await self._escalate_recovery(node_name, state)

    async def _on_p2p_node_dead(self, event: Any) -> None:
        """Handle P2P node dead event."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        node_name = payload.get("node_id") or payload.get("peer_id")
        if not node_name:
            return

        logger.info(f"P2P node dead: {node_name}")

        state = self._get_or_create_state(node_name)
        state.p2p_connected = False
        state.consecutive_failures += 1

    async def _on_host_offline(self, event: Any) -> None:
        """Handle host offline event."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        node_name = payload.get("node_id") or payload.get("hostname")
        if not node_name:
            return

        logger.info(f"Host offline: {node_name}")

        state = self._get_or_create_state(node_name)
        state.tailscale_connected = False
        state.p2p_connected = False
        state.consecutive_failures += 1

        # Queue for recovery
        if node_name not in self._pending_recoveries:
            self._pending_recoveries.add(node_name)

    async def _on_host_online(self, event: Any) -> None:
        """Handle host online event."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        node_name = payload.get("node_id") or payload.get("hostname")
        if not node_name:
            return

        logger.info(f"Host online: {node_name}")

        state = self._get_or_create_state(node_name)
        state.tailscale_connected = True
        state.p2p_connected = True
        state.consecutive_failures = 0
        state.last_seen = time.time()
        self._pending_recoveries.discard(node_name)

    # =========================================================================
    # Recovery Logic
    # =========================================================================

    async def _check_pending_recoveries(self) -> None:
        """Check and process pending recoveries."""
        for node_name in list(self._pending_recoveries):
            state = self._node_states.get(node_name)
            if not state:
                self._pending_recoveries.discard(node_name)
                continue

            # Check cooldown
            if state.last_recovery_time:
                elapsed = time.time() - state.last_recovery_time
                if elapsed < self._config.recovery_cooldown_seconds:
                    continue

            # Check max attempts
            if state.recovery_attempts >= self._config.max_recovery_attempts:
                await self._escalate_recovery(node_name, state)
                self._pending_recoveries.discard(node_name)
                continue

            # Attempt recovery
            await self._attempt_ssh_recovery(node_name)

    async def _attempt_ssh_recovery(self, node_name: str) -> bool:
        """Attempt SSH-based Tailscale recovery.

        Args:
            node_name: Name of the node to recover

        Returns:
            True if recovery succeeded
        """
        state = self._get_or_create_state(node_name)
        state.pending_recovery = True
        state.last_recovery_time = time.time()
        state.recovery_attempts += 1

        logger.info(
            f"Attempting SSH recovery for {node_name} "
            f"(attempt {state.recovery_attempts}/{self._config.max_recovery_attempts})"
        )

        try:
            # Get SSH details from config
            ssh_info = await self._get_ssh_info(node_name)
            if not ssh_info:
                logger.warning(f"No SSH info for {node_name}, cannot recover")
                return False

            # Build recovery command
            recovery_cmd = self._build_recovery_command(ssh_info)

            # Execute via SSH
            success = await self._execute_ssh_command(
                ssh_info["host"],
                ssh_info.get("port", 22),
                ssh_info.get("user", "root"),
                ssh_info.get("key"),
                recovery_cmd,
            )

            # Record attempt
            attempt = RecoveryAttempt(
                node_name=node_name,
                strategy=RecoveryStrategy.SSH_RECOVERY,
                timestamp=time.time(),
                success=success,
            )
            self._recovery_history.append(attempt)

            if success:
                state.consecutive_failures = 0
                self._pending_recoveries.discard(node_name)
                logger.info(f"SSH recovery succeeded for {node_name}")

                # Emit recovery event
                safe_emit_event(
                    "TAILSCALE_RECOVERED",
                    {"node_name": node_name, "recovery_method": "ssh"},
                    context="connectivity_recovery",
                )

            return success

        except Exception as e:
            logger.error(f"SSH recovery failed for {node_name}: {e}")
            return False

        finally:
            state.pending_recovery = False

    def _build_recovery_command(self, ssh_info: dict) -> str:
        """Build the Tailscale recovery command.

        Args:
            ssh_info: SSH connection info including:
                - is_container: Whether node is a container (uses userspace networking)
                - hostname: Node hostname for Tailscale identification

        Returns:
            Shell command to recover Tailscale connectivity.
        """
        is_container = ssh_info.get("is_container", False)
        hostname = ssh_info.get("hostname", "unknown")

        # Get authkey from environment for reauthorization
        authkey = os.environ.get("TAILSCALE_AUTH_KEY", "")
        authkey_arg = f"--authkey={authkey}" if authkey else ""

        if is_container:
            # Container: Use userspace networking (Vast.ai, RunPod, etc.)
            return f"""
pkill -9 tailscaled 2>/dev/null || true
sleep 2
mkdir -p /var/lib/tailscale /var/run/tailscale
nohup tailscaled --tun=userspace-networking --statedir=/var/lib/tailscale > /tmp/tailscaled.log 2>&1 &
sleep 5
tailscale up {authkey_arg} --accept-routes --hostname='{hostname}'
tailscale ip -4
"""
        else:
            # Regular host: Use systemctl (Lambda, Nebius, etc.)
            return f"""
systemctl restart tailscaled 2>/dev/null || {{
    pkill -9 tailscaled
    sleep 2
    tailscaled --state=/var/lib/tailscale/tailscaled.state &
    sleep 5
}}
tailscale up {authkey_arg} --accept-routes --hostname='{hostname}'
tailscale ip -4
"""

    async def _execute_ssh_command(
        self,
        host: str,
        port: int,
        user: str,
        key_path: Optional[str],
        command: str,
    ) -> bool:
        """Execute command via SSH."""
        try:
            ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
            if port != 22:
                ssh_cmd.extend(["-p", str(port)])
            if key_path:
                ssh_cmd.extend(["-i", os.path.expanduser(key_path)])
            ssh_cmd.extend([f"{user}@{host}", command])

            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._config.ssh_timeout_seconds,
            )

            if proc.returncode == 0:
                # Check if Tailscale IP is in output
                output = stdout.decode()
                if "100." in output:  # Tailscale IPs start with 100.
                    return True

            logger.debug(f"SSH command failed: {stderr.decode()}")
            return False

        except asyncio.TimeoutError:
            logger.warning(f"SSH command timed out for {host}")
            return False
        except Exception as e:
            logger.error(f"SSH command error for {host}: {e}")
            return False

    async def _get_ssh_info(self, node_name: str) -> Optional[dict]:
        """Get SSH connection info for a node from config."""
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            # Feb 2026: get_cluster_nodes() returns dict, not list
            node_iter = nodes.values() if isinstance(nodes, dict) else nodes
            for node in node_iter:
                node_id = node.name if hasattr(node, 'name') else str(node)
                if node_id == node_name:
                    return {
                        "host": node.ssh_host or node.tailscale_ip,
                        "port": node.ssh_port,
                        "user": node.ssh_user or "root",
                        "key": node.ssh_key,
                        "is_container": getattr(node, "is_container", False),
                        "hostname": node_name,  # For Tailscale --hostname arg
                    }
            return None
        except ImportError:
            logger.warning("cluster_config not available")
            return None

    async def _escalate_recovery(
        self,
        node_name: str,
        state: NodeConnectivityState,
    ) -> None:
        """Escalate recovery - generate alerts."""
        logger.warning(
            f"Escalating recovery for {node_name} after {state.recovery_attempts} attempts"
        )

        # Emit escalation event
        safe_emit_event(
            "CONNECTIVITY_RECOVERY_ESCALATED",
            {
                "node_name": node_name,
                "recovery_attempts": state.recovery_attempts,
                "consecutive_failures": state.consecutive_failures,
            },
            context="connectivity_recovery",
        )

        # Send Slack alert if configured
        if self._config.slack_webhook:
            await self._send_slack_alert(node_name, state)

    async def _send_slack_alert(
        self,
        node_name: str,
        state: NodeConnectivityState,
    ) -> None:
        """Send alert to Slack."""
        try:
            import aiohttp

            message = {
                "text": f":warning: Connectivity recovery failed for *{node_name}*\n"
                        f"Attempts: {state.recovery_attempts}\n"
                        f"Failures: {state.consecutive_failures}\n"
                        f"Manual intervention required."
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.slack_webhook,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Slack alert failed: {resp.status}")

        except ImportError:
            logger.debug("aiohttp not available, skipping Slack alert")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    # =========================================================================
    # State Management
    # =========================================================================

    def _get_or_create_state(self, node_name: str) -> NodeConnectivityState:
        """Get or create state for a node."""
        if node_name not in self._node_states:
            self._node_states[node_name] = NodeConnectivityState(node_name=node_name)
        return self._node_states[node_name]

    def get_node_state(self, node_name: str) -> Optional[NodeConnectivityState]:
        """Get connectivity state for a node."""
        return self._node_states.get(node_name)

    def get_all_states(self) -> dict[str, NodeConnectivityState]:
        """Get all node states."""
        return dict(self._node_states)

    def get_disconnected_nodes(self) -> list[str]:
        """Get list of disconnected nodes."""
        return [
            name for name, state in self._node_states.items()
            if not state.tailscale_connected or not state.p2p_connected
        ]

    def get_recovery_history(self, limit: int = 100) -> list[RecoveryAttempt]:
        """Get recent recovery history."""
        return self._recovery_history[-limit:]


# =============================================================================
# Factory Functions
# =============================================================================

def get_connectivity_recovery_coordinator(
    config: Optional[RecoveryConfig] = None,
) -> ConnectivityRecoveryCoordinator:
    """Get or create the ConnectivityRecoveryCoordinator singleton."""
    return ConnectivityRecoveryCoordinator.get_instance()  # type: ignore


def create_connectivity_recovery_coordinator(
    config: Optional[RecoveryConfig] = None,
) -> ConnectivityRecoveryCoordinator:
    """Create a new ConnectivityRecoveryCoordinator instance (for testing)."""
    return ConnectivityRecoveryCoordinator(config=config)

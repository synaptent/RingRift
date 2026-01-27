"""Escalating node recovery engine.

This module implements automated recovery for failed cluster nodes with
escalating strategies:
1. RESTART_P2P - Soft restart of P2P orchestrator
2. RESTART_TAILSCALE - Network reset via Tailscale restart
3. REBOOT_INSTANCE - Provider-level instance reboot
4. RECREATE_INSTANCE - Destroy and recreate instance

Created: Dec 28, 2025
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from .node_monitor import NodeHealthResult

if TYPE_CHECKING:
    from app.config.cluster_config import ClusterNode

logger = logging.getLogger(__name__)


# December 2025: Import from canonical source (renamed to SystemRecoveryAction)
# Backward-compatible alias RecoveryAction retained for existing code
from app.coordination.enums import SystemRecoveryAction

# Backward-compatible alias (deprecated, remove Q2 2026)
RecoveryAction = SystemRecoveryAction


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    node_id: str
    action: RecoveryAction
    success: bool
    duration_seconds: float
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for event emission."""
        return {
            "node_id": self.node_id,
            "action": self.action.name,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class RecoveryState:
    """Tracks recovery state for a node."""
    current_action_index: int = 0
    attempts: dict[RecoveryAction, int] = field(default_factory=dict)
    last_attempt: datetime | None = None
    last_success: datetime | None = None
    cooldown_until: datetime | None = None


@dataclass
class RecoveryEngineConfig:
    """Configuration for RecoveryEngine.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.
    """
    check_interval_seconds: int = 60
    max_attempts_per_action: int = 3
    backoff_base_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    cooldown_after_success_seconds: float = 300.0
    cooldown_after_exhausted_seconds: float = 3600.0
    enabled_actions: list[RecoveryAction] = field(
        default_factory=lambda: [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.REBOOT_INSTANCE,
            RecoveryAction.RECREATE_INSTANCE,
        ]
    )


class RecoveryEngine(HandlerBase):
    """Escalating node recovery engine.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking
    - Inherits safe event emission from HandlerBase (via SafeEventEmitterMixin)

    Subscribes to NODE_UNHEALTHY events and attempts recovery using
    escalating strategies. Tracks attempts per-node to avoid repeated
    failed recovery attempts.

    Example:
        engine = RecoveryEngine()
        await engine.start()
        # Will listen for NODE_UNHEALTHY events and attempt recovery
    """

    # For SafeEventEmitterMixin (inherited via HandlerBase)
    _event_source = "RecoveryEngine"

    # Escalation order
    ESCALATION_ORDER = [
        RecoveryAction.RESTART_P2P,
        RecoveryAction.RESTART_TAILSCALE,
        RecoveryAction.REBOOT_INSTANCE,
        RecoveryAction.RECREATE_INSTANCE,
    ]

    def __init__(self, config: RecoveryEngineConfig | None = None):
        self._daemon_config = config or RecoveryEngineConfig()

        super().__init__(
            name="RecoveryEngine",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._recovery_states: dict[str, RecoveryState] = {}
        self._recovery_queue: asyncio.Queue[tuple[str, NodeHealthResult]] = asyncio.Queue()
        self._recovery_history: list[RecoveryResult] = []

    @property
    def config(self) -> RecoveryEngineConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict:
        """Subscribe to recovery-related events."""
        return {
            "NODE_UNHEALTHY": self._on_node_unhealthy,
            "RECOVERY_INITIATED": self._on_recovery_initiated,
            "NODE_RECOVERED": self._on_node_recovered,
        }

    async def _on_node_unhealthy(self, event: dict) -> None:
        """Handle NODE_UNHEALTHY event from NodeMonitor.

        This is the primary trigger for recovery - when NodeMonitor detects
        an unhealthy node, queue it for escalating recovery.
        """
        payload = event.get("payload", event)
        node_id = payload.get("node_id")
        layer = payload.get("layer", "unknown")
        error = payload.get("error", "")

        if node_id:
            # Create a minimal health result for the recovery queue
            # Convert layer string to HealthCheckLayer enum
            from .node_monitor import HealthCheckLayer

            try:
                layer_enum = HealthCheckLayer(layer) if layer else HealthCheckLayer.P2P
            except ValueError:
                layer_enum = HealthCheckLayer.P2P

            health_result = NodeHealthResult(
                node_id=node_id,
                layer=layer_enum,
                healthy=False,
                latency_ms=0.0,
                error=error or None,
            )
            await self._recovery_queue.put((node_id, health_result))
            logger.info(f"RecoveryEngine: Queued recovery for unhealthy node {node_id} (layer: {layer})")

    async def _on_recovery_initiated(self, event: dict) -> None:
        """Handle RECOVERY_INITIATED event."""
        payload = event.get("payload", event)
        node_id = payload.get("node_id")
        health_result = payload.get("health_result")

        if node_id:
            await self._recovery_queue.put((node_id, health_result))
            logger.info(f"RecoveryEngine: Queued recovery for {node_id}")

    async def _on_node_recovered(self, event: dict) -> None:
        """Handle NODE_RECOVERED event."""
        payload = event.get("payload", event)
        node_id = payload.get("node_id")

        if node_id and node_id in self._recovery_states:
            # Reset recovery state on natural recovery
            self._recovery_states[node_id] = RecoveryState(
                last_success=datetime.now()
            )
            logger.info(f"RecoveryEngine: Reset state for recovered node {node_id}")

    async def _run_cycle(self) -> None:
        """Process recovery queue."""
        # Process up to 5 items per cycle
        for _ in range(5):
            try:
                node_id, health_result = self._recovery_queue.get_nowait()
                await self._attempt_recovery(node_id, health_result)
            except asyncio.QueueEmpty:
                break

    async def _attempt_recovery(
        self,
        node_id: str,
        health_result: dict | None,
    ) -> RecoveryResult | None:
        """Attempt recovery for a node using escalating strategies."""
        # Get or create recovery state
        if node_id not in self._recovery_states:
            self._recovery_states[node_id] = RecoveryState()
        state = self._recovery_states[node_id]

        # Check cooldown
        if state.cooldown_until and datetime.now() < state.cooldown_until:
            remaining = (state.cooldown_until - datetime.now()).total_seconds()
            logger.debug(f"Node {node_id} in cooldown for {remaining:.0f}s more")
            return None

        # Get node info
        node = await self._get_node(node_id)
        if not node:
            logger.warning(f"Cannot recover unknown node: {node_id}")
            return None

        # Iterate through escalation order
        for action in self.ESCALATION_ORDER[state.current_action_index:]:
            if action not in self.config.enabled_actions:
                continue

            # Check attempt count
            attempts = state.attempts.get(action, 0)
            if attempts >= self.config.max_attempts_per_action:
                state.current_action_index += 1
                continue

            # Attempt this action
            state.attempts[action] = attempts + 1
            state.last_attempt = datetime.now()

            logger.info(
                f"Attempting {action.name} for {node_id} "
                f"(attempt {attempts + 1}/{self.config.max_attempts_per_action})"
            )

            result = await self._execute_recovery(node, action)
            self._recovery_history.append(result)

            if result.success:
                # Recovery succeeded
                await self._emit_recovery_success(result)
                state.current_action_index = 0
                state.attempts = {}
                state.last_success = datetime.now()
                state.cooldown_until = datetime.now() + \
                    self._get_cooldown(success=True)
                return result
            else:
                # Recovery failed, apply backoff
                await self._emit_recovery_failed(result)
                backoff = self._get_backoff(action, attempts + 1)
                await asyncio.sleep(backoff)

        # All actions exhausted
        logger.error(f"All recovery actions exhausted for {node_id}")
        await self._emit_node_failed_permanently(node_id)
        state.cooldown_until = datetime.now() + \
            self._get_cooldown(success=False)
        return None

    async def _execute_recovery(
        self,
        node: ClusterNode,
        action: RecoveryAction,
    ) -> RecoveryResult:
        """Execute a specific recovery action."""
        start_time = time.time()
        node_id = node.name

        try:
            if action == RecoveryAction.RESTART_P2P:
                success, error = await self._restart_p2p(node)
            elif action == RecoveryAction.RESTART_TAILSCALE:
                success, error = await self._restart_tailscale(node)
            elif action == RecoveryAction.REBOOT_INSTANCE:
                success, error = await self._reboot_instance(node)
            elif action == RecoveryAction.RECREATE_INSTANCE:
                success, error = await self._recreate_instance(node)
            else:
                success, error = False, f"Unknown action: {action}"

            duration = time.time() - start_time

            # Verify recovery
            if success:
                success = await self._verify_recovery(node, action)
                if not success:
                    error = "Recovery verification failed"

            return RecoveryResult(
                node_id=node_id,
                action=action,
                success=success,
                duration_seconds=duration,
                error=error,
            )

        except Exception as e:
            return RecoveryResult(
                node_id=node_id,
                action=action,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def _restart_p2p(self, node: ClusterNode) -> tuple[bool, str | None]:
        """Restart P2P orchestrator process."""
        ip = getattr(node, "best_ip", None) or getattr(node, "tailscale_ip", None)
        user = getattr(node, "ssh_user", "root")
        port = getattr(node, "ssh_port", 22)

        if not ip:
            return False, "No IP address"

        try:
            # Kill and restart P2P
            # Jan 2026: Switched to nohup and added screen cleanup to prevent dead sessions
            cmd = (
                "pkill -f 'python.*p2p_orchestrator' 2>/dev/null || true; "
                "screen -X -S p2p quit 2>/dev/null || true; "
                "screen -wipe 2>/dev/null || true; "
                "sleep 2; "
                "cd ~/ringrift/ai-service && "
                "mkdir -p logs && "
                "PYTHONPATH=. nohup python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &"
            )

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"{user}@{ip}",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(proc.wait(), timeout=30)

            if proc.returncode == 0:
                return True, None
            else:
                stderr = ""
                if proc.stderr:
                    stderr_data = await proc.stderr.read()
                    stderr = stderr_data.decode()[:100]
                return False, f"Exit code {proc.returncode}: {stderr}"

        except asyncio.TimeoutError:
            return False, "SSH timeout"
        except Exception as e:
            return False, str(e)

    async def _restart_tailscale(self, node: ClusterNode) -> tuple[bool, str | None]:
        """Restart Tailscale networking."""
        ip = getattr(node, "best_ip", None) or getattr(node, "tailscale_ip", None)
        user = getattr(node, "ssh_user", "root")
        port = getattr(node, "ssh_port", 22)

        if not ip:
            return False, "No IP address"

        try:
            # Use public IP if Tailscale is down
            ssh_ip = getattr(node, "ssh_host", None) or ip

            cmd = "sudo systemctl restart tailscaled && sleep 5 && tailscale status"

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=30",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"{user}@{ssh_ip}",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(proc.wait(), timeout=60)

            if proc.returncode == 0:
                return True, None
            else:
                stderr = ""
                if proc.stderr:
                    stderr_data = await proc.stderr.read()
                    stderr = stderr_data.decode()[:100]
                return False, f"Exit code {proc.returncode}: {stderr}"

        except asyncio.TimeoutError:
            return False, "SSH timeout"
        except Exception as e:
            return False, str(e)

    async def _reboot_instance(self, node: ClusterNode) -> tuple[bool, str | None]:
        """Reboot instance via provider API."""
        provider_name = getattr(node, "provider", None)
        instance_id = getattr(node, "instance_id", None)

        if not provider_name:
            return False, "No provider configured"
        if not instance_id:
            return False, "No instance_id configured"

        try:
            from app.coordination.providers.registry import get_provider

            provider = get_provider(provider_name)
            if not provider:
                return False, f"Provider {provider_name} not available"

            # Check if provider has reboot method
            if hasattr(provider, "reboot_instance"):
                success = await provider.reboot_instance(instance_id)
                if success:
                    # Wait for instance to come back
                    await asyncio.sleep(60)
                    return True, None
                else:
                    return False, "Reboot API call failed"
            else:
                return False, f"Provider {provider_name} does not support reboot"

        except Exception as e:
            return False, str(e)

    async def _recreate_instance(self, node: ClusterNode) -> tuple[bool, str | None]:
        """Terminate and recreate instance."""
        provider_name = getattr(node, "provider", None)
        instance_id = getattr(node, "instance_id", None)

        if not provider_name:
            return False, "No provider configured"
        if not instance_id:
            return False, "No instance_id configured"

        try:
            from app.coordination.providers.registry import get_provider

            provider = get_provider(provider_name)
            if not provider:
                return False, f"Provider {provider_name} not available"

            # Terminate existing instance
            result = await provider.scale_down([instance_id])
            if not result.get(instance_id, False):
                return False, "Failed to terminate instance"

            # Get node specs for recreation
            gpu_type_str = getattr(node, "gpu", None)
            if not gpu_type_str:
                return False, "No GPU type configured for recreation"

            from app.coordination.providers.base import GPUType

            gpu_type = GPUType.from_string(gpu_type_str)

            # Create new instance
            new_instances = await provider.scale_up(
                gpu_type=gpu_type,
                count=1,
                name_prefix=f"ringrift-{node.name}",
            )

            if new_instances:
                new_instance = new_instances[0]
                logger.info(
                    f"Recreated instance {node.name}: "
                    f"new_id={new_instance.id}, ip={new_instance.ip_address}"
                )
                # Update cluster config with new instance info
                from app.config.cluster_config import update_node_status

                config_updated = update_node_status(
                    node.name,
                    status="ready",
                    ssh_host=new_instance.ip_address,
                    # Preserve GPU info from original node
                    gpu=node.gpu if node.gpu else str(new_instance.gpu_type.value),
                    gpu_vram_gb=node.gpu_vram_gb if node.gpu_vram_gb else 0,
                )
                if not config_updated:
                    logger.warning(
                        f"Failed to update config for recreated instance {node.name}"
                    )
                return True, None
            else:
                return False, "Failed to create new instance"

        except Exception as e:
            return False, str(e)

    async def _verify_recovery(
        self,
        node: ClusterNode,
        action: RecoveryAction,
    ) -> bool:
        """Verify that recovery was successful."""
        # Wait for node to stabilize
        await asyncio.sleep(10)

        # Try P2P health check
        from .node_monitor import NodeMonitor, NodeMonitorConfig

        temp_monitor = NodeMonitor(
            config=NodeMonitorConfig(p2p_timeout_seconds=15),
            nodes=[node],
        )

        result = await temp_monitor._check_p2p(node)
        return result.healthy

    async def _get_node(self, node_id: str) -> ClusterNode | None:
        """Get node from cluster config."""
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            return nodes.get(node_id)
        except Exception as e:
            logger.warning(f"Failed to get node {node_id}: {e}")
            return None

    def _get_backoff(self, action: RecoveryAction, attempt: int) -> float:
        """Calculate backoff duration for failed attempt."""
        base = self.config.backoff_base_seconds
        multiplier = self.config.backoff_multiplier
        return min(base * (multiplier ** (attempt - 1)), 300)

    def _get_cooldown(self, success: bool) -> float:
        """Get cooldown duration after recovery attempt."""
        from datetime import timedelta

        if success:
            return timedelta(
                seconds=self.config.cooldown_after_success_seconds
            )
        else:
            return timedelta(
                seconds=self.config.cooldown_after_exhausted_seconds
            )

    async def _emit_recovery_success(self, result: RecoveryResult) -> None:
        """Emit RECOVERY_COMPLETED event."""
        try:
            from app.distributed.data_events import DataEventType

            await self._safe_emit_event_async(
                DataEventType.RECOVERY_COMPLETED.value,
                {
                    "node_id": result.node_id,
                    "action": result.action.name,
                    "duration_seconds": result.duration_seconds,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit RECOVERY_COMPLETED: {e}")

    async def _emit_recovery_failed(self, result: RecoveryResult) -> None:
        """Emit RECOVERY_FAILED event."""
        try:
            from app.distributed.data_events import DataEventType

            await self._safe_emit_event_async(
                DataEventType.RECOVERY_FAILED.value,
                result.to_dict(),
            )
        except Exception as e:
            logger.debug(f"Failed to emit RECOVERY_FAILED: {e}")

    async def _emit_node_failed_permanently(self, node_id: str) -> None:
        """Emit event when all recovery attempts exhausted."""
        try:
            await self._safe_emit_event_async(
                "NODE_FAILED_PERMANENTLY",
                {
                    "node_id": node_id,
                    "reason": "all_recovery_actions_exhausted",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit NODE_FAILED_PERMANENTLY: {e}")

    def get_recovery_state(self, node_id: str) -> dict | None:
        """Get current recovery state for a node."""
        state = self._recovery_states.get(node_id)
        if not state:
            return None

        return {
            "node_id": node_id,
            "current_action_index": state.current_action_index,
            "attempts": {a.name: c for a, c in state.attempts.items()},
            "last_attempt": state.last_attempt.isoformat() if state.last_attempt else None,
            "last_success": state.last_success.isoformat() if state.last_success else None,
            "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
            "in_cooldown": bool(
                state.cooldown_until and datetime.now() < state.cooldown_until
            ),
        }

    def get_recovery_history(self, limit: int = 50) -> list[dict]:
        """Get recent recovery history."""
        return [r.to_dict() for r in self._recovery_history[-limit:]]

    def health_check(self) -> HealthCheckResult:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult with status based on recovery queue state.
        """
        in_recovery = sum(
            1 for state in self._recovery_states.values()
            if state.current_action_index > 0
        )
        queue_size = self._recovery_queue.qsize()

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING if self._running else CoordinatorStatus.STOPPED,
            message=f"Recovery engine: {in_recovery} nodes in recovery, {queue_size} queued",
            details={
                "nodes_in_recovery": in_recovery,
                "queue_size": queue_size,
                "total_recoveries": len(self._recovery_history),
                "cycles_completed": self._stats.cycles_completed,
                "errors_count": self._stats.errors_count,
            },
        )


# =============================================================================
# Singleton Access (using HandlerBase class methods)
# =============================================================================


def get_recovery_engine() -> RecoveryEngine:
    """Get or create the singleton RecoveryEngine instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return RecoveryEngine.get_instance()


def reset_recovery_engine() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    RecoveryEngine.reset_instance()

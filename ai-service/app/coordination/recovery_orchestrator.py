"""Recovery Orchestrator for automated node recovery.

Implements graduated recovery actions when nodes become unhealthy,
from simple restarts to full reprovisioning.

Recovery Ladder (in order):
1. Restart P2P daemon
2. Restart Tailscale
3. Soft reboot (provider API)
4. Hard reboot (provider API)
5. Reprovision (Vast.ai only)
6. Alert human

Features:
- Cooldown tracking per node per action
- Maximum retries per hour
- Circuit breaker for repeated failures
- Slack alerting for escalations

Usage:
    from app.coordination.recovery_orchestrator import (
        RecoveryOrchestrator,
        get_recovery_orchestrator,
    )

    orchestrator = get_recovery_orchestrator()
    result = await orchestrator.attempt_recovery("gpu-node-1")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from app.providers import (
    AWSManager,
    HetznerManager,
    LambdaManager,
    Provider,
    ProviderInstance,
    TailscaleManager,
    VastManager,
)
from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    NodeHealthDetails,
    NodeHealthState,
    get_health_orchestrator,
)

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Recovery actions in escalation order."""

    RESTART_P2P = "restart_p2p"
    RESTART_TAILSCALE = "restart_tailscale"
    SOFT_REBOOT = "soft_reboot"
    HARD_REBOOT = "hard_reboot"
    REPROVISION = "reprovision"  # Vast.ai only
    ALERT_HUMAN = "alert_human"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    node_id: str
    action: RecoveryAction
    success: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    next_action: RecoveryAction | None = None


@dataclass
class NodeRecoveryState:
    """Tracks recovery attempts for a node."""

    node_id: str
    last_action: RecoveryAction | None = None
    last_attempt: float = 0.0
    attempts_this_hour: int = 0
    hour_start: float = 0.0
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_open_until: float = 0.0

    # Per-action cooldowns
    action_cooldowns: dict[RecoveryAction, float] = field(default_factory=dict)

    def is_action_on_cooldown(self, action: RecoveryAction, cooldown_minutes: int) -> bool:
        """Check if action is on cooldown."""
        last = self.action_cooldowns.get(action, 0)
        return (time.time() - last) < (cooldown_minutes * 60)

    def record_attempt(self, action: RecoveryAction, success: bool) -> None:
        """Record a recovery attempt."""
        now = time.time()
        self.last_action = action
        self.last_attempt = now
        self.action_cooldowns[action] = now

        # Reset hourly counter if new hour
        if now - self.hour_start > 3600:
            self.attempts_this_hour = 0
            self.hour_start = now

        self.attempts_this_hour += 1

        if success:
            self.consecutive_failures = 0
            self.circuit_open = False
        else:
            self.consecutive_failures += 1
            # Open circuit breaker after 3 consecutive failures
            if self.consecutive_failures >= 3:
                self.circuit_open = True
                self.circuit_open_until = now + 1800  # 30 min

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_open:
            return False
        if time.time() > self.circuit_open_until:
            self.circuit_open = False
            return False
        return True


class RecoveryOrchestrator:
    """Orchestrates automated recovery of failing nodes.

    Implements an escalation ladder with cooldowns and circuit breakers
    to prevent recovery storms.
    """

    # Cooldowns (minutes)
    ACTION_COOLDOWNS = {
        RecoveryAction.RESTART_P2P: 5,
        RecoveryAction.RESTART_TAILSCALE: 10,
        RecoveryAction.SOFT_REBOOT: 15,
        RecoveryAction.HARD_REBOOT: 30,
        RecoveryAction.REPROVISION: 60,
        RecoveryAction.ALERT_HUMAN: 120,
    }

    # Max attempts per hour per node
    MAX_ATTEMPTS_PER_HOUR = 6

    # Recovery escalation order
    ESCALATION_ORDER = [
        RecoveryAction.RESTART_P2P,
        RecoveryAction.RESTART_TAILSCALE,
        RecoveryAction.SOFT_REBOOT,
        RecoveryAction.HARD_REBOOT,
        RecoveryAction.REPROVISION,
        RecoveryAction.ALERT_HUMAN,
    ]

    def __init__(
        self,
        health_orchestrator: HealthCheckOrchestrator | None = None,
        slack_webhook_url: str | None = None,
        alerts_config_path: str = "config/alerts.yaml",
    ):
        """Initialize recovery orchestrator.

        Args:
            health_orchestrator: Health check orchestrator instance
            slack_webhook_url: Slack webhook for alerts
            alerts_config_path: Path to alerts config file
        """
        self.health_orchestrator = health_orchestrator or get_health_orchestrator()

        # Provider managers
        self.lambda_mgr = LambdaManager()
        self.vast_mgr = VastManager()
        self.hetzner_mgr = HetznerManager()
        self.aws_mgr = AWSManager()
        self.tailscale_mgr = TailscaleManager()

        # Recovery state per node
        self.node_states: dict[str, NodeRecoveryState] = {}

        # Alerting
        self.slack_webhook_url = slack_webhook_url
        self._load_alerts_config(alerts_config_path)

    def _load_alerts_config(self, config_path: str) -> None:
        """Load alerting configuration."""
        path = Path(config_path)
        if path.exists():
            try:
                with open(path) as f:
                    config = yaml.safe_load(f) or {}
                    if not self.slack_webhook_url:
                        self.slack_webhook_url = config.get("slack_webhook_url")
                    logger.info("[RecoveryOrchestrator] Loaded alerts config")
            except Exception as e:
                logger.warning(f"[RecoveryOrchestrator] Could not load alerts config: {e}")

    def _get_node_state(self, node_id: str) -> NodeRecoveryState:
        """Get or create recovery state for a node."""
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeRecoveryState(node_id=node_id)
        return self.node_states[node_id]

    def _get_manager_for_provider(self, provider: Provider):
        """Get appropriate manager for provider."""
        managers = {
            Provider.LAMBDA: self.lambda_mgr,
            Provider.VAST: self.vast_mgr,
            Provider.HETZNER: self.hetzner_mgr,
            Provider.AWS: self.aws_mgr,
        }
        return managers.get(provider)

    def _get_next_action(
        self,
        node_state: NodeRecoveryState,
        health: NodeHealthDetails,
    ) -> RecoveryAction | None:
        """Determine next recovery action to try.

        Returns:
            Next action to try, or None if no action available
        """
        # Check rate limit
        if node_state.attempts_this_hour >= self.MAX_ATTEMPTS_PER_HOUR:
            logger.warning(
                f"[RecoveryOrchestrator] {node_state.node_id}: "
                f"Hit hourly limit ({self.MAX_ATTEMPTS_PER_HOUR} attempts)"
            )
            return RecoveryAction.ALERT_HUMAN

        # Check circuit breaker
        if node_state.is_circuit_open():
            logger.warning(
                f"[RecoveryOrchestrator] {node_state.node_id}: Circuit breaker open"
            )
            return RecoveryAction.ALERT_HUMAN

        # Find first action not on cooldown
        for action in self.ESCALATION_ORDER:
            # Skip reprovision for non-Vast nodes
            if action == RecoveryAction.REPROVISION:
                if health.provider != Provider.VAST:
                    continue

            cooldown = self.ACTION_COOLDOWNS.get(action, 10)
            if not node_state.is_action_on_cooldown(action, cooldown):
                return action

        # All actions on cooldown
        return None

    async def attempt_recovery(
        self,
        node_id: str,
        force_action: RecoveryAction | None = None,
    ) -> RecoveryResult:
        """Attempt to recover an unhealthy node.

        Args:
            node_id: Node to recover
            force_action: Force a specific action (bypass escalation)

        Returns:
            RecoveryResult with outcome
        """
        start = time.time()
        node_state = self._get_node_state(node_id)

        # Get current health
        health = self.health_orchestrator.get_node_health(node_id)
        if not health:
            return RecoveryResult(
                node_id=node_id,
                action=RecoveryAction.ALERT_HUMAN,
                success=False,
                message="Node not found in health data",
            )

        # Check if recovery needed
        if health.state == NodeHealthState.HEALTHY:
            return RecoveryResult(
                node_id=node_id,
                action=RecoveryAction.RESTART_P2P,  # Placeholder
                success=True,
                message="Node is healthy, no recovery needed",
            )

        # Determine action
        action = force_action or self._get_next_action(node_state, health)
        if not action:
            return RecoveryResult(
                node_id=node_id,
                action=RecoveryAction.ALERT_HUMAN,
                success=False,
                message="All recovery actions on cooldown",
            )

        logger.info(
            f"[RecoveryOrchestrator] Attempting {action.value} on {node_id} "
            f"(state={health.state.value})"
        )

        # Execute action
        success, message = await self._execute_action(action, health)

        # Record attempt
        node_state.record_attempt(action, success)

        duration = (time.time() - start) * 1000

        # Determine next action if this failed
        next_action = None
        if not success:
            next_action = self._get_next_action(node_state, health)

        result = RecoveryResult(
            node_id=node_id,
            action=action,
            success=success,
            message=message,
            duration_ms=duration,
            next_action=next_action,
        )

        # Log result
        status = "succeeded" if success else "failed"
        logger.info(
            f"[RecoveryOrchestrator] {action.value} on {node_id} {status}: {message}"
        )

        return result

    async def _execute_action(
        self,
        action: RecoveryAction,
        health: NodeHealthDetails,
    ) -> tuple[bool, str]:
        """Execute a recovery action.

        Returns:
            Tuple of (success, message)
        """
        instance = health.instance
        if not instance:
            return False, "No instance data available"

        manager = self._get_manager_for_provider(health.provider)
        if not manager:
            return False, f"No manager for provider {health.provider}"

        try:
            if action == RecoveryAction.RESTART_P2P:
                return await self._restart_p2p(manager, instance)

            elif action == RecoveryAction.RESTART_TAILSCALE:
                return await self._restart_tailscale(manager, instance)

            elif action == RecoveryAction.SOFT_REBOOT:
                return await self._soft_reboot(manager, instance)

            elif action == RecoveryAction.HARD_REBOOT:
                return await self._hard_reboot(manager, instance)

            elif action == RecoveryAction.REPROVISION:
                if health.provider != Provider.VAST:
                    return False, "Reprovision only supported for Vast.ai"
                return await self._reprovision_vast(instance)

            elif action == RecoveryAction.ALERT_HUMAN:
                return await self._alert_human(health)

            else:
                return False, f"Unknown action: {action}"

        except Exception as e:
            return False, f"Error executing {action.value}: {e}"

    async def _restart_p2p(
        self,
        manager,
        instance: ProviderInstance,
    ) -> tuple[bool, str]:
        """Restart P2P daemon on instance."""
        cmd = """
cd ~/ringrift/ai-service 2>/dev/null || cd /root/ringrift/ai-service
pkill -f 'app.p2p.orchestrator' 2>/dev/null
sleep 2
source venv/bin/activate 2>/dev/null || true
nohup python -m app.p2p.orchestrator > logs/p2p.log 2>&1 &
sleep 3
pgrep -f 'app.p2p.orchestrator' && echo "P2P restarted"
"""
        code, stdout, stderr = await manager.run_ssh_command(instance, cmd, timeout=30)

        if code == 0 and "P2P restarted" in stdout:
            return True, "P2P daemon restarted successfully"
        return False, f"P2P restart failed: {stderr or stdout}"

    async def _restart_tailscale(
        self,
        manager,
        instance: ProviderInstance,
    ) -> tuple[bool, str]:
        """Restart Tailscale on instance."""
        # Try systemctl first (Lambda/Hetzner), then direct command (Vast)
        cmd = """
if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl restart tailscaled
else
    sudo pkill tailscaled
    sleep 2
    sudo tailscaled &
fi
sleep 3
tailscale status >/dev/null 2>&1 && echo "Tailscale OK"
"""
        code, stdout, stderr = await manager.run_ssh_command(instance, cmd, timeout=30)

        if code == 0 and "Tailscale OK" in stdout:
            return True, "Tailscale restarted successfully"
        return False, f"Tailscale restart failed: {stderr or stdout}"

    async def _soft_reboot(
        self,
        manager,
        instance: ProviderInstance,
    ) -> tuple[bool, str]:
        """Soft reboot via provider API."""
        success = await manager.reboot_instance(instance.instance_id)
        if success:
            return True, "Soft reboot initiated"
        return False, "Soft reboot failed"

    async def _hard_reboot(
        self,
        manager,
        instance: ProviderInstance,
    ) -> tuple[bool, str]:
        """Hard reboot (power cycle) via provider API."""
        # Most providers don't distinguish soft/hard reboot
        # For now, just retry the reboot
        success = await manager.reboot_instance(instance.instance_id)
        if success:
            return True, "Hard reboot initiated"
        return False, "Hard reboot failed"

    async def _reprovision_vast(
        self,
        instance: ProviderInstance,
    ) -> tuple[bool, str]:
        """Reprovision a Vast.ai instance.

        Terminates the current instance and launches a replacement.
        """
        old_id = instance.instance_id
        gpu_name = instance.metadata.get("gpu_name")

        # Find a replacement offer
        offers = await self.vast_mgr.search_offers(
            gpu_name=gpu_name,
            max_price=instance.hourly_cost * 1.2,  # Allow 20% higher
            limit=5,
        )

        if not offers:
            return False, f"No replacement offers found for {gpu_name}"

        # Launch replacement
        new_id = await self.vast_mgr.launch_instance({
            "offer_id": offers[0].offer_id,
            "disk_gb": 50,
            "onstart_cmd": "apt-get update && apt-get install -y git curl",
        })

        if not new_id:
            return False, "Failed to launch replacement instance"

        # Terminate old instance
        await self.vast_mgr.terminate_instance(old_id)

        return True, f"Reprovisioned: {old_id} → {new_id}"

    async def _alert_human(
        self,
        health: NodeHealthDetails,
    ) -> tuple[bool, str]:
        """Send alert to human operators."""
        message = (
            f"🚨 *Node Recovery Escalation*\n"
            f"Node: `{health.node_id}`\n"
            f"Provider: {health.provider.value if health.provider else 'unknown'}\n"
            f"State: {health.state.value}\n"
            f"Last error: {health.last_error or 'N/A'}\n"
            f"Consecutive failures: {health.consecutive_failures}\n"
            f"Timestamp: {datetime.now().isoformat()}"
        )

        # Log to file
        logger.error(f"[RecoveryOrchestrator] ALERT: {health.node_id} needs human intervention")

        # Send Slack if configured
        if self.slack_webhook_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.slack_webhook_url,
                        json={"text": message},
                        timeout=aiohttp.ClientTimeout(total=10),
                    )
                return True, "Alert sent to Slack"
            except Exception as e:
                logger.error(f"[RecoveryOrchestrator] Slack alert failed: {e}")
                return False, f"Slack alert failed: {e}"

        return True, "Alert logged (no Slack configured)"

    async def recover_all_unhealthy(self) -> list[RecoveryResult]:
        """Attempt recovery on all unhealthy nodes.

        Returns:
            List of recovery results
        """
        results = []

        # Get unhealthy nodes
        unhealthy_states = [
            NodeHealthState.DEGRADED,
            NodeHealthState.UNHEALTHY,
            NodeHealthState.OFFLINE,
        ]

        for state in unhealthy_states:
            node_ids = self.health_orchestrator.get_nodes_by_state(state)
            for node_id in node_ids:
                result = await self.attempt_recovery(node_id)
                results.append(result)

                # Small delay between recoveries
                await asyncio.sleep(1)

        return results

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        total_attempts = sum(s.attempts_this_hour for s in self.node_states.values())
        circuit_open = sum(1 for s in self.node_states.values() if s.is_circuit_open())

        return {
            "total_nodes_tracked": len(self.node_states),
            "total_attempts_this_hour": total_attempts,
            "nodes_with_circuit_open": circuit_open,
            "nodes_on_cooldown": sum(
                1 for s in self.node_states.values()
                if any(s.is_action_on_cooldown(a, self.ACTION_COOLDOWNS[a]) for a in RecoveryAction)
            ),
        }

    async def deploy_ssh_key(
        self,
        node_id: str,
        public_key: str,
    ) -> bool:
        """Deploy SSH public key to a node.

        Args:
            node_id: Target node
            public_key: SSH public key to deploy

        Returns:
            True if successful
        """
        health = self.health_orchestrator.get_node_health(node_id)
        if not health or not health.instance:
            logger.error(f"[RecoveryOrchestrator] Node {node_id} not found")
            return False

        manager = self._get_manager_for_provider(health.provider)
        if not manager:
            return False

        # Escape the key for shell
        escaped_key = public_key.replace("'", "'\"'\"'")
        cmd = f"""
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo '{escaped_key}' >> ~/.ssh/authorized_keys
sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
echo "Key deployed"
"""
        code, stdout, stderr = await manager.run_ssh_command(
            health.instance, cmd, timeout=30
        )

        if code == 0 and "deployed" in stdout:
            logger.info(f"[RecoveryOrchestrator] Deployed SSH key to {node_id}")
            return True

        logger.error(f"[RecoveryOrchestrator] SSH key deploy failed on {node_id}: {stderr}")
        return False


# Global instance
_recovery_orchestrator: RecoveryOrchestrator | None = None


def get_recovery_orchestrator() -> RecoveryOrchestrator:
    """Get or create the global recovery orchestrator."""
    global _recovery_orchestrator

    if _recovery_orchestrator is None:
        _recovery_orchestrator = RecoveryOrchestrator()

    return _recovery_orchestrator


async def attempt_recovery(node_id: str) -> RecoveryResult:
    """Attempt recovery on a node.

    Convenience function using global orchestrator.
    """
    return await get_recovery_orchestrator().attempt_recovery(node_id)

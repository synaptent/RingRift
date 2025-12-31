"""Node Recovery Daemon (December 2025 - Phase 21).

Monitors cluster for terminated or failed nodes and automatically triggers
recovery actions.

Key features:
- Monitors P2P cluster for node failures
- Detects terminated Lambda instances via API
- Auto-restarts terminated instances (requires API credentials)
- Proactive recovery based on resource trends
- Integration with event system for cluster health events

Usage:
    from app.coordination.node_recovery_daemon import NodeRecoveryDaemon

    daemon = NodeRecoveryDaemon()
    await daemon.start()

Environment variables:
    LAMBDA_API_KEY: Lambda Cloud API key for instance management
    RINGRIFT_NODE_RECOVERY_ENABLED: Enable/disable recovery (default: 1)
    RINGRIFT_NODE_RECOVERY_INTERVAL: Check interval in seconds (default: 300)

Dec 2025: Refactored to use BaseDaemon base class (~100 LOC reduction).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import JobDaemonStats

logger = logging.getLogger(__name__)

# Health event emission (uses safe fallbacks internally)
from app.coordination.event_emitters import (
    emit_node_unhealthy,
    emit_node_recovered,
)


class NodeRecoveryAction(Enum):
    """Types of node-level recovery actions.

    NOTE (Dec 2025): Renamed from RecoveryAction to avoid collision with
    JobRecoveryAction in unified_health_manager.py and SystemRecoveryAction
    in recovery_orchestrator.py which have different semantics.

    December 2025: Added escalation tiers with graduated responses:
    - SOFT_RESTART: Restart P2P service only (3 failures)
    - RESTART: Full instance restart (6 failures)
    - FAILOVER: Migrate workload to another node (10 failures)
    - RETIRE: Mark as retired, alert operator (20 failures)
    """
    NONE = "none"
    SOFT_RESTART = "soft_restart"  # December 2025: P2P service restart only
    RESTART = "restart"  # Full instance restart
    PREEMPTIVE_RESTART = "preemptive_restart"
    NOTIFY = "notify"  # Just notify, don't auto-recover
    FAILOVER = "failover"  # Migrate workload to another node
    RETIRE = "retire"  # December 2025: Mark as retired, alert operator


# Backward-compat alias (deprecated)
RecoveryAction = NodeRecoveryAction


class NodeProvider(Enum):
    """Cloud provider types."""
    LAMBDA = "lambda"
    VAST = "vast"
    RUNPOD = "runpod"
    HETZNER = "hetzner"
    UNKNOWN = "unknown"


@dataclass
class NodeRecoveryConfig:
    """Configuration for node recovery.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.

    Added escalation tiers for graduated recovery responses.
    """

    # Daemon control
    check_interval_seconds: int = 300  # 5 minutes
    enabled: bool = True

    # Provider-specific settings
    lambda_api_key: str = ""
    vast_api_key: str = ""
    runpod_api_key: str = ""
    # Recovery thresholds
    max_consecutive_failures: int = 3
    recovery_cooldown_seconds: int = 600  # 10 minutes between recovery attempts
    # Proactive recovery settings
    memory_exhaustion_threshold: float = 0.02  # 2% per minute memory growth
    memory_exhaustion_window_minutes: int = 30
    preemptive_recovery_enabled: bool = True

    # December 2025: Escalation tiers for graduated recovery
    # Failures -> Action:
    #   3 failures: soft restart (P2P service only)
    #   6 failures: hard restart (instance restart)
    #   10 failures: failover (migrate workload)
    #   20 failures: retire (mark as retired, alert operator)
    escalation_soft_restart_threshold: int = 3
    escalation_hard_restart_threshold: int = 6
    escalation_failover_threshold: int = 10
    escalation_retire_threshold: int = 20

    # Escalating cooldowns (seconds) by tier
    # Tier 1 (soft restart): 60s
    # Tier 2 (hard restart): 300s
    # Tier 3 (failover): 900s
    # Tier 4 (retire): 3600s
    escalation_cooldowns: tuple[int, ...] = (60, 300, 900, 3600)

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_NODE_RECOVERY") -> "NodeRecoveryConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get(f"{prefix}_ENABLED", "1") == "1"
        if os.environ.get(f"{prefix}_INTERVAL"):
            config.check_interval_seconds = int(os.environ.get(f"{prefix}_INTERVAL", "300"))
        config.lambda_api_key = os.environ.get("LAMBDA_API_KEY", "")
        config.vast_api_key = os.environ.get("VAST_API_KEY", "")
        config.runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")
        config.preemptive_recovery_enabled = (
            os.environ.get("RINGRIFT_PREEMPTIVE_RECOVERY", "1") == "1"
        )
        return config


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    provider: NodeProvider = NodeProvider.UNKNOWN
    status: str = "unknown"  # running, terminated, failed, unreachable
    last_seen: float = 0.0
    consecutive_failures: int = 0
    last_recovery_attempt: float = 0.0
    instance_id: str = ""  # Provider-specific instance ID
    # Resource tracking for proactive recovery
    memory_samples: list[float] = field(default_factory=list)
    sample_timestamps: list[float] = field(default_factory=list)


@dataclass
class RecoveryStats(JobDaemonStats):
    """Statistics for recovery operations.

    December 2025: Now extends JobDaemonStats for consistent tracking.
    Inherits: jobs_processed, jobs_succeeded, jobs_failed, is_healthy(), etc.
    """

    # Recovery-specific fields
    preemptive_recoveries: int = 0

    # Backward compatibility aliases
    @property
    def total_checks(self) -> int:
        """Alias for jobs_processed (backward compatibility)."""
        return self.jobs_processed

    @property
    def nodes_recovered(self) -> int:
        """Alias for jobs_succeeded (backward compatibility)."""
        return self.jobs_succeeded

    @property
    def recovery_failures(self) -> int:
        """Alias for jobs_failed (backward compatibility)."""
        return self.jobs_failed

    def record_check(self) -> None:
        """Record a node health check."""
        self.last_job_time = time.time()
        self.jobs_processed += 1

    def record_recovery_success(self, preemptive: bool = False) -> None:
        """Record a successful recovery."""
        self.record_job_success()
        if preemptive:
            self.preemptive_recoveries += 1

    def record_recovery_failure(self, error: str) -> None:
        """Record a failed recovery."""
        self.record_job_failure(error)


class NodeRecoveryDaemon(HandlerBase):
    """Daemon that monitors nodes and triggers recovery actions.

    Continuously monitors cluster for terminated or failing nodes
    and automatically triggers recovery when possible.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _get_event_subscriptions() for event registration
    """

    def __init__(self, config: NodeRecoveryConfig | None = None):
        self._daemon_config = config or NodeRecoveryConfig.from_env()

        super().__init__(
            name="NodeRecoveryDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        # Recovery-specific stats (supplement HandlerBase._stats)
        self._recovery_stats = RecoveryStats()

        # Track node states
        self._node_states: dict[str, NodeInfo] = {}

        # HTTP session for API calls
        self._http_session = None

        # Log startup configuration
        logger.info(
            f"[{self.name}] Config: "
            f"interval={self._daemon_config.check_interval_seconds}s, "
            f"lambda_api={'set' if self._daemon_config.lambda_api_key else 'not set'}"
        )

    @property
    def config(self) -> NodeRecoveryConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions for HandlerBase.

        December 2025: Converted from _subscribe_to_events() for HandlerBase pattern.
        """
        return {
            "P2P_NODES_DEAD": self._on_nodes_dead,
            "P2P_NODE_DEAD": self._on_single_node_dead,
        }

    def _on_nodes_dead(self, event) -> None:
        """Handle P2P_NODES_DEAD event (batch of dead nodes)."""
        payload = event.payload if hasattr(event, 'payload') else event
        dead_nodes = payload.get("nodes", [])

        for node_id in dead_nodes:
            if node_id in self._node_states:
                node = self._node_states[node_id]
                node.consecutive_failures += 1
                node.status = "unreachable"
                logger.info(
                    f"[NodeRecoveryDaemon] Node {node_id} marked unreachable "
                    f"(failures: {node.consecutive_failures})"
                )

    def _on_single_node_dead(self, event) -> None:
        """Handle P2P_NODE_DEAD event (single node).

        Dec 2025: Added to handle single node death events from P2P orchestrator.
        Previously only batch P2P_NODES_DEAD was handled, causing single node
        failures to be missed until batch heartbeat timeout.
        """
        payload = event.payload if hasattr(event, 'payload') else event
        node_id = payload.get("node_id")
        reason = payload.get("reason", "unknown")

        if not node_id:
            logger.warning("[NodeRecoveryDaemon] Received P2P_NODE_DEAD without node_id")
            return

        if node_id in self._node_states:
            node = self._node_states[node_id]
            node.consecutive_failures += 1
            node.status = "unreachable"
            node.last_failure_reason = reason
            logger.info(
                f"[NodeRecoveryDaemon] Single node {node_id} confirmed dead "
                f"(reason: {reason}, failures: {node.consecutive_failures})"
            )
        else:
            # Create new state for unknown node
            logger.info(
                f"[NodeRecoveryDaemon] Unknown node {node_id} reported dead, "
                f"will track on next cluster scan"
            )

    async def stop(self) -> None:
        """Graceful shutdown handler.

        December 2025: Override HandlerBase.stop() for clean daemon shutdown.
        """
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()

        # Log stats
        logger.info(
            f"[{self.name}] Stats: "
            f"{self._recovery_stats.nodes_recovered} recovered, "
            f"{self._recovery_stats.recovery_failures} failures"
        )

        try:
            # Emit shutdown event
            from app.coordination.event_emitters import emit_coordinator_shutdown

            await emit_coordinator_shutdown(
                coordinator_name=self.name,
                reason="graceful",
                remaining_tasks=0,
                state_snapshot={
                    "total_checks": self._recovery_stats.total_checks,
                    "nodes_recovered": self._recovery_stats.nodes_recovered,
                    "recovery_failures": self._recovery_stats.recovery_failures,
                    "tracked_nodes": len(self._node_states),
                },
            )
            logger.info(f"[{self.name}] Graceful shutdown complete")
        except ImportError:
            logger.debug(f"[{self.name}] Event emitters not available for shutdown")
        except Exception as e:
            logger.warning(f"[{self.name}] Error during shutdown: {e}")

        # Call parent stop
        await super().stop()

    async def _run_cycle(self) -> None:
        """Run one recovery cycle.

        Called by HandlerBase's main loop.
        """
        await self._check_nodes()
        self._recovery_stats.record_check()  # Updates jobs_processed and last_job_time

    async def _check_nodes(self) -> None:
        """Check all known nodes and trigger recovery if needed."""
        try:
            # Get current cluster state
            await self._update_node_states()

            # Check each node for recovery needs
            for node_id, node in list(self._node_states.items()):
                action = self._determine_recovery_action(node)

                if action != RecoveryAction.NONE:
                    await self._execute_recovery(node, action)

        except Exception as e:
            logger.warning(f"Node check error: {e}")

    async def _update_node_states(self) -> None:
        """Update node states from P2P cluster."""
        try:
            from app.coordination.p2p_integration import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return

            status = await p2p.get_status()
            if not status:
                return

            # Get alive and dead peers
            alive_peers = status.get("alive_peers", [])
            dead_peers = status.get("dead_peers", [])

            # Update alive nodes
            if isinstance(alive_peers, list):
                for peer in alive_peers:
                    if isinstance(peer, dict):
                        node_id = peer.get("node_id", "")
                        if node_id:
                            self._update_node_info(node_id, peer, "running")

            # Update dead nodes
            if isinstance(dead_peers, list):
                for peer_id in dead_peers:
                    if isinstance(peer_id, str) and peer_id in self._node_states:
                        node = self._node_states[peer_id]
                        node.status = "unreachable"
                        node.consecutive_failures += 1

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to update node states: {e}")

    def _update_node_info(
        self,
        node_id: str,
        info: dict[str, Any],
        status: str,
    ) -> None:
        """Update or create node info from P2P data."""
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeInfo(
                node_id=node_id,
                host=info.get("host", ""),
                provider=self._detect_provider(node_id, info),
                instance_id=info.get("instance_id", ""),
            )

        node = self._node_states[node_id]
        node.status = status
        node.last_seen = time.time()

        # Reset failure count on successful contact
        if status == "running":
            node.consecutive_failures = 0

        # Track memory for proactive recovery
        memory_used = info.get("memory_used_percent", 0.0)
        if memory_used > 0:
            now = time.time()
            node.memory_samples.append(memory_used)
            node.sample_timestamps.append(now)

            # Keep last N samples
            max_samples = 60  # 1 sample per check, 60 = 5 hours at 5-min intervals
            if len(node.memory_samples) > max_samples:
                node.memory_samples = node.memory_samples[-max_samples:]
                node.sample_timestamps = node.sample_timestamps[-max_samples:]

    def _detect_provider(
        self,
        node_id: str,
        info: dict[str, Any],
    ) -> NodeProvider:
        """Detect the cloud provider for a node."""
        provider = info.get("provider", "").lower()

        if "lambda" in provider or "lambda" in node_id.lower():
            return NodeProvider.LAMBDA
        elif "vast" in provider or "vast" in node_id.lower():
            return NodeProvider.VAST
        elif "runpod" in provider or "runpod" in node_id.lower():
            return NodeProvider.RUNPOD
        elif "hetzner" in provider or "hetzner" in node_id.lower():
            return NodeProvider.HETZNER

        return NodeProvider.UNKNOWN

    def _determine_recovery_action(self, node: NodeInfo) -> RecoveryAction:
        """Determine what recovery action to take for a node.

        December 2025: Updated with escalation tiers for graduated responses.
        The action escalates based on consecutive failures:
        - 3 failures: soft restart (P2P service only)
        - 6 failures: hard restart (instance restart)
        - 10 failures: failover (migrate workload)
        - 20 failures: retire (mark as retired, alert operator)

        Cooldowns also escalate: 60s -> 300s -> 900s -> 3600s
        """
        now = time.time()

        # December 2025: Determine escalation tier and corresponding cooldown
        tier = self._get_escalation_tier(node.consecutive_failures)
        cooldown = self._get_tier_cooldown(tier)

        # Check cooldown based on tier
        if node.last_recovery_attempt > 0:
            time_since_recovery = now - node.last_recovery_attempt
            if time_since_recovery < cooldown:
                return RecoveryAction.NONE

        # Check for terminated/failed status
        if node.status in ("terminated", "failed"):
            return self._get_escalated_action(node.consecutive_failures)

        # Check for unreachable nodes
        if node.status == "unreachable":
            return self._get_escalated_action(node.consecutive_failures)

        # Proactive recovery based on resource trends
        if self.config.preemptive_recovery_enabled:
            trend_action = self._check_resource_trends(node)
            if trend_action != RecoveryAction.NONE:
                return trend_action

        return RecoveryAction.NONE

    def _get_escalation_tier(self, consecutive_failures: int) -> int:
        """Get escalation tier based on consecutive failures.

        Returns:
            Tier 0: < soft_restart threshold (notify only)
            Tier 1: >= soft_restart threshold (soft restart)
            Tier 2: >= hard_restart threshold (full restart)
            Tier 3: >= failover threshold (migrate workload)
            Tier 4: >= retire threshold (mark as retired)

        December 2025: Added for graduated recovery responses.
        """
        if consecutive_failures >= self.config.escalation_retire_threshold:
            return 4
        elif consecutive_failures >= self.config.escalation_failover_threshold:
            return 3
        elif consecutive_failures >= self.config.escalation_hard_restart_threshold:
            return 2
        elif consecutive_failures >= self.config.escalation_soft_restart_threshold:
            return 1
        else:
            return 0

    def _get_tier_cooldown(self, tier: int) -> int:
        """Get cooldown seconds for an escalation tier.

        December 2025: Cooldowns escalate with severity:
        - Tier 0/1: 60s (soft restart)
        - Tier 2: 300s (hard restart)
        - Tier 3: 900s (failover)
        - Tier 4: 3600s (retire)
        """
        if tier <= 0:
            return self.config.escalation_cooldowns[0] if self.config.escalation_cooldowns else 60
        elif tier >= len(self.config.escalation_cooldowns):
            return self.config.escalation_cooldowns[-1]
        else:
            return self.config.escalation_cooldowns[tier - 1]

    def _get_escalated_action(self, consecutive_failures: int) -> RecoveryAction:
        """Get the appropriate recovery action based on failure count.

        December 2025: Maps failure count to escalated recovery action:
        - < 3 failures: notify only
        - 3+ failures: soft restart (P2P service)
        - 6+ failures: hard restart (instance)
        - 10+ failures: failover (migrate workload)
        - 20+ failures: retire (mark as retired)
        """
        tier = self._get_escalation_tier(consecutive_failures)

        action_map = {
            0: RecoveryAction.NOTIFY,
            1: RecoveryAction.SOFT_RESTART,
            2: RecoveryAction.RESTART,
            3: RecoveryAction.FAILOVER,
            4: RecoveryAction.RETIRE,
        }

        action = action_map.get(tier, RecoveryAction.RETIRE)
        logger.debug(
            f"Escalation tier {tier} (failures={consecutive_failures}) -> {action.value}"
        )
        return action

    def _check_resource_trends(self, node: NodeInfo) -> RecoveryAction:
        """Check for resource exhaustion trends for preemptive recovery."""
        if len(node.memory_samples) < 5:
            return RecoveryAction.NONE

        now = time.time()
        window_seconds = self.config.memory_exhaustion_window_minutes * 60

        # Filter to samples within the window
        recent_samples = [
            (t, m) for t, m in zip(node.sample_timestamps, node.memory_samples)
            if now - t < window_seconds
        ]

        if len(recent_samples) < 5:
            return RecoveryAction.NONE

        # Calculate memory growth rate (% per minute)
        first_time, first_mem = recent_samples[0]
        last_time, last_mem = recent_samples[-1]

        time_diff_minutes = (last_time - first_time) / 60
        if time_diff_minutes < 5:
            return RecoveryAction.NONE

        growth_rate = (last_mem - first_mem) / time_diff_minutes

        # Check if memory is growing too fast
        if growth_rate > self.config.memory_exhaustion_threshold:
            # Project when memory will be exhausted (assuming 100% = exhaustion)
            remaining_memory = 100.0 - last_mem
            if remaining_memory > 0:
                minutes_to_exhaustion = remaining_memory / growth_rate
                if minutes_to_exhaustion < 60:  # Less than 60 minutes
                    logger.warning(
                        f"[NodeRecoveryDaemon] Node {node.node_id} memory exhaustion "
                        f"projected in {minutes_to_exhaustion:.0f} minutes "
                        f"(growth rate: {growth_rate:.2f}%/min)"
                    )
                    return RecoveryAction.PREEMPTIVE_RESTART

        return RecoveryAction.NONE

    async def _execute_recovery(
        self,
        node: NodeInfo,
        action: RecoveryAction,
    ) -> bool:
        """Execute a recovery action for a node.

        P0.5 Dec 2025: Now includes post-restart health verification to ensure
        the node actually came back up after restart API call.
        """
        node.last_recovery_attempt = time.time()

        if action == RecoveryAction.NOTIFY:
            # Just emit an event
            self._emit_recovery_event(node, action, success=True)
            return True

        if action == RecoveryAction.SOFT_RESTART:
            # Dec 31, 2025: Restart P2P service only via SSH (not full instance)
            ssh_success = await self._soft_restart_p2p(node)
            if ssh_success:
                # Verify node rejoined P2P cluster
                verified = await self._verify_node_health_after_restart(node)
                if verified:
                    self._recovery_stats.record_recovery_success()
                    self._emit_recovery_event(node, action, success=True)
                    return True
                else:
                    self._recovery_stats.record_recovery_failure(
                        f"soft_restart_verification_failed:{node.node_id}"
                    )
                    self._emit_recovery_event(node, action, success=False)
                    return False
            else:
                self._recovery_stats.record_recovery_failure(
                    f"soft_restart_ssh_failed:{node.node_id}"
                )
                self._emit_recovery_event(node, action, success=False)
                return False

        if action == RecoveryAction.RESTART:
            api_success = await self._restart_node(node)
            if api_success:
                # P0.5 Dec 2025: Verify node actually came back up
                verified = await self._verify_node_health_after_restart(node)
                if verified:
                    self._recovery_stats.record_recovery_success()
                    self._emit_recovery_event(node, action, success=True)
                    return True
                else:
                    # API succeeded but node didn't come back up
                    self._recovery_stats.record_recovery_failure(
                        f"verification_failed:{node.node_id}"
                    )
                    self._emit_recovery_event(node, action, success=False)
                    return False
            else:
                self._recovery_stats.record_recovery_failure(f"restart_failed:{node.node_id}")
                self._emit_recovery_event(node, action, success=False)
                return False

        if action == RecoveryAction.PREEMPTIVE_RESTART:
            api_success = await self._restart_node(node)
            if api_success:
                # P0.5 Dec 2025: Verify node actually came back up
                verified = await self._verify_node_health_after_restart(node)
                if verified:
                    self._recovery_stats.record_recovery_success(preemptive=True)
                    self._emit_recovery_event(node, action, success=True)
                    return True
                else:
                    # API succeeded but node didn't come back up
                    self._recovery_stats.record_recovery_failure(
                        f"preemptive_verification_failed:{node.node_id}"
                    )
                    self._emit_recovery_event(node, action, success=False)
                    return False
            else:
                self._recovery_stats.record_recovery_failure(
                    f"preemptive_restart_failed:{node.node_id}"
                )
                self._emit_recovery_event(node, action, success=False)
                return False

        return False

    async def _restart_node(self, node: NodeInfo) -> bool:
        """Restart a node via its provider's API."""
        if node.provider == NodeProvider.LAMBDA:
            return await self._restart_lambda_node(node)
        elif node.provider == NodeProvider.VAST:
            return await self._restart_vast_node(node)
        elif node.provider == NodeProvider.RUNPOD:
            return await self._restart_runpod_node(node)
        else:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart {node.node_id}: "
                f"unknown provider {node.provider}"
            )
            return False

    async def _soft_restart_p2p(self, node: NodeInfo) -> bool:
        """Restart P2P service only via SSH (not full instance restart).

        Dec 31, 2025: Implements SOFT_RESTART tier for nodes where the instance
        is running but P2P service has crashed or become unresponsive.

        This is preferred over full instance restart when:
        - Node is reachable via SSH
        - Instance is running but P2P port not responding
        - Want to minimize downtime and avoid instance restart delays
        """
        try:
            from app.core.ssh import run_ssh_command_async

            # Get the best IP to use (Tailscale preferred)
            target_ip = node.tailscale_ip or node.public_ip
            if not target_ip:
                logger.warning(
                    f"[NodeRecoveryDaemon] Cannot soft-restart P2P on {node.node_id}: "
                    "no IP address available"
                )
                return False

            # Determine user based on provider
            ssh_user = "ubuntu"
            if node.provider == NodeProvider.VAST:
                ssh_user = "root"
            elif node.provider == NodeProvider.RUNPOD:
                ssh_user = "root"

            # P2P restart command - kills existing process and starts new one
            restart_cmd = (
                "pkill -SIGTERM -f 'python.*p2p_orchestrator' 2>/dev/null; "
                "sleep 2; "
                "cd ~/ringrift/ai-service && "
                f"nohup python scripts/p2p_orchestrator.py --node-id {node.node_id} "
                "> logs/p2p.log 2>&1 &"
            )

            logger.info(
                f"[NodeRecoveryDaemon] Soft-restarting P2P on {node.node_id} "
                f"({target_ip}) via SSH"
            )

            result = await run_ssh_command_async(
                node.node_id,
                restart_cmd,
                timeout=30,
            )

            if result.success:
                logger.info(
                    f"[NodeRecoveryDaemon] P2P soft-restart command sent to {node.node_id}"
                )
                # Wait a bit for P2P to start
                await asyncio.sleep(5)
                return True
            else:
                logger.warning(
                    f"[NodeRecoveryDaemon] P2P soft-restart failed on {node.node_id}: "
                    f"{result.stderr[:200] if result.stderr else 'unknown error'}"
                )
                return False

        except ImportError:
            logger.error(
                "[NodeRecoveryDaemon] app.core.ssh not available for soft-restart"
            )
            return False
        except Exception as e:
            logger.error(
                f"[NodeRecoveryDaemon] P2P soft-restart exception on {node.node_id}: {e}"
            )
            return False

    async def _restart_lambda_node(self, node: NodeInfo) -> bool:
        """Restart a Lambda Cloud instance.

        Uses the Lambda Cloud API instance-operations/restart endpoint.
        API Reference: https://cloud.lambdalabs.com/api/v1/docs
        """
        if not self.config.lambda_api_key:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart Lambda node {node.node_id}: "
                "LAMBDA_API_KEY not set"
            )
            return False

        if not node.instance_id:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart Lambda node {node.node_id}: "
                "no instance_id"
            )
            return False

        try:
            import aiohttp

            if self._http_session is None:
                # Use 60s default timeout for Lambda API operations
                timeout = aiohttp.ClientTimeout(total=60)
                self._http_session = aiohttp.ClientSession(timeout=timeout)

            # Lambda API uses Basic Auth with API key as username
            # The trailing colon indicates no password
            auth = aiohttp.BasicAuth(self.config.lambda_api_key, "")
            headers = {
                "Content-Type": "application/json",
            }

            # Lambda Cloud API restart endpoint
            # POST https://cloud.lambdalabs.com/api/v1/instance-operations/restart
            # Body: {"instance_ids": ["instance-id-here"]}
            url = "https://cloud.lambdalabs.com/api/v1/instance-operations/restart"
            payload = {"instance_ids": [node.instance_id]}

            async with self._http_session.post(
                url, headers=headers, auth=auth, json=payload, timeout=30
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    restarted = result.get("restarted_instances", [])
                    if node.instance_id in [i.get("id") for i in restarted]:
                        logger.info(
                            f"[NodeRecoveryDaemon] Successfully restarted Lambda node "
                            f"{node.node_id} (instance: {node.instance_id})"
                        )
                        return True
                    else:
                        logger.warning(
                            f"[NodeRecoveryDaemon] Lambda restart response OK but instance "
                            f"not in restarted list: {result}"
                        )
                        return True  # API call succeeded, assume restart initiated
                else:
                    body = await resp.text()
                    logger.error(
                        f"[NodeRecoveryDaemon] Failed to restart Lambda node "
                        f"{node.node_id}: {resp.status} - {body}"
                    )
                    return False

        except ImportError:
            logger.error("[NodeRecoveryDaemon] aiohttp not available for API calls")
            return False
        except Exception as e:
            logger.error(f"[NodeRecoveryDaemon] Lambda restart failed: {e}")
            return False

    async def _restart_vast_node(self, node: NodeInfo) -> bool:
        """Restart a Vast.ai instance.

        Uses the VastManager to start/stop instances via vastai CLI.
        The vast_api_key env var is read by vastai CLI directly.
        """
        try:
            from app.providers.vast_manager import VastManager

            manager = VastManager()

            # Extract instance ID from node metadata or node_id
            instance_id = node.metadata.get("instance_id") or node.node_id

            # Check if it's a valid Vast.ai instance ID (numeric)
            if not instance_id or not str(instance_id).isdigit():
                # Try to find by host match
                instances = await manager.list_instances()
                for inst in instances:
                    if node.host in [inst.public_ip, inst.metadata.get("ssh_host", "")]:
                        instance_id = inst.instance_id
                        break

            if not instance_id:
                logger.warning(
                    f"[NodeRecoveryDaemon] Cannot find Vast instance for {node.node_id}"
                )
                return False

            logger.info(
                f"[NodeRecoveryDaemon] Restarting Vast.ai instance {instance_id} "
                f"for node {node.node_id}"
            )

            # Stop then start for full restart
            stop_success = await manager.stop_instance(str(instance_id))
            if not stop_success:
                logger.warning(
                    f"[NodeRecoveryDaemon] Failed to stop Vast instance {instance_id}, "
                    "attempting start anyway"
                )

            # Wait briefly for stop to complete
            await asyncio.sleep(5)

            # Start the instance
            start_success = await manager.start_instance(str(instance_id))
            if start_success:
                logger.info(
                    f"[NodeRecoveryDaemon] Successfully restarted Vast instance {instance_id}"
                )
                return True
            else:
                logger.error(
                    f"[NodeRecoveryDaemon] Failed to start Vast instance {instance_id}"
                )
                return False

        except ImportError:
            logger.warning(
                "[NodeRecoveryDaemon] VastManager not available for Vast restart"
            )
            return False
        except Exception as e:
            logger.error(f"[NodeRecoveryDaemon] Vast restart error: {e}")
            return False

    async def _restart_runpod_node(self, node: NodeInfo) -> bool:
        """Restart a RunPod instance.

        December 2025: Implements RunPod API integration for node recovery.
        Uses the RunPod REST API to stop and start pods.

        API Reference: https://docs.runpod.io/reference/api-reference
        """
        if not self.config.runpod_api_key:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart RunPod node {node.node_id}: "
                "RUNPOD_API_KEY not set"
            )
            return False

        try:
            import aiohttp

            # Extract pod ID from node metadata or node_id
            # RunPod pod IDs are typically in format like "abc123xyz"
            pod_id = node.metadata.get("pod_id") or node.metadata.get("runpod_id")

            if not pod_id:
                # Try to extract from node_id if it contains runpod pattern
                if "runpod-" in node.node_id.lower():
                    # Format: runpod-abc123xyz or similar
                    pod_id = node.node_id.lower().replace("runpod-", "").split("-")[0]
                else:
                    pod_id = node.node_id

            logger.info(
                f"[NodeRecoveryDaemon] Attempting to restart RunPod pod {pod_id} "
                f"for node {node.node_id}"
            )

            headers = {
                "Authorization": f"Bearer {self.config.runpod_api_key}",
                "Content-Type": "application/json",
            }

            base_url = "https://api.runpod.io/v2"

            async with aiohttp.ClientSession() as session:
                # First, check pod status
                async with session.get(
                    f"{base_url}/pods/{pod_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 404:
                        logger.warning(
                            f"[NodeRecoveryDaemon] RunPod pod {pod_id} not found"
                        )
                        return False
                    elif resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(
                            f"[NodeRecoveryDaemon] RunPod API error: {resp.status} - {error_text}"
                        )
                        return False

                    pod_data = await resp.json()
                    current_status = pod_data.get("status", "unknown")
                    logger.debug(f"[NodeRecoveryDaemon] Pod {pod_id} status: {current_status}")

                # Stop the pod
                async with session.post(
                    f"{base_url}/pods/{pod_id}/stop",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status not in (200, 202, 204):
                        error_text = await resp.text()
                        logger.warning(
                            f"[NodeRecoveryDaemon] Failed to stop RunPod pod: "
                            f"{resp.status} - {error_text}"
                        )
                        # Continue anyway - pod might already be stopped

                # Wait for stop to complete
                logger.debug(f"[NodeRecoveryDaemon] Waiting for pod {pod_id} to stop...")
                await asyncio.sleep(10)

                # Start the pod
                async with session.post(
                    f"{base_url}/pods/{pod_id}/start",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in (200, 202):
                        logger.info(
                            f"[NodeRecoveryDaemon] Successfully restarted RunPod pod {pod_id}"
                        )
                        return True
                    else:
                        error_text = await resp.text()
                        logger.error(
                            f"[NodeRecoveryDaemon] Failed to start RunPod pod: "
                            f"{resp.status} - {error_text}"
                        )
                        return False

        except ImportError:
            logger.warning(
                "[NodeRecoveryDaemon] aiohttp not available for RunPod API calls"
            )
            return False
        except asyncio.TimeoutError:
            logger.error(f"[NodeRecoveryDaemon] RunPod API timeout for pod {pod_id}")
            return False
        except Exception as e:
            logger.error(f"[NodeRecoveryDaemon] RunPod restart error: {e}")
            return False

    async def _verify_node_health_after_restart(
        self,
        node: NodeInfo,
        poll_intervals: list[float] | None = None,
    ) -> bool:
        """Verify node health after restart by polling at specified intervals.

        P0.5 Dec 2025: Ensures restart actually succeeded by verifying node
        becomes reachable after restart API call completes.

        Args:
            node: The node to verify
            poll_intervals: List of seconds to wait between polls.
                           Default: [10, 20, 30] (poll at 10s, 30s, 60s after restart)

        Returns:
            True if node becomes healthy within the poll window, False otherwise.
        """
        if poll_intervals is None:
            poll_intervals = [10.0, 20.0, 30.0]

        logger.info(
            f"[NodeRecoveryDaemon] Verifying health of {node.node_id} "
            f"after restart (intervals: {poll_intervals}s)"
        )

        for i, interval in enumerate(poll_intervals):
            logger.debug(
                f"[NodeRecoveryDaemon] Waiting {interval}s before health check {i + 1}"
            )
            await asyncio.sleep(interval)

            # Check node health via P2P status or direct SSH
            is_healthy = await self._check_single_node_health(node)

            if is_healthy:
                total_wait = sum(poll_intervals[:i + 1])
                logger.info(
                    f"[NodeRecoveryDaemon] Node {node.node_id} verified healthy "
                    f"after {total_wait:.0f}s"
                )
                # Reset failure state on successful verification
                node.status = "running"
                node.consecutive_failures = 0
                return True

            logger.debug(
                f"[NodeRecoveryDaemon] Node {node.node_id} not yet healthy "
                f"after {sum(poll_intervals[:i + 1]):.0f}s"
            )

        # All polls failed
        total_wait = sum(poll_intervals)
        logger.warning(
            f"[NodeRecoveryDaemon] Node {node.node_id} failed health verification "
            f"after {total_wait:.0f}s"
        )
        return False

    async def _check_single_node_health(self, node: NodeInfo) -> bool:
        """Check if a single node is healthy via P2P or SSH.

        P0.5 Dec 2025: Helper for post-restart verification.

        Args:
            node: The node to check

        Returns:
            True if node responds to health check, False otherwise.
        """
        # Try P2P status first
        try:
            from app.coordination.p2p_integration import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is not None:
                status = await p2p.get_status()
                if status:
                    alive_peers = status.get("alive_peers", [])
                    if isinstance(alive_peers, list):
                        for peer in alive_peers:
                            if isinstance(peer, dict):
                                if peer.get("node_id") == node.node_id:
                                    return True
                            elif isinstance(peer, str) and peer == node.node_id:
                                return True
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[NodeRecoveryDaemon] P2P check failed: {e}")

        # Fallback: Try SSH health check
        if node.host:
            try:
                from app.core.ssh import run_ssh_command_async

                result = await asyncio.wait_for(
                    run_ssh_command_async(
                        node.node_id,  # Uses cluster config for SSH details
                        "echo healthy",
                        timeout=10,
                    ),
                    timeout=15.0,
                )
                if result and result.success and "healthy" in result.stdout:
                    return True
            except (ImportError, asyncio.TimeoutError, RuntimeError, OSError) as e:
                logger.debug(f"[NodeRecoveryDaemon] SSH health check failed: {e}")

        return False

    def _emit_recovery_event(
        self,
        node: NodeInfo,
        action: RecoveryAction,
        success: bool,
    ) -> None:
        """Emit event for recovery action."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            # Phase 22.2 fix: Use publish_sync instead of emit (which doesn't exist)
            router.publish_sync(
                "node_recovery_triggered",
                {
                    "node_id": node.node_id,
                    "provider": node.provider.value,
                    "action": action.value,
                    "success": success,
                    "consecutive_failures": node.consecutive_failures,
                    "timestamp": time.time(),
                },
                source="node_recovery_daemon",
            )
        except Exception as e:
            logger.debug(f"Could not publish recovery event: {e}")

        # Emit standard health events
        self._emit_health_event(node, action, success)

    def _emit_health_event(
        self,
        node: NodeInfo,
        action: RecoveryAction,
        success: bool,
    ) -> None:
        """Emit standard cluster health events for coordination."""
        import asyncio

        async def _emit():
            try:
                if success and action in (RecoveryAction.RESTART, RecoveryAction.PREEMPTIVE_RESTART):
                    # Node was successfully recovered
                    await emit_node_recovered(
                        node_id=node.node_id,
                        node_ip=node.host,
                        recovery_time_seconds=0.0,  # Could track this if needed
                        source="node_recovery_daemon",
                    )
                elif not success or node.status in ("terminated", "failed", "unreachable"):
                    # Node is unhealthy
                    await emit_node_unhealthy(
                        node_id=node.node_id,
                        reason=f"Status: {node.status}, Action: {action.value}",
                        node_ip=node.host,
                        consecutive_failures=node.consecutive_failures,
                        source="node_recovery_daemon",
                    )
            except Exception as e:
                logger.debug(f"Could not emit health event: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_emit())
        except RuntimeError:
            # No running loop - try to run directly
            try:
                asyncio.run(_emit())
            except RuntimeError as e:
                logger.debug(f"Failed to run recovery event emission (nested loop?): {e}")
            except (OSError, IOError) as e:
                logger.debug(f"Failed to emit recovery event (I/O error): {e}")

    def health_check(self) -> HealthCheckResult:
        """Check if the daemon is healthy.

        Returns HealthCheckResult for protocol compliance.
        Used by DaemonManager for crash detection and auto-restart.

        December 2025: Sync method for HandlerBase pattern.
        """
        is_healthy = True
        message = "Node recovery daemon running"
        status = CoordinatorStatus.RUNNING

        # Check if daemon is running
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Node recovery daemon not running",
                details={"running": False},
            )

        # Check if we have recent check data (within 2x check interval)
        if self._recovery_stats.last_job_time > 0:
            max_age = self.config.check_interval_seconds * 2
            age = time.time() - self._recovery_stats.last_job_time
            if age > max_age:
                is_healthy = False
                status = CoordinatorStatus.DEGRADED
                message = f"Stale check data (age={age:.0f}s, max={max_age}s)"
                logger.warning(f"[{self.name}] Health check: {message}")

        # Check for excessive failures
        if is_healthy and self._recovery_stats.recovery_failures > 10:
            # Allow if we also have successes (not just all failures)
            if self._recovery_stats.nodes_recovered == 0:
                is_healthy = False
                status = CoordinatorStatus.DEGRADED
                message = f"Only failures, no recoveries (failures={self._recovery_stats.recovery_failures})"
                logger.warning(f"[{self.name}] Health check: {message}")

        # Check HTTP session health if used
        if is_healthy and self._http_session is not None and self._http_session.closed:
            is_healthy = False
            status = CoordinatorStatus.DEGRADED
            message = "HTTP session closed unexpectedly"
            logger.warning(f"[{self.name}] Health check: {message}")

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details={
                "running": self._running,
                "cycles_completed": self._stats.cycles_completed,
                "nodes_recovered": self._recovery_stats.nodes_recovered,
                "recovery_failures": self._recovery_stats.recovery_failures,
                "last_job_time": self._recovery_stats.last_job_time,
                "error_count": self._stats.errors_count,
            },
        )

    def get_node_states(self) -> dict[str, dict[str, Any]]:
        """Get current node states."""
        return {
            node_id: {
                "node_id": node.node_id,
                "host": node.host,
                "provider": node.provider.value,
                "status": node.status,
                "consecutive_failures": node.consecutive_failures,
                "last_seen": node.last_seen,
            }
            for node_id, node in self._node_states.items()
        }

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring.

        December 2025: Updated for HandlerBase pattern.
        Uses HandlerStats attributes (last_activity, errors_count).
        """
        from datetime import datetime

        # Convert last_activity timestamp to ISO string if set
        last_activity_iso = None
        if self._stats.last_activity > 0:
            last_activity_iso = datetime.fromtimestamp(self._stats.last_activity).isoformat()

        return {
            "running": self._running,
            "config": {
                "enabled": self.config.enabled,
                "check_interval_seconds": self.config.check_interval_seconds,
                "preemptive_recovery_enabled": self.config.preemptive_recovery_enabled,
            },
            "stats": {
                "cycles_completed": self._stats.cycles_completed,
                "last_activity": last_activity_iso,
                "error_count": self._stats.errors_count,
                "events_processed": self._stats.events_processed,
            },
            "recovery_stats": {
                "total_checks": self._recovery_stats.total_checks,
                "nodes_recovered": self._recovery_stats.nodes_recovered,
                "recovery_failures": self._recovery_stats.recovery_failures,
                "preemptive_recoveries": self._recovery_stats.preemptive_recoveries,
                "last_check_time": self._recovery_stats.last_job_time,
                "last_error": self._recovery_stats.last_error,
            },
            "tracked_nodes": len(self._node_states),
            "nodes": self.get_node_states(),
        }


# =============================================================================
# Singleton Access (using HandlerBase class methods)
# =============================================================================


def get_node_recovery_daemon() -> NodeRecoveryDaemon:
    """Get or create the singleton NodeRecoveryDaemon instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return NodeRecoveryDaemon.get_instance()


def reset_node_recovery_daemon() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    NodeRecoveryDaemon.reset_instance()

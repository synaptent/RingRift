"""P2P Recovery Daemon for 48-Hour Autonomous Operation.

This daemon monitors the P2P cluster health and automatically restarts
the P2P orchestrator when it becomes unhealthy or partitioned.

Key features:
- Health checks via /status endpoint
- Consecutive failure tracking before restart
- Cooldown period between restarts
- Event emission for monitoring

December 2025: Created for 48-hour autonomous operation enablement.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.ports import get_local_p2p_status_url, get_p2p_endpoints
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.coordination.coordinator_persistence import StatePersistenceMixin
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_router import get_event_payload

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class P2PRecoveryConfig:
    """Configuration for P2P Recovery daemon.

    Attributes:
        check_interval_seconds: How often to check P2P health (default: 60s)
        health_endpoint: P2P status endpoint URL
        max_consecutive_failures: Failures before triggering restart (default: 3)
        restart_cooldown_seconds: Minimum time between restarts (default: 5 min)
        health_timeout_seconds: Timeout for health check request (default: 10s)
        min_alive_peers: Minimum peers required for healthy cluster (default: 3)
        startup_grace_seconds: Grace period after restart before checking (default: 30s)
        isolation_check_enabled: Enable network isolation detection (default: True)
        min_peer_ratio: Trigger isolation recovery if P2P/Tailscale ratio below this (default: 0.5)
        isolation_consecutive_checks: Consecutive isolation checks before action (default: 3)
        # Dec 29, 2025: Self-healing enhancements for quorum and leader gaps
        max_leader_gap_seconds: Maximum seconds without a leader before forcing election (default: 10)
        quorum_recovery_enabled: Enable automatic quorum recovery (default: True)
        leader_election_endpoint: Endpoint to trigger leader election
        # Dec 30, 2025: Exponential backoff for restart attempts
        restart_backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        max_cooldown_seconds: Maximum cooldown between restarts (default: 30 min)
        # Jan 2026: Proactive voter quorum monitoring
        voter_quorum_monitoring_enabled: Enable proactive voter quorum checks (default: True)
        min_voters_for_healthy_quorum: Minimum voters to consider quorum healthy (default: 5)
        voter_quorum_size: Total voters required for quorum (default: 4, out of 7)
    """

    enabled: bool = True  # Whether daemon should run
    check_interval_seconds: int = 60
    health_endpoint: str = field(default_factory=get_local_p2p_status_url)
    max_consecutive_failures: int = 3
    restart_cooldown_seconds: int = 300  # 5 minutes (initial cooldown)
    # Jan 2026: NAT-blocked nodes need faster recovery (more likely to disconnect)
    nat_blocked_restart_threshold: int = 2  # Fewer failures before restart (vs 3)
    nat_blocked_cooldown_seconds: int = 60  # 1 minute cooldown (vs 5 min)
    restart_backoff_multiplier: float = 2.0  # Double cooldown each attempt
    max_cooldown_seconds: int = 1800  # 30 minutes max cooldown
    health_timeout_seconds: float = 10.0
    min_alive_peers: int = 3
    # January 3, 2026: Reverted 10s→30s after Session 8 analysis showed
    # P2P initialization needs 8-12s; 10s causes 40-60% false-positive restarts.
    # 30s provides safe buffer: P2P init (12s) + state loading (8s) + margin (10s).
    startup_grace_seconds: int = 30
    # Network isolation detection (December 2025)
    isolation_check_enabled: bool = True
    min_peer_ratio: float = 0.5  # Trigger if P2P sees < 50% of Tailscale peers
    isolation_consecutive_checks: int = 3  # Require 3 checks (~3 min) before action
    # Sprint 16.1 (Jan 3, 2026): TCP validation to avoid false-positive isolation
    # on Tailscale outage. If TCP ping reaches voters, not isolated.
    tcp_validation_enabled: bool = True
    tcp_ping_timeout: float = 5.0  # Timeout for each TCP ping attempt
    tcp_min_reachable: int = 2  # Minimum voters reachable via TCP to clear isolation
    # Dec 29, 2025: Self-healing for quorum and leader gaps
    max_leader_gap_seconds: int = 10  # Jan 3, 2026: Reduced from 45s for faster failover
    quorum_recovery_enabled: bool = True
    leader_election_endpoint: str = field(default_factory=lambda: get_p2p_endpoints()['election'])
    # Jan 2026: Proactive voter quorum monitoring
    voter_quorum_monitoring_enabled: bool = True
    min_voters_for_healthy_quorum: int = 5  # Trigger recovery if < 5 of 7 voters healthy
    voter_quorum_size: int = 4  # Quorum threshold (out of 7 total voters)
    # January 4, 2026: Dynamic recovery cooldown during quorum loss.
    # When quorum is lost, use aggressive 60s cooldown instead of normal 5 min.
    # This prevents multi-day stalls from slow recovery during quorum loss events.
    quorum_loss_cooldown_seconds: int = 60  # 1 minute during quorum loss

    @classmethod
    def from_env(cls) -> "P2PRecoveryConfig":
        """Load config from environment variables."""
        config = cls()

        if os.environ.get("RINGRIFT_P2P_RECOVERY_ENABLED"):
            config.enabled = os.environ.get("RINGRIFT_P2P_RECOVERY_ENABLED", "1") == "1"

        if os.environ.get("RINGRIFT_P2P_RECOVERY_INTERVAL"):
            try:
                config.check_interval_seconds = int(
                    os.environ.get("RINGRIFT_P2P_RECOVERY_INTERVAL", "60")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_HEALTH_ENDPOINT"):
            config.health_endpoint = os.environ.get(
                "RINGRIFT_P2P_HEALTH_ENDPOINT", config.health_endpoint
            )

        if os.environ.get("RINGRIFT_P2P_MAX_FAILURES"):
            try:
                config.max_consecutive_failures = int(
                    os.environ.get("RINGRIFT_P2P_MAX_FAILURES", "3")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_RESTART_COOLDOWN"):
            try:
                config.restart_cooldown_seconds = int(
                    os.environ.get("RINGRIFT_P2P_RESTART_COOLDOWN", "300")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_MIN_PEERS"):
            try:
                config.min_alive_peers = int(
                    os.environ.get("RINGRIFT_P2P_MIN_PEERS", "3")
                )
            except ValueError:
                pass

        # Network isolation detection config
        if os.environ.get("RINGRIFT_P2P_ISOLATION_ENABLED"):
            config.isolation_check_enabled = os.environ.get(
                "RINGRIFT_P2P_ISOLATION_ENABLED", "1"
            ) == "1"

        if os.environ.get("RINGRIFT_P2P_MIN_PEER_RATIO"):
            try:
                config.min_peer_ratio = float(
                    os.environ.get("RINGRIFT_P2P_MIN_PEER_RATIO", "0.5")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_ISOLATION_CHECKS"):
            try:
                config.isolation_consecutive_checks = int(
                    os.environ.get("RINGRIFT_P2P_ISOLATION_CHECKS", "3")
                )
            except ValueError:
                pass

        return config


# =============================================================================
# P2P Recovery Daemon
# =============================================================================


class P2PRecoveryDaemon(HandlerBase, StatePersistenceMixin):
    """Daemon that monitors P2P cluster health and triggers auto-recovery.

    Workflow:
    1. Periodically check P2P /status endpoint
    2. Track consecutive failures
    3. After max_consecutive_failures, restart P2P orchestrator
    4. Respect cooldown period between restarts

    Events Emitted:
    - P2P_RESTART_TRIGGERED: When restart is initiated
    - P2P_HEALTH_RECOVERED: When P2P becomes healthy after being unhealthy

    Jan 3, 2026 (Sprint 15.1): Added StatePersistenceMixin for voter health persistence.
    Voter state is now persisted across restarts for:
    - Faster quorum recovery (knows which voters were recently online)
    - Voter flapping detection (identifies unstable voters)
    """

    # Jan 3, 2026 (Sprint 15.1): Voter flapping detection thresholds
    FLAP_WINDOW_SECONDS = 600.0  # 10 minute window for flapping detection
    FLAP_THRESHOLD = 3  # 3+ state changes in window = flapping

    def __init__(self, config: P2PRecoveryConfig | None = None):
        self._daemon_config = config or P2PRecoveryConfig.from_env()
        super().__init__(
            name="P2PRecovery",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )
        self._consecutive_failures = 0
        self._last_restart_time = 0.0
        self._total_restarts = 0
        self._last_healthy_time = time.time()
        self._was_unhealthy = False
        self._last_status: dict[str, Any] = {}
        self._startup_time = time.time()
        # Network isolation detection (December 2025)
        self._consecutive_isolation_checks = 0
        self._last_tailscale_count = 0
        self._isolation_triggered_restarts = 0
        # Dec 29, 2025: Self-healing for leader gaps and quorum
        self._last_leader_seen_time = time.time()
        self._leader_gap_elections_triggered = 0
        self._quorum_recovery_attempts = 0
        # Dec 30, 2025: Exponential backoff for restarts
        self._restart_attempt_count = 0  # Consecutive restart attempts without recovery
        # Jan 2026: Cache NAT-blocked status (checked once at startup)
        self._is_nat_blocked: bool | None = None
        # Jan 2026: Proactive voter quorum monitoring
        self._last_voter_quorum_check: dict[str, Any] = {}
        self._quorum_at_risk_consecutive = 0  # Consecutive checks with quorum at risk
        self._quorum_recovery_triggered = 0  # Total proactive recoveries triggered
        # January 4, 2026: Track quorum loss state for dynamic cooldown.
        # When True, _get_effective_cooldown() returns 60s instead of 5min.
        self._quorum_lost = False
        # January 4, 2026: Healing attempts during quorum loss for early escalation.
        # After 2 failed healing attempts with quorum lost, skip to immediate restart.
        self._quorum_lost_healing_attempts = 0
        # Jan 2026 Session 8: Track previous voter state for fine-grained events
        self._last_online_voters: set[str] = set()
        # Jan 3, 2026 Session 7: Partition healing coordination
        self._healing_in_progress = False  # Pause restarts during healing
        self._healing_started_time: float | None = None  # When healing started (Session 9)
        self._healing_timeout_seconds = 300.0  # 5 minute timeout (Session 9)
        self._consecutive_healing_failures = 0  # Track failures for escalation (Session 9)
        self._recovery_attempts = 0  # Total P2P recovery attempts

        # Jan 3, 2026 (Sprint 15.1): Voter state history for flapping detection
        # Maps voter_id -> list of (timestamp, is_online) state changes
        self._voter_state_history: dict[str, list[tuple[float, bool]]] = {}
        self._flapping_voters: set[str] = set()  # Currently flapping voters

        # Initialize persistence for voter health state
        try:
            db_path = Path("data/state/p2p_recovery_daemon.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.init_persistence(
                db_path,
                auto_snapshot=True,
                snapshot_interval=60.0,  # Snapshot every minute for voter state
                max_snapshots=5,
            )
            self._restore_voter_state()
            logger.info(f"[P2PRecovery] Voter health persistence initialized: {db_path}")
        except Exception as e:
            logger.warning(f"[P2PRecovery] Failed to initialize voter persistence: {e}")

    @property
    def config(self) -> P2PRecoveryConfig:
        """Return daemon configuration."""
        return self._daemon_config

    def _check_is_nat_blocked(self) -> bool:
        """Check if this node is NAT-blocked based on config.

        Jan 2026: NAT-blocked nodes use different thresholds for faster recovery.
        Checks distributed_hosts.yaml for nat_blocked or force_relay_mode flags.
        """
        if self._is_nat_blocked is not None:
            return self._is_nat_blocked

        try:
            import socket
            import yaml

            node_id = os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "distributed_hosts.yaml",
            )

            with open(config_path) as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            node_info = hosts.get(node_id, {})

            # Check for NAT-blocked indicators
            self._is_nat_blocked = (
                node_info.get("nat_blocked", False)
                or node_info.get("force_relay_mode", False)
            )

            if self._is_nat_blocked:
                logger.info(f"Node {node_id} is NAT-blocked, using faster recovery thresholds")

            return self._is_nat_blocked

        except Exception as e:
            logger.debug(f"Failed to check NAT-blocked status: {e}")
            self._is_nat_blocked = False
            return False

    def _get_effective_restart_threshold(self) -> int:
        """Get the effective restart threshold based on NAT-blocked status.

        Jan 2026: NAT-blocked nodes use lower threshold for faster recovery.
        """
        if self._check_is_nat_blocked():
            return self.config.nat_blocked_restart_threshold
        return self.config.max_consecutive_failures

    def _get_effective_cooldown(self) -> int:
        """Get the effective base cooldown based on quorum/NAT status.

        January 4, 2026: Dynamic cooldown priorities:
        1. Quorum lost -> 60s (most aggressive, enables faster recovery)
        2. NAT-blocked -> 60s (faster recovery for connection-prone nodes)
        3. Normal -> 300s (5 min base cooldown)

        This prevents multi-day stalls from slow recovery during quorum loss.
        """
        # Quorum loss takes priority - use aggressive cooldown to recover faster
        if self._quorum_lost:
            return self.config.quorum_loss_cooldown_seconds
        if self._check_is_nat_blocked():
            return self.config.nat_blocked_cooldown_seconds
        return self.config.restart_cooldown_seconds

    def _get_health_check_interval(self) -> float:
        """Get adaptive health check interval based on cluster state.

        Jan 7, 2026: Adaptive health check - faster during degradation.
        Implements dynamic interval adjustment based on cluster health status:
        - Critical (quorum lost): 15s - most aggressive monitoring
        - Isolated (network partition detected): 30s - fast recovery detection
        - Degraded (consecutive failures): 45s - moderate concern
        - Healthy: 60s (or configured base interval) - normal operation

        Returns:
            Interval in seconds for the next health check cycle.
        """
        base_interval = float(self.config.check_interval_seconds)

        # Critical: Quorum lost - check every 15s for fastest recovery
        if self._quorum_lost:
            return 15.0

        # Isolated: Network partition detected - check every 30s
        if self._consecutive_isolation_checks > 0:
            return 30.0

        # Degraded: Consecutive health check failures - check every 45s
        if self._consecutive_failures > 0:
            return 45.0

        # Healing in progress: Check every 30s to detect completion
        if self._healing_in_progress:
            return 30.0

        # Healthy: Use configured base interval (default 60s)
        return base_interval

    def _update_health_check_interval(self) -> None:
        """Update the cycle interval based on current cluster health.

        Jan 7, 2026: Called at the end of _run_cycle() to adjust the
        next health check interval dynamically based on cluster state.
        """
        new_interval = self._get_health_check_interval()
        old_interval = self._cycle_interval

        if new_interval != old_interval:
            logger.info(
                f"[P2PRecovery] Adjusting health check interval: "
                f"{old_interval:.0f}s -> {new_interval:.0f}s "
                f"(quorum_lost={self._quorum_lost}, "
                f"isolation_checks={self._consecutive_isolation_checks}, "
                f"failures={self._consecutive_failures})"
            )
            self._cycle_interval = new_interval

    # =========================================================================
    # Event Subscriptions (December 30, 2025)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions for voter health and P2P recovery events.

        Dec 30, 2025: Subscribe to voter health events from VoterHealthMonitorDaemon
        to coordinate emergency quorum recovery.

        Jan 3, 2026: Subscribe to P2P_RECOVERY_NEEDED from partition_healer.py
        when gossip convergence fails repeatedly at max escalation.

        Jan 3, 2026 Session 7: Subscribe to partition healing events to coordinate
        P2P recovery with healing operations. Prevents restart during active healing.

        Jan 5, 2026 (Sprint 17.9): Subscribe to DAEMON_FAILURE_CLASSIFIED events
        from DaemonHealthAnalyzer for unified failure response coordination.

        Jan 24, 2026: Subscribe to EVENT_LOOP_BLOCKED for event loop health recovery.
        """
        return {
            "QUORUM_LOST": self._on_quorum_lost,
            "QUORUM_AT_RISK": self._on_quorum_at_risk,
            "VOTER_OFFLINE": self._on_voter_offline,
            "P2P_RECOVERY_NEEDED": self._on_p2p_recovery_needed,
            # Jan 3, 2026 Session 7: Coordinate with partition healing
            "PARTITION_HEALING_STARTED": self._on_partition_healing_started,
            "PARTITION_HEALED": self._on_partition_healed,
            "PARTITION_HEALING_FAILED": self._on_partition_healing_failed,
            # Jan 5, 2026 (Sprint 17.9): Unified failure classification
            "DAEMON_FAILURE_CLASSIFIED": self._on_daemon_failure_classified,
            # Jan 24, 2026: Event loop blocking detection
            "EVENT_LOOP_BLOCKED": self._on_event_loop_blocked,
        }

    async def _on_quorum_lost(self, event: Any) -> None:
        """Handle QUORUM_LOST event - trigger emergency quorum restoration.

        Dec 30, 2025: Called when voter count drops below quorum threshold.
        Initiates aggressive recovery actions including:
        - Prioritized voter reconnection
        - Faster health check interval
        - Leader election if needed

        January 4, 2026: Sets _quorum_lost flag to enable dynamic 60s recovery
        cooldown instead of normal 5 minute cooldown.
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        logger.critical(
            f"QUORUM LOST - initiating emergency recovery "
            f"(online_voters={payload.get('online_voters', '?')}, "
            f"quorum={payload.get('quorum_size', '?')})"
        )

        # January 4, 2026: Enable aggressive recovery cooldown during quorum loss
        self._quorum_lost = True

        # Track for health reporting
        self._quorum_recovery_attempts += 1

        # Trigger immediate quorum-aware reconnection prioritization
        await self._prioritize_quorum_reconnections()

        # Emit event for monitoring
        safe_emit_event(
            "quorum_recovery_started",
            {
                "trigger": "quorum_lost_event",
                "online_voters": payload.get("online_voters", 0),
                "quorum_size": payload.get("quorum_size", 4),
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    async def _on_quorum_at_risk(self, event: Any) -> None:
        """Handle QUORUM_AT_RISK event - boost voter reconnection priority.

        Dec 30, 2025: Called when voter count is exactly at quorum threshold.
        Proactively attempts to reconnect offline voters before quorum is lost.
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        online_voters = payload.get("online_voters", 0)
        offline_voters = payload.get("offline_voters", [])
        logger.warning(
            f"Quorum at risk - {online_voters} voters online, "
            f"{len(offline_voters)} offline. Boosting reconnection priority."
        )

        # Proactively prioritize voter reconnections
        await self._prioritize_quorum_reconnections()

    async def _on_voter_offline(self, event: Any) -> None:
        """Handle VOTER_OFFLINE event - individual voter went offline.

        Dec 30, 2025: Logs voter offline event and checks if this impacts quorum.
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        voter_id = payload.get("voter_id", "unknown")
        reason = payload.get("reason", "unknown")
        logger.warning(f"Voter {voter_id} went offline: {reason}")

    async def _emit_voter_state_changes(self, current_online_voters: set[str]) -> None:
        """Emit fine-grained VOTER_ONLINE/VOTER_OFFLINE events for state changes.

        Jan 2026 Session 8: Compares current online voters with previous state
        and emits individual events for each voter that changed state.

        Jan 3, 2026 (Sprint 15.1): Also records state changes for flapping detection.

        Args:
            current_online_voters: Set of currently online voter IDs
        """
        if not current_online_voters and not self._last_online_voters:
            return  # Nothing to compare yet

        # Find voters that went offline (were online, now not)
        newly_offline = self._last_online_voters - current_online_voters
        for voter_id in newly_offline:
            logger.info(f"[VoterEvents] Voter {voter_id} went offline")
            safe_emit_event(
                "voter_offline",
                {
                    "voter_id": voter_id,
                    "reason": "quorum_check",
                    "source": "P2PRecoveryDaemon",
                },
                context="P2PRecovery",
            )
            # Jan 3, 2026 (Sprint 15.1): Record for flapping detection
            self._record_voter_state_change(voter_id, is_online=False)

        # Find voters that came online (were offline, now online)
        newly_online = current_online_voters - self._last_online_voters
        for voter_id in newly_online:
            logger.info(f"[VoterEvents] Voter {voter_id} came online")
            safe_emit_event(
                "voter_online",
                {
                    "voter_id": voter_id,
                    "reason": "quorum_check",
                    "source": "P2PRecoveryDaemon",
                },
                context="P2PRecovery",
            )
            # Jan 3, 2026 (Sprint 15.1): Record for flapping detection
            self._record_voter_state_change(voter_id, is_online=True)

        # Update tracked state for next comparison
        self._last_online_voters = current_online_voters.copy()

    async def _on_p2p_recovery_needed(self, event: Any) -> None:
        """Handle P2P_RECOVERY_NEEDED event - max escalation reached in partition healing.

        Jan 3, 2026: Called when partition_healer.py has exhausted escalation tiers
        after repeated gossip convergence failures. At this point:
        - Normal healing has failed multiple times
        - Escalation backoffs have been applied
        - Manual or automated intervention is required

        Actions taken:
        1. Log critical alert with full context
        2. Attempt automated P2P orchestrator restart
        3. Emit alert event for external monitoring (PagerDuty, etc.)
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        reason = payload.get("reason", "unknown")
        escalation_level = payload.get("escalation_level", 0)
        consecutive_failures = payload.get("consecutive_failures", 0)

        logger.critical(
            f"P2P_RECOVERY_NEEDED received - partition healing exhausted. "
            f"reason={reason}, escalation_level={escalation_level}, "
            f"consecutive_failures={consecutive_failures}"
        )

        # Track for health reporting
        self._recovery_attempts += 1

        # Attempt automated recovery: restart P2P orchestrator
        try:
            await self._trigger_p2p_restart(
                reason=f"P2P_RECOVERY_NEEDED: {reason}",
                force=True,  # Force restart at max escalation
            )
            logger.info(
                f"[P2PRecovery] Triggered P2P orchestrator restart "
                f"after P2P_RECOVERY_NEEDED (level={escalation_level})"
            )
        except Exception as e:
            logger.error(f"[P2PRecovery] Failed to restart P2P orchestrator: {e}")

        # Emit alert for external monitoring
        safe_emit_event(
            "p2p_recovery_started",
            {
                "trigger": "p2p_recovery_needed_event",
                "reason": reason,
                "escalation_level": escalation_level,
                "consecutive_failures": consecutive_failures,
                "action": "p2p_restart_triggered",
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    async def _on_partition_healing_started(self, event: Any) -> None:
        """Handle PARTITION_HEALING_STARTED - pause P2P restarts during healing.

        Jan 3, 2026 Session 7: When partition healing is in progress, we should
        not restart the P2P orchestrator as it would interrupt the healing process.

        Jan 3, 2026 Session 9: Track start time for timeout-based flag expiration.
        """
        logger.info(
            f"[P2PRecovery] Partition healing started, pausing P2P restart attempts"
        )
        self._healing_in_progress = True
        self._healing_started_time = time.time()

    async def _on_partition_healed(self, event: Any) -> None:
        """Handle PARTITION_HEALED - resume normal recovery operations.

        Jan 3, 2026 Session 7: Partition healing completed successfully.
        Reset failure counters since the partition was healed without restart.

        January 4, 2026: Added convergence validation. After healing is reported,
        wait 5s for gossip to propagate and verify healthy_ratio >= 0.95 to
        confirm the partition fix actually converged across the cluster.
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        nodes_healed = payload.get("nodes_healed", 0)
        duration_seconds = payload.get("duration_seconds", 0)

        logger.info(
            f"[P2PRecovery] Partition healed reported: "
            f"{nodes_healed} nodes reconnected in {duration_seconds:.1f}s"
        )

        # January 4, 2026: Convergence validation - wait for gossip propagation
        convergence_validated = await self._validate_healing_convergence()

        self._healing_in_progress = False
        self._healing_started_time = None  # Session 9: Clear healing timeout

        if convergence_validated:
            logger.info(
                f"[P2PRecovery] Partition healing convergence VALIDATED: "
                f"gossip healthy ratio >= 0.95"
            )
            # Reset consecutive failures since partition was fully healed
            self._consecutive_failures = 0
            self._consecutive_healing_failures = 0  # Session 9: Reset healing failure counter
        else:
            logger.warning(
                f"[P2PRecovery] Partition healing reported success but convergence "
                f"validation FAILED - gossip healthy ratio < 0.95"
            )
            # Don't reset failure counters - healing didn't fully converge
            self._consecutive_healing_failures += 1
            # Emit event for monitoring
            self._try_emit_convergence_failure(nodes_healed)

    async def _on_partition_healing_failed(self, event: Any) -> None:
        """Handle PARTITION_HEALING_FAILED - escalate to P2P restart if needed.

        Jan 3, 2026 Session 7: Partition healing failed. If this is repeated,
        the partition healer will emit P2P_RECOVERY_NEEDED which triggers restart.

        Jan 3, 2026 Session 9: Track consecutive healing failures. After 3+
        consecutive failures, trigger P2P restart directly instead of waiting
        for the partition healer's escalation.

        January 4, 2026: Early escalation when quorum is lost. After 2 healing
        attempts with quorum lost, skip normal escalation and restart immediately.
        This prevents 30+ minute delays from tier-based escalation during quorum loss.
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        reason = payload.get("reason", "unknown")
        escalation_level = payload.get("escalation_level", 0)

        self._healing_in_progress = False
        self._healing_started_time = None  # Session 9: Clear healing timeout
        self._consecutive_healing_failures += 1  # Session 9: Track consecutive failures

        # January 4, 2026: Track healing attempts during quorum loss
        if self._quorum_lost:
            self._quorum_lost_healing_attempts += 1

        logger.warning(
            f"[P2PRecovery] Partition healing failed: {reason} "
            f"(escalation_level={escalation_level}, "
            f"consecutive_failures={self._consecutive_healing_failures}, "
            f"quorum_lost={self._quorum_lost}, "
            f"quorum_lost_healing_attempts={self._quorum_lost_healing_attempts})"
        )

        # January 4, 2026: Early escalation when quorum is lost
        # After 2 healing attempts with quorum lost, restart immediately (skip normal escalation)
        if self._quorum_lost and self._quorum_lost_healing_attempts >= 2:
            logger.error(
                f"[P2PRecovery] QUORUM LOST + {self._quorum_lost_healing_attempts} healing "
                f"attempts failed - triggering IMMEDIATE P2P restart (early escalation)"
            )
            await self._restart_p2p(force=True)
            return

        # Session 9: If 3+ consecutive healing failures (normal case), trigger P2P restart
        if self._consecutive_healing_failures >= 3:
            logger.error(
                f"[P2PRecovery] {self._consecutive_healing_failures} consecutive healing "
                f"failures - triggering P2P restart"
            )
            await self._restart_p2p(force=True)

    async def _on_daemon_failure_classified(self, event: Any) -> None:
        """Handle DAEMON_FAILURE_CLASSIFIED - unified failure response coordination.

        Jan 5, 2026 (Sprint 17.9): Subscribes to DaemonHealthAnalyzer's classification
        events to coordinate recovery actions based on failure severity.

        Actions taken based on failure category:
        - TRANSIENT: Log debug, no action (expected to self-recover)
        - DEGRADED: Log warning, track for monitoring
        - PERSISTENT: Log error, emit recovery request if needed
        - CRITICAL: Log critical, trigger immediate recovery actions

        For P2P-related daemons (P2P_*, AUTO_SYNC, etc.), critical failures
        may trigger P2P restart to restore cluster connectivity.
        """
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)
        daemon_name = payload.get("daemon_name", "unknown")
        category = payload.get("category", "transient")
        recommended_action = payload.get("recommended_action", "monitor")
        consecutive_failures = payload.get("consecutive_failures", 0)
        needs_intervention = payload.get("needs_intervention", False)
        hostname = payload.get("hostname", "unknown")

        # Categorize response based on failure severity
        if category == "critical":
            logger.critical(
                f"[P2PRecovery] CRITICAL daemon failure: {daemon_name} "
                f"(failures={consecutive_failures}, host={hostname}, "
                f"action={recommended_action})"
            )

            # Check if this is a P2P-related daemon that warrants cluster recovery
            p2p_related_daemons = {
                "P2P_ORCHESTRATOR",
                "P2P_RECOVERY",
                "AUTO_SYNC",
                "GOSSIP_SYNC",
                "ELO_SYNC",
                "SYNC_ROUTER",
            }
            if daemon_name.upper() in p2p_related_daemons:
                logger.error(
                    f"[P2PRecovery] Critical failure in P2P-related daemon {daemon_name} "
                    f"- considering P2P restart"
                )
                # Only restart if not already healing
                if not self._is_healing_in_progress():
                    await self._trigger_p2p_restart(
                        reason=f"critical_daemon_failure:{daemon_name}",
                        force=False,  # Don't force, allow cooldown
                    )

            # Emit recovery event for monitoring
            self._try_emit_daemon_recovery_event(daemon_name, category, recommended_action)

        elif category == "persistent":
            logger.error(
                f"[P2PRecovery] Persistent daemon failure: {daemon_name} "
                f"(failures={consecutive_failures}, action={recommended_action})"
            )
            # Track persistent failures for trending
            self._persistent_failure_count = getattr(self, "_persistent_failure_count", 0) + 1

        elif category == "degraded":
            logger.warning(
                f"[P2PRecovery] Degraded daemon: {daemon_name} "
                f"(action={recommended_action})"
            )

        else:  # transient
            logger.debug(
                f"[P2PRecovery] Transient daemon failure: {daemon_name} "
                f"(expected to self-recover)"
            )

    async def _on_event_loop_blocked(self, event: Any) -> None:
        """Handle EVENT_LOOP_BLOCKED - event loop health recovery.

        Jan 24, 2026: Added to recover from blocked event loops that cause
        P2P unresponsiveness. When the event loop is blocked by synchronous
        operations (SQLite, subprocess.run, etc.), the node becomes unresponsive
        to HTTP requests and leader probes.

        Actions:
        - Track consecutive blocking events
        - After 5+ consecutive blocks, consider P2P restart to recover
        - Emit monitoring event for alerting
        """
        payload = get_event_payload(event)
        node_id = payload.get("node_id", "unknown")
        latency_seconds = payload.get("latency_seconds", 0)
        consecutive_blocks = payload.get("consecutive_blocks", 0)
        severity = payload.get("severity", "warning")

        logger.warning(
            f"[P2PRecovery] Event loop blocked on {node_id}: "
            f"latency={latency_seconds:.2f}s, consecutive={consecutive_blocks}, "
            f"severity={severity}"
        )

        # Track consecutive blocks for this daemon
        self._event_loop_blocks = getattr(self, "_event_loop_blocks", 0) + 1

        # After 5+ consecutive critical blocks, consider P2P restart
        # This is a fallback - the blocking operations should be fixed first
        if severity == "critical" and consecutive_blocks >= 5:
            logger.critical(
                f"[P2PRecovery] Critical event loop blocking detected - "
                f"{consecutive_blocks} consecutive blocks. P2P may be unresponsive."
            )

            # Emit alert event for monitoring
            safe_emit_event(
                "p2p_event_loop_critical",
                {
                    "node_id": node_id,
                    "consecutive_blocks": consecutive_blocks,
                    "latency_seconds": latency_seconds,
                    "action": "alert",
                },
                context="P2PRecovery",
            )

            # Only trigger restart if we've seen sustained blocking
            # and not already healing
            if consecutive_blocks >= 10 and not self._is_healing_in_progress():
                logger.critical(
                    f"[P2PRecovery] Sustained event loop blocking ({consecutive_blocks} blocks) "
                    f"- triggering P2P restart for recovery"
                )
                await self._trigger_p2p_restart(
                    reason=f"event_loop_blocked:consecutive={consecutive_blocks}",
                    force=False,  # Allow cooldown
                )

    def _try_emit_daemon_recovery_event(
        self, daemon_name: str, category: str, recommended_action: str
    ) -> None:
        """Emit event for daemon recovery action - best effort."""
        safe_emit_event(
            "p2p_recovery_started",
            {
                "trigger": "daemon_failure_classified",
                "daemon_name": daemon_name,
                "failure_category": category,
                "recommended_action": recommended_action,
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    def _is_healing_in_progress(self) -> bool:
        """Check if partition healing is in progress (with timeout protection).

        Jan 3, 2026 Session 9: Returns True only if healing is active AND
        hasn't exceeded the timeout. If healing started but exceeded the
        timeout, clear the flag and return False to allow recovery operations.
        """
        if not self._healing_in_progress:
            return False

        # Check if healing has exceeded timeout
        if self._healing_started_time is not None:
            elapsed = time.time() - self._healing_started_time
            if elapsed > self._healing_timeout_seconds:
                logger.warning(
                    f"[P2PRecovery] Healing timeout exceeded ({elapsed:.1f}s > "
                    f"{self._healing_timeout_seconds}s) - clearing stuck flag"
                )
                self._healing_in_progress = False
                self._healing_started_time = None
                self._consecutive_healing_failures += 1  # Count as a failure
                return False

        return True

    # =========================================================================
    # Healing Convergence Validation (January 4, 2026)
    # =========================================================================

    async def _validate_healing_convergence(
        self,
        wait_seconds: float = 5.0,
        min_healthy_ratio: float = 0.95,
    ) -> bool:
        """Validate that partition healing actually converged across the cluster.

        January 4, 2026: After partition_healer reports success, wait for gossip
        to propagate and verify that the cluster has actually converged to a
        healthy state. This prevents premature success reports.

        Args:
            wait_seconds: Time to wait for gossip propagation (default: 5s)
            min_healthy_ratio: Minimum healthy ratio to consider converged (0.95)

        Returns:
            True if gossip healthy ratio >= min_healthy_ratio, False otherwise
        """
        import asyncio

        # Wait for gossip to propagate
        logger.debug(
            f"[ConvergenceValidation] Waiting {wait_seconds}s for gossip propagation"
        )
        await asyncio.sleep(wait_seconds)

        # Check gossip health
        try:
            healthy_ratio = self._get_gossip_healthy_ratio()
            logger.info(
                f"[ConvergenceValidation] Gossip healthy ratio: {healthy_ratio:.2%} "
                f"(threshold: {min_healthy_ratio:.0%})"
            )
            return healthy_ratio >= min_healthy_ratio
        except Exception as e:
            logger.warning(
                f"[ConvergenceValidation] Failed to get gossip health: {e}"
            )
            # If we can't check, assume not converged
            return False

    def _get_gossip_healthy_ratio(self) -> float:
        """Get the current gossip healthy ratio from P2P status.

        Returns:
            Ratio of healthy peers (0.0-1.0), or 0.0 if unavailable.
        """
        try:
            import requests

            # Query local P2P orchestrator status
            response = requests.get(
                f"http://localhost:{self.config.p2p_port}/status",
                timeout=5.0,
            )
            if response.status_code == 200:
                status = response.json()
                alive_peers = status.get("alive_peers", 0)
                total_peers = status.get("total_peers", 0)
                if total_peers > 0:
                    return alive_peers / total_peers
            return 0.0
        except Exception:
            return 0.0

    def _try_emit_convergence_failure(self, nodes_healed: int) -> None:
        """Emit HEALING_CONVERGENCE_FAILED event for monitoring.

        January 4, 2026: Emitted when partition healing reported success but
        gossip convergence validation failed.
        """
        safe_emit_event(
            "HEALING_CONVERGENCE_FAILED",
            {
                "nodes_healed": nodes_healed,
                "reason": "gossip_healthy_ratio_below_threshold",
                "threshold": 0.95,
                "consecutive_failures": self._consecutive_healing_failures,
                "timestamp": time.time(),
            },
            source="p2p_recovery_daemon",
            context="convergence_failure",
        )

    # =========================================================================
    # HealthCoordinator Integration (Jan 3, 2026 Sprint 12 Session 10)
    # =========================================================================

    def _get_health_coordinator_recommendation(self) -> tuple[str | None, dict | None]:
        """Get recovery recommendation from HealthCoordinator if available.

        Jan 3, 2026: Integrates with the centralized HealthCoordinator for
        unified recovery decisions. This provides a secondary signal that
        aggregates all health sources (gossip, circuit breakers, quorum, daemons).

        Returns:
            Tuple of (action_name, health_state) or (None, None) if unavailable
        """
        try:
            from scripts.p2p.health_coordinator import (
                get_health_coordinator,
                RecoveryAction,
            )

            coordinator = get_health_coordinator()
            action = coordinator.get_recovery_action()
            health = coordinator.get_cluster_health()

            return (
                action.value if action else None,
                {
                    "overall_health": health.overall_health.value if health.overall_health else None,
                    "quorum_health": health.quorum_health.value if health.quorum_health else None,
                    "open_circuits": len(health.open_circuits) if health.open_circuits else 0,
                    "unhealthy_peers": len(health.unhealthy_peers) if health.unhealthy_peers else 0,
                    "health_score": health.health_score if hasattr(health, "health_score") else None,
                }
            )
        except ImportError:
            # HealthCoordinator not available (standalone mode or P2P not running)
            logger.debug("HealthCoordinator not available for recovery decisions")
            return None, None
        except Exception as e:
            # Don't let HealthCoordinator errors affect P2P recovery
            logger.warning(f"Error getting HealthCoordinator recommendation: {e}")
            return None, None

    def _should_escalate_with_health_coordinator(
        self,
        local_decision: str,
        status: dict,
    ) -> bool:
        """Check if HealthCoordinator recommends escalation beyond local decision.

        Jan 3, 2026: Uses HealthCoordinator's unified view to detect cases where:
        - Local checks pass but aggregated health is critical
        - Circuit breakers are open that local checks don't see
        - Quorum issues detected by HealthCoordinator but not local checks

        Args:
            local_decision: Current recovery decision ("healthy", "restart", "election", etc.)
            status: Current P2P status dict

        Returns:
            True if HealthCoordinator recommends more aggressive recovery
        """
        action, health_state = self._get_health_coordinator_recommendation()

        if action is None or health_state is None:
            return False

        # Log the HealthCoordinator's view for observability
        logger.debug(
            f"HealthCoordinator: action={action}, health={health_state.get('overall_health')}, "
            f"score={health_state.get('health_score')}"
        )

        # Add to status for health reporting
        status["health_coordinator"] = {
            "action": action,
            "health_state": health_state,
        }

        # Check for escalation scenarios
        if local_decision == "healthy":
            # Local says healthy, but HealthCoordinator recommends action
            if action in ("restart_p2p", "trigger_election", "heal_partitions"):
                overall = health_state.get("overall_health")
                if overall in ("critical", "degraded"):
                    logger.warning(
                        f"HealthCoordinator disagrees with local healthy check: "
                        f"action={action}, overall_health={overall}, "
                        f"open_circuits={health_state.get('open_circuits')}"
                    )
                    return True

        return False

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check P2P health and recover if needed."""
        # Skip during startup grace period
        if time.time() - self._startup_time < self.config.startup_grace_seconds:
            logger.debug("P2P Recovery: In startup grace period, skipping check")
            return

        # Check P2P health
        is_healthy, status = await self._check_p2p_health()
        self._last_status = status

        # Also check for network isolation (December 2025)
        alive_peers = status.get("alive_peers", 0)
        is_isolated, isolation_details = await self._check_network_isolation(alive_peers)
        status["isolation"] = isolation_details

        # Jan 2026: Proactive voter quorum monitoring
        quorum_healthy, quorum_details = await self._check_voter_quorum(status)
        status["voter_quorum"] = quorum_details

        # Dec 29, 2025: Track leader presence for leader gap detection
        leader_id = status.get("leader_id")
        if leader_id:
            self._last_leader_seen_time = time.time()

        # Jan 3, 2026: Consult HealthCoordinator for unified recovery decision
        # This aggregates gossip, circuit breaker, quorum, and daemon health signals
        local_decision = "healthy" if (is_healthy and not is_isolated) else "unhealthy"
        should_escalate = self._should_escalate_with_health_coordinator(local_decision, status)

        if should_escalate and is_healthy and not is_isolated:
            # HealthCoordinator detected issues that local checks missed
            logger.warning(
                "HealthCoordinator recommends recovery despite local healthy check, "
                "triggering partition healing"
            )
            healing_result = await self._trigger_partition_healing()
            if not healing_result or healing_result.partitions_healed == 0:
                # Healing didn't help, consider restart on next cycle
                self._consecutive_failures += 1
                return

        if is_healthy and not is_isolated:
            # Reset failure counter
            if self._consecutive_failures > 0:
                logger.info(
                    f"P2P healthy after {self._consecutive_failures} failures"
                )
            self._consecutive_failures = 0
            self._last_healthy_time = time.time()

            # Emit recovery event if was previously unhealthy
            if self._was_unhealthy:
                await self._emit_recovery_event(status)
                self._was_unhealthy = False
                # Dec 30, 2025: Reset exponential backoff on recovery
                if self._restart_attempt_count > 0:
                    logger.info(
                        f"P2P recovered after {self._restart_attempt_count} restart attempts, "
                        "resetting backoff"
                    )
                    self._restart_attempt_count = 0

        # Dec 29, 2025: Check for leader gap and trigger election if needed
        leader_gap_seconds = time.time() - self._last_leader_seen_time
        if not leader_id and leader_gap_seconds > self.config.max_leader_gap_seconds:
            logger.warning(
                f"Leader gap detected: no leader for {leader_gap_seconds:.0f}s "
                f"(threshold: {self.config.max_leader_gap_seconds}s), triggering election"
            )
            await self._trigger_leader_election()
            return  # Skip other recovery actions this cycle

        elif is_isolated and self._consecutive_isolation_checks >= self.config.isolation_consecutive_checks:
            # Network isolation confirmed - try partition healing first, then restart
            logger.warning(
                f"Network isolation confirmed after {self._consecutive_isolation_checks} checks, "
                f"attempting partition healing before restart"
            )
            await self._emit_isolation_event(isolation_details)

            # January 2026: Try partition healing first (less disruptive than restart)
            healing_result = await self._trigger_partition_healing()
            if healing_result and healing_result.partitions_healed > 0:
                logger.info(
                    f"Partition healing succeeded: {healing_result.partitions_healed} partitions healed, "
                    f"skipping P2P restart"
                )
                self._consecutive_isolation_checks = 0
                return

            # Fall back to restart if healing didn't fix the problem
            if self._can_restart():
                await self._restart_p2p()
                self._consecutive_isolation_checks = 0
                self._isolation_triggered_restarts += 1
                self._was_unhealthy = True
            else:
                cooldown_remaining = self._get_cooldown_remaining()
                logger.warning(
                    f"P2P restart needed for isolation but in cooldown ({cooldown_remaining:.0f}s remaining)"
                )

        elif not is_healthy:
            # Standard health check failure
            self._consecutive_failures += 1
            effective_threshold = self._get_effective_restart_threshold()
            logger.warning(
                f"P2P health check failed ({self._consecutive_failures}/{effective_threshold})"
            )

            # Check if we should restart (Jan 2026: use NAT-aware threshold)
            if self._consecutive_failures >= effective_threshold:
                if self._can_restart():
                    await self._restart_p2p()
                    self._consecutive_failures = 0
                    self._was_unhealthy = True
                else:
                    cooldown_remaining = self._get_cooldown_remaining()
                    logger.warning(
                        f"P2P restart needed but in cooldown ({cooldown_remaining:.0f}s remaining)"
                    )

        # Jan 7, 2026: Update health check interval based on current cluster state
        # Faster checks during degradation, normal interval when healthy
        self._update_health_check_interval()

    async def _check_p2p_health(self) -> tuple[bool, dict[str, Any]]:
        """Check P2P orchestrator health via /status endpoint.

        Returns:
            Tuple of (is_healthy, status_dict)
        """
        try:
            import aiohttp

            # Jan 3, 2026 (Sprint 15.1): Use adaptive timeout based on system load
            try:
                from app.config.coordination_defaults import get_adaptive_health_timeout
                timeout_seconds = max(
                    get_adaptive_health_timeout(),
                    self.config.health_timeout_seconds,
                )
            except ImportError:
                timeout_seconds = self.config.health_timeout_seconds

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.health_endpoint,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as resp:
                    if resp.status != 200:
                        return False, {"error": f"HTTP {resp.status}"}

                    data = await resp.json()

                    # Check key health indicators
                    alive_peers = data.get("alive_peers", 0)
                    leader_id = data.get("leader_id")
                    role = data.get("role", "unknown")

                    # Consider healthy if:
                    # 1. Has minimum peers
                    # 2. Has a leader (or is leader)
                    is_healthy = (
                        alive_peers >= self.config.min_alive_peers
                        and leader_id is not None
                    )

                    return is_healthy, {
                        "alive_peers": alive_peers,
                        "leader_id": leader_id,
                        "role": role,
                        "is_healthy": is_healthy,
                    }
        except asyncio.TimeoutError:
            return False, {"error": "timeout"}
        except Exception as e:
            return False, {"error": str(e)}

    async def _get_tailscale_online_count(self) -> int:
        """Get count of online Tailscale peers from local Tailscale status.

        Uses TailscaleChecker to query the mesh network and count peers
        that are marked as online.

        Returns:
            Number of online Tailscale peers, or 0 if query fails.
        """
        try:
            from app.coordination.node_availability.providers.tailscale_checker import (
                TailscaleChecker,
            )
            from app.config.cluster_config import load_cluster_config

            checker = TailscaleChecker()
            if not await checker.check_api_availability():
                logger.debug("Tailscale CLI not available for isolation check")
                return 0

            config = load_cluster_config()
            online_nodes = await checker.get_online_nodes(config.hosts_raw)
            count = len(online_nodes)
            self._last_tailscale_count = count
            return count

        except Exception as e:
            logger.debug(f"Failed to get Tailscale peer count: {e}")
            return 0  # Fail open - don't trigger isolation on error

    async def _check_network_isolation(
        self, alive_peers: int
    ) -> tuple[bool, dict[str, Any]]:
        """Check if we're experiencing network isolation.

        Compares P2P visible peers to Tailscale online peers.
        If P2P sees significantly fewer than Tailscale, we may be isolated.

        Args:
            alive_peers: Number of peers visible in P2P cluster

        Returns:
            Tuple of (is_isolated, details_dict)
        """
        if not self.config.isolation_check_enabled:
            return False, {"isolation_check": "disabled"}

        tailscale_count = await self._get_tailscale_online_count()

        if tailscale_count == 0:
            # Can't determine isolation without Tailscale data
            return False, {"isolation_check": "no_tailscale_data"}

        peer_ratio = alive_peers / tailscale_count
        is_isolated = peer_ratio < self.config.min_peer_ratio

        details = {
            "p2p_peers": alive_peers,
            "tailscale_peers": tailscale_count,
            "peer_ratio": round(peer_ratio, 3),
            "min_ratio": self.config.min_peer_ratio,
            "is_isolated": is_isolated,
        }

        if is_isolated:
            self._consecutive_isolation_checks += 1
            details["consecutive_isolation_checks"] = self._consecutive_isolation_checks
            logger.warning(
                f"Network isolation detected: P2P sees {alive_peers} peers "
                f"but Tailscale shows {tailscale_count} online "
                f"(ratio={peer_ratio:.2f}, check {self._consecutive_isolation_checks}/"
                f"{self.config.isolation_consecutive_checks})"
            )
        else:
            if self._consecutive_isolation_checks > 0:
                logger.info(
                    f"Network isolation cleared after {self._consecutive_isolation_checks} checks"
                )
            self._consecutive_isolation_checks = 0

        # Sprint 16.1 (Jan 3, 2026): TCP validation to avoid false-positive on Tailscale outage
        if is_isolated and self.config.tcp_validation_enabled:
            tcp_reachable = await self._tcp_ping_voters()
            details["tcp_reachable_voters"] = tcp_reachable
            details["tcp_min_required"] = self.config.tcp_min_reachable

            if tcp_reachable >= self.config.tcp_min_reachable:
                # TCP can reach voters - this is likely a Tailscale outage, not isolation
                logger.info(
                    f"TCP validation: {tcp_reachable} voters reachable via TCP "
                    f"(>= {self.config.tcp_min_reachable}), clearing isolation flag"
                )
                is_isolated = False
                details["is_isolated"] = False
                details["isolation_cleared_by_tcp"] = True
                self._consecutive_isolation_checks = 0
            else:
                logger.warning(
                    f"TCP validation confirms isolation: only {tcp_reachable} voters "
                    f"reachable via TCP (< {self.config.tcp_min_reachable} required)"
                )
                details["tcp_confirms_isolation"] = True

        return is_isolated, details

    async def _tcp_ping_voters(self) -> int:
        """Check TCP connectivity to voter nodes.

        Sprint 16.1 (Jan 3, 2026): Secondary signal for network isolation detection.
        Attempts TCP connection to P2P port on each voter node to determine
        if nodes are truly unreachable or if it's a Tailscale-specific outage.

        Returns:
            Number of voters reachable via TCP connection.
        """
        try:
            from app.config.cluster_config import get_p2p_voters, ClusterConfigCache
            import socket

            voters = get_p2p_voters()
            if not voters:
                logger.debug("No voters configured for TCP ping")
                return 0

            cache = ClusterConfigCache.get_config_cache()
            config = cache.get_config()

            reachable_count = 0
            p2p_port = 8770  # Default P2P port

            async def try_tcp_connect(voter_id: str) -> bool:
                """Attempt TCP connection to voter."""
                try:
                    # Get voter's IP from config
                    voter_config = config.hosts.get(voter_id)
                    if not voter_config:
                        return False

                    # Try Tailscale IP first, then SSH host
                    ip = voter_config.get("tailscale_ip") or voter_config.get("ssh_host")
                    if not ip:
                        return False

                    # Try TCP connection with timeout
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.config.tcp_ping_timeout)
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: sock.connect_ex((ip, p2p_port))
                        )
                        return result == 0
                    finally:
                        sock.close()
                except Exception as e:
                    logger.debug(f"TCP ping to {voter_id} failed: {e}")
                    return False

            # Run TCP pings in parallel with overall timeout
            tasks = [try_tcp_connect(voter) for voter in voters]
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.tcp_ping_timeout * 2,  # 2x single timeout for all
                )
                for i, result in enumerate(results):
                    if result is True:
                        reachable_count += 1
                        logger.debug(f"TCP ping: {voters[i]} reachable")
                    elif isinstance(result, Exception):
                        logger.debug(f"TCP ping: {voters[i]} error: {result}")
            except asyncio.TimeoutError:
                logger.warning("TCP ping overall timeout, returning partial count")

            logger.info(f"TCP ping: {reachable_count}/{len(voters)} voters reachable")
            return reachable_count

        except ImportError as e:
            logger.debug(f"TCP ping unavailable (import error): {e}")
            return 0
        except Exception as e:
            logger.debug(f"TCP ping failed: {e}")
            return 0

    async def _check_voter_quorum(self, status: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Check voter quorum health proactively.

        Jan 2026: Added for proactive voter quorum monitoring during regular
        health check cycles. Triggers recovery actions when voter count drops
        below min_voters_for_healthy_quorum (default: 5 of 7 voters).

        Args:
            status: P2P status dict from health check

        Returns:
            Tuple of (quorum_healthy, details_dict)
        """
        if not self._daemon_config.voter_quorum_monitoring_enabled:
            return True, {"voter_quorum_check": "disabled"}

        try:
            from app.config.cluster_config import get_p2p_voters

            # Get all configured voters
            all_voters = set(get_p2p_voters())
            total_voters = len(all_voters)

            if total_voters == 0:
                return True, {"voter_quorum_check": "no_voters_configured"}

            # Get list of alive peers from P2P status
            alive_peers_list = status.get("alive_peers_list", [])

            # If we only have a count (not a list), try to get from P2P
            if isinstance(alive_peers_list, int) or not alive_peers_list:
                alive_peers_list = await self._get_alive_peers_list()

            # Count online voters
            alive_peers_set = set(alive_peers_list) if alive_peers_list else set()
            online_voters = all_voters & alive_peers_set
            offline_voters = all_voters - online_voters
            online_count = len(online_voters)

            details = {
                "total_voters": total_voters,
                "online_voters": online_count,
                "offline_voters": list(offline_voters),
                "min_for_healthy": self._daemon_config.min_voters_for_healthy_quorum,
                "quorum_size": self._daemon_config.voter_quorum_size,
            }

            self._last_voter_quorum_check = details

            # Jan 2026 Session 8: Emit fine-grained voter events for state changes
            await self._emit_voter_state_changes(online_voters)

            # Check if quorum is at risk (< min_voters_for_healthy_quorum)
            quorum_healthy = online_count >= self._daemon_config.min_voters_for_healthy_quorum

            if not quorum_healthy:
                self._quorum_at_risk_consecutive += 1
                details["consecutive_at_risk_checks"] = self._quorum_at_risk_consecutive

                # Determine severity
                if online_count < self._daemon_config.voter_quorum_size:
                    severity = "QUORUM_LOST"
                    logger.critical(
                        f"VOTER QUORUM LOST: {online_count}/{total_voters} voters online "
                        f"(need {self._daemon_config.voter_quorum_size} for quorum)"
                    )
                else:
                    severity = "QUORUM_AT_RISK"
                    logger.warning(
                        f"Voter quorum at risk: {online_count}/{total_voters} voters online "
                        f"(healthy threshold: {self._daemon_config.min_voters_for_healthy_quorum})"
                    )

                details["severity"] = severity

                # Trigger recovery after 2 consecutive checks to avoid spurious triggers
                if self._quorum_at_risk_consecutive >= 2:
                    logger.info(
                        f"Triggering proactive quorum recovery "
                        f"(consecutive at-risk checks: {self._quorum_at_risk_consecutive})"
                    )
                    await self._trigger_proactive_quorum_recovery(offline_voters, severity)
            else:
                if self._quorum_at_risk_consecutive > 0 or self._quorum_lost:
                    logger.info(
                        f"Voter quorum restored: {online_count}/{total_voters} voters online "
                        f"(was at risk for {self._quorum_at_risk_consecutive} checks)"
                    )
                    # January 4, 2026: Clear quorum loss state to restore normal behavior
                    if self._quorum_lost:
                        logger.info("Quorum restored - reverting to normal recovery cooldown")
                        self._quorum_lost = False
                        self._quorum_lost_healing_attempts = 0  # Reset early escalation counter
                self._quorum_at_risk_consecutive = 0

            return quorum_healthy, details

        except ImportError:
            return True, {"voter_quorum_check": "cluster_config_unavailable"}
        except Exception as e:
            logger.debug(f"Failed to check voter quorum: {e}")
            return True, {"voter_quorum_check": "error", "error": str(e)}

    async def _get_alive_peers_list(self) -> list[str]:
        """Get list of alive peer IDs from P2P status.

        Jan 2026: Helper to get peer list when only count is available in status.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Try extended status endpoint that includes peer list
                async with session.get(
                    f"{self._daemon_config.health_endpoint}/peers",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("alive_peers", [])
        except Exception:
            pass
        return []

    async def _trigger_proactive_quorum_recovery(
        self, offline_voters: set[str], severity: str
    ) -> None:
        """Trigger proactive recovery when voter quorum is at risk.

        Jan 2026: Initiates recovery actions when voters go offline.
        Actions depend on severity:
        - QUORUM_AT_RISK: Boost reconnection priority for offline voters
        - QUORUM_LOST: Aggressive recovery + potential P2P restart

        Args:
            offline_voters: Set of offline voter node IDs
            severity: Either "QUORUM_AT_RISK" or "QUORUM_LOST"
        """
        self._quorum_recovery_triggered += 1

        # Emit event for monitoring
        event_type = "quorum_recovery_started" if severity == "QUORUM_LOST" else "quorum_at_risk"
        safe_emit_event(
            event_type,
            {
                "trigger": "proactive_monitoring",
                "offline_voters": list(offline_voters),
                "severity": severity,
                "consecutive_checks": self._quorum_at_risk_consecutive,
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

        # Always prioritize voter reconnections
        await self._prioritize_quorum_reconnections()

        # For QUORUM_LOST, consider additional recovery actions
        if severity == "QUORUM_LOST":
            logger.warning(
                f"Quorum lost with {len(offline_voters)} voters offline. "
                "Triggering cluster healing for voter nodes."
            )

            # Try to trigger cluster healing specifically for voter nodes
            try:
                await self._trigger_voter_healing(offline_voters)
            except Exception as e:
                logger.warning(f"Voter healing trigger failed: {e}")

    async def _trigger_voter_healing(self, offline_voters: set[str]) -> None:
        """Trigger cluster healing specifically for offline voters.

        Jan 2026: Coordinates with ClusterHealingLoop to prioritize voter recovery.
        """
        from app.distributed.data_events import DataEventType

        # Emit event that ClusterHealingLoop can subscribe to
        safe_emit_event(
            DataEventType.VOTER_HEALING_REQUESTED.value,
            {
                "offline_voters": list(offline_voters),
                "priority": "urgent",
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )
        logger.info(f"Requested urgent healing for {len(offline_voters)} offline voters")

    def _get_current_cooldown(self) -> float:
        """Calculate current cooldown with exponential backoff.

        Dec 30, 2025: Implements exponential backoff for consecutive restart
        attempts. Cooldown doubles after each failed restart until max is reached.

        Formula: base_cooldown * (multiplier ** attempt_count)
        Capped at max_cooldown_seconds.

        Jan 2026: Uses NAT-aware base cooldown (60s for NAT-blocked, 300s normal).
        """
        base_cooldown = float(self._get_effective_cooldown())
        if self._restart_attempt_count == 0:
            return base_cooldown

        backoff_cooldown = (
            base_cooldown
            * (self.config.restart_backoff_multiplier ** self._restart_attempt_count)
        )
        return min(backoff_cooldown, float(self.config.max_cooldown_seconds))

    def _can_restart(self) -> bool:
        """Check if we can restart (cooldown has passed)."""
        if self._last_restart_time == 0:
            return True
        elapsed = time.time() - self._last_restart_time
        current_cooldown = self._get_current_cooldown()
        return elapsed >= current_cooldown

    def _get_cooldown_remaining(self) -> float:
        """Get seconds remaining in cooldown period (with exponential backoff)."""
        if self._last_restart_time == 0:
            return 0
        elapsed = time.time() - self._last_restart_time
        current_cooldown = self._get_current_cooldown()
        return max(0, current_cooldown - elapsed)

    async def _restart_p2p(self, force: bool = False) -> None:
        """Restart the P2P orchestrator process.

        December 2025: Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
        Dec 30, 2025: Tracks restart attempts for exponential backoff.
        Jan 3, 2026 Session 7: Respects partition healing state unless force=True.

        Args:
            force: If True, restart even during partition healing (for P2P_RECOVERY_NEEDED).
        """
        # Jan 3, 2026 Session 7: Don't restart during partition healing unless forced
        # Jan 3, 2026 Session 9: Use helper that includes timeout protection
        if self._is_healing_in_progress() and not force:
            logger.info(
                "[P2PRecovery] Partition healing in progress, deferring P2P restart"
            )
            return

        # Increment restart attempt counter for exponential backoff
        self._restart_attempt_count += 1
        current_cooldown = self._get_current_cooldown()
        logger.warning(
            f"Initiating P2P orchestrator restart (attempt {self._restart_attempt_count}, "
            f"next cooldown: {current_cooldown:.0f}s)"
        )

        # Update tracking
        self._last_restart_time = time.time()
        self._total_restarts += 1
        self._startup_time = time.time()  # Reset for grace period

        # Emit restart event
        await self._emit_restart_event()

        try:
            # Kill existing P2P process (non-blocking)
            proc = await asyncio.create_subprocess_exec(
                "pkill", "-f", "p2p_orchestrator.py",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
                if proc.returncode == 0:
                    logger.info("Killed existing P2P process")
                else:
                    logger.debug("No existing P2P process to kill")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("Timeout killing P2P process")

            # Wait for process to fully terminate
            await asyncio.sleep(5)

            # Start new P2P process
            # The P2P orchestrator should be started by the master loop,
            # but we can also start it directly if needed
            p2p_script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "scripts",
                "p2p_orchestrator.py",
            )

            if os.path.exists(p2p_script):
                # December 30, 2025: Wrap Popen in asyncio.to_thread for consistency
                # Popen is mostly non-blocking, but fork can briefly block
                def _start_p2p() -> subprocess.Popen[bytes]:
                    return subprocess.Popen(
                        [sys.executable, p2p_script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )

                process = await asyncio.to_thread(_start_p2p)
                logger.info(f"Started new P2P process (PID {process.pid})")

                # Dec 30, 2025: Emit P2P_RESTARTED event for subscription resilience
                await self._emit_p2p_restarted_event()

                # Dec 30, 2025: Trigger quorum-aware reconnection prioritization
                await self._prioritize_quorum_reconnections()
            else:
                logger.warning(
                    "P2P script not found, relying on master_loop to restart"
                )

        except Exception as e:
            logger.error(f"Error restarting P2P: {e}")
            self._stats.errors_count += 1
            self._last_error = str(e)

    async def _prioritize_quorum_reconnections(self) -> None:
        """Prioritize reconnection to voter nodes after P2P restart.

        Dec 30, 2025: Added for faster quorum restoration during 48h autonomous ops.
        Uses QuorumRecoveryManager to prioritize voter nodes when quorum is at risk.
        """
        try:
            from app.coordination.quorum_recovery import get_quorum_manager

            manager = get_quorum_manager()

            # Wait for P2P to start accepting connections
            await asyncio.sleep(5)

            # Get current P2P status to find offline voters
            is_healthy, status = await self._check_p2p_health()
            if not is_healthy:
                logger.debug("P2P not yet healthy, skipping quorum prioritization")
                return

            # Get list of known voters from config
            from app.config.cluster_config import get_p2p_voters

            all_voters = set(get_p2p_voters())
            alive_peers = status.get("alive_peers_list", [])

            if isinstance(alive_peers, int):
                # Only count, not list - can't determine offline voters
                logger.debug("P2P status doesn't include peer list, skipping prioritization")
                return

            online_voters = all_voters & set(alive_peers)
            offline_voters = all_voters - online_voters

            manager.update_online_voters(online_voters)

            if manager.needs_more_voters() and offline_voters:
                logger.info(
                    f"QuorumRecovery: Need {manager.voters_needed_for_quorum()} more voters, "
                    f"{len(offline_voters)} voters offline"
                )

                # Log prioritization (actual reconnection handled by gossip protocol)
                prioritized = manager.get_prioritized_reconnection_order(list(offline_voters))
                if prioritized:
                    logger.info(
                        f"QuorumRecovery: Prioritized voter reconnection order: {prioritized}"
                    )
                    self._quorum_recovery_attempts += 1

                    # Emit event to notify P2P of prioritized reconnection order
                    await self._emit_quorum_priority_event(prioritized)
            else:
                logger.debug(
                    f"QuorumRecovery: Quorum met ({len(online_voters)}/{manager._config.quorum_size})"
                )

        except ImportError:
            logger.debug("QuorumRecoveryManager not available, skipping prioritization")
        except Exception as e:
            logger.warning(f"Error in quorum prioritization: {e}")

    async def _emit_quorum_priority_event(self, prioritized_nodes: list[str]) -> None:
        """Emit event with prioritized reconnection order.

        Dec 30, 2025: Notifies P2P orchestrator of priority reconnection order.
        """
        safe_emit_event(
            "quorum_priority_reconnect",
            {
                "prioritized_nodes": prioritized_nodes,
                "voters_needed": len(prioritized_nodes),
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    async def _trigger_leader_election(self) -> bool:
        """Trigger a leader election via the P2P orchestrator API.

        Dec 29, 2025: Added for self-healing when leader gaps are detected.
        This allows the daemon to proactively trigger elections without
        requiring a full P2P restart.

        Returns:
            True if election was triggered successfully, False otherwise.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.leader_election_endpoint,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status == 200:
                        self._leader_gap_elections_triggered += 1
                        logger.info(
                            f"Leader election triggered successfully "
                            f"(total: {self._leader_gap_elections_triggered})"
                        )
                        await self._emit_leader_gap_event()
                        return True
                    else:
                        logger.warning(
                            f"Leader election trigger failed: HTTP {resp.status}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.warning("Leader election trigger timed out")
            return False
        except Exception as e:
            logger.error(f"Error triggering leader election: {e}")
            return False

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_restart_event(self) -> None:
        """Emit event when P2P restart is triggered."""
        safe_emit_event(
            "p2p_restart_triggered",
            {
                "consecutive_failures": self._consecutive_failures,
                "last_status": self._last_status,
                "total_restarts": self._total_restarts,
                "unhealthy_duration_seconds": time.time() - self._last_healthy_time,
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    async def _emit_recovery_event(self, status: dict[str, Any]) -> None:
        """Emit event when P2P recovers."""
        safe_emit_event(
            "p2p_health_recovered",
            {
                "alive_peers": status.get("alive_peers", 0),
                "leader_id": status.get("leader_id"),
                "recovery_duration_seconds": time.time() - self._last_healthy_time,
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    async def _emit_p2p_restarted_event(self) -> None:
        """Emit event when P2P orchestrator successfully restarts.

        Dec 30, 2025: Added for event subscription resilience. This allows
        subscribers like SelfplayScheduler to verify their subscriptions
        are still active after a P2P mesh recovery.
        """
        try:
            from app.distributed.data_events import emit_p2p_restarted

            await emit_p2p_restarted(
                trigger="recovery_daemon",
                restart_count=self._total_restarts,
                previous_state=self._last_status or "unknown",
                source="P2PRecoveryDaemon",
            )
            logger.info(f"Emitted P2P_RESTARTED event (restart #{self._total_restarts})")
        except Exception as e:
            logger.debug(f"Failed to emit P2P_RESTARTED: {e}")

    async def _emit_isolation_event(self, isolation_details: dict[str, Any]) -> None:
        """Emit event when network isolation is detected.

        December 2025: New event for monitoring network partition scenarios.
        """
        safe_emit_event(
            "network_isolation_detected",
            {
                "p2p_peers": isolation_details.get("p2p_peers", 0),
                "tailscale_peers": isolation_details.get("tailscale_peers", 0),
                "peer_ratio": isolation_details.get("peer_ratio", 0),
                "consecutive_checks": self._consecutive_isolation_checks,
                "isolation_triggered_restarts": self._isolation_triggered_restarts,
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    async def _trigger_partition_healing(self) -> Any:
        """Trigger partition healing as a softer alternative to P2P restart.

        January 2026: Auto-trigger partition healing before resorting to full restart.
        Uses the partition healer's rate limiting to prevent healing loops.

        Returns:
            HealingResult if healing was attempted, None if not available or rate-limited
        """
        try:
            from scripts.p2p.partition_healer import trigger_partition_healing

            logger.info("Triggering partition healing for network isolation recovery")
            result = await trigger_partition_healing(delay=0, force=False)
            if result is None:
                logger.debug("Partition healing was rate-limited or disabled")
            return result
        except ImportError:
            logger.debug("Partition healer not available")
            return None
        except Exception as e:
            logger.warning(f"Partition healing failed: {e}")
            return None

    async def _emit_leader_gap_event(self) -> None:
        """Emit event when leader gap triggers an election.

        Dec 29, 2025: New event for monitoring leader gap scenarios.
        """
        leader_gap_seconds = time.time() - self._last_leader_seen_time
        safe_emit_event(
            "leader_election_triggered",
            {
                "reason": "leader_gap",
                "leader_gap_seconds": leader_gap_seconds,
                "threshold_seconds": self.config.max_leader_gap_seconds,
                "total_leader_gap_elections": self._leader_gap_elections_triggered,
                "source": "P2PRecoveryDaemon",
            },
            context="P2PRecovery",
        )

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="P2PRecovery not running",
                details={},
            )

        # Check if P2P is currently unhealthy (Jan 2026: use NAT-aware threshold)
        effective_threshold = self._get_effective_restart_threshold()
        if self._consecutive_failures >= effective_threshold:
            return HealthCheckResult(
                healthy=True,  # Daemon is healthy, P2P is not
                status=CoordinatorStatus.RUNNING,
                message=f"P2P unhealthy, {self._consecutive_failures} consecutive failures",
                details={
                    "p2p_healthy": False,
                    "consecutive_failures": self._consecutive_failures,
                    "last_status": self._last_status,
                    "total_restarts": self._total_restarts,
                    "isolation_checks": self._consecutive_isolation_checks,
                    "restart_attempt_count": self._restart_attempt_count,
                    "current_cooldown_seconds": self._get_current_cooldown(),
                    "cooldown_remaining_seconds": self._get_cooldown_remaining(),
                },
            )

        # Check if network isolation is being detected
        if self._consecutive_isolation_checks > 0:
            return HealthCheckResult(
                healthy=True,  # Daemon is healthy, but isolation detected
                status=CoordinatorStatus.RUNNING,
                message=f"Network isolation detected ({self._consecutive_isolation_checks} checks)",
                details={
                    "p2p_healthy": True,
                    "network_isolated": True,
                    "isolation_checks": self._consecutive_isolation_checks,
                    "isolation_threshold": self.config.isolation_consecutive_checks,
                    "last_tailscale_count": self._last_tailscale_count,
                    "last_status": self._last_status,
                },
            )

        # Jan 3, 2026: Include HealthCoordinator info if available
        health_coordinator_info = None
        action, health_state = self._get_health_coordinator_recommendation()
        if action is not None:
            health_coordinator_info = {
                "recommended_action": action,
                "overall_health": health_state.get("overall_health") if health_state else None,
                "health_score": health_state.get("health_score") if health_state else None,
            }

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="P2P cluster healthy",
            details={
                "p2p_healthy": True,
                "consecutive_failures": self._consecutive_failures,
                "cycles_completed": self._stats.cycles_completed,
                "total_restarts": self._total_restarts,
                "isolation_triggered_restarts": self._isolation_triggered_restarts,
                "leader_gap_elections_triggered": self._leader_gap_elections_triggered,
                "restart_attempt_count": self._restart_attempt_count,
                "current_cooldown_seconds": self._get_current_cooldown(),
                "cooldown_remaining_seconds": self._get_cooldown_remaining(),
                "last_status": self._last_status,
                "health_coordinator": health_coordinator_info,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Return detailed status."""
        base_status = super().get_status()

        base_status["p2p_status"] = {
            "consecutive_failures": self._consecutive_failures,
            "last_restart_time": self._last_restart_time,
            "total_restarts": self._total_restarts,
            "last_healthy_time": self._last_healthy_time,
            "unhealthy_duration": time.time() - self._last_healthy_time if self._was_unhealthy else 0,
            "cooldown_remaining": self._get_cooldown_remaining(),
            "last_status": self._last_status,
        }

        # Network isolation detection status (December 2025)
        base_status["isolation_status"] = {
            "enabled": self.config.isolation_check_enabled,
            "consecutive_checks": self._consecutive_isolation_checks,
            "threshold": self.config.isolation_consecutive_checks,
            "min_peer_ratio": self.config.min_peer_ratio,
            "last_tailscale_count": self._last_tailscale_count,
            "isolation_triggered_restarts": self._isolation_triggered_restarts,
        }

        # Dec 29, 2025: Leader gap detection status
        leader_gap_seconds = time.time() - self._last_leader_seen_time
        base_status["leader_gap_status"] = {
            "last_leader_seen_time": self._last_leader_seen_time,
            "current_leader_gap_seconds": leader_gap_seconds,
            "max_leader_gap_seconds": self.config.max_leader_gap_seconds,
            "elections_triggered": self._leader_gap_elections_triggered,
            "quorum_recovery_attempts": self._quorum_recovery_attempts,
        }

        # Jan 2026: Voter quorum monitoring status
        base_status["voter_quorum_status"] = {
            "enabled": self.config.voter_quorum_monitoring_enabled,
            "min_voters_for_healthy": self.config.min_voters_for_healthy_quorum,
            "quorum_size": self.config.voter_quorum_size,
            "consecutive_at_risk_checks": self._quorum_at_risk_consecutive,
            "recovery_triggered_count": self._quorum_recovery_triggered,
            "last_check": self._last_voter_quorum_check,
            # January 4, 2026: Dynamic cooldown state
            "quorum_lost": self._quorum_lost,
            "effective_cooldown_seconds": self._get_effective_cooldown(),
            # January 4, 2026: Early escalation counter for quorum loss
            "quorum_lost_healing_attempts": self._quorum_lost_healing_attempts,
        }

        # Jan 3, 2026: HealthCoordinator integration status
        action, health_state = self._get_health_coordinator_recommendation()
        base_status["health_coordinator"] = {
            "available": action is not None,
            "recommended_action": action,
            "overall_health": health_state.get("overall_health") if health_state else None,
            "health_score": health_state.get("health_score") if health_state else None,
            "open_circuits": health_state.get("open_circuits") if health_state else None,
            "unhealthy_peers": health_state.get("unhealthy_peers") if health_state else None,
        }

        return base_status

    def is_p2p_healthy(self) -> bool:
        """Check if P2P is currently healthy based on last check.

        Jan 2026: Uses NAT-aware threshold for healthiness check.
        """
        return self._consecutive_failures < self._get_effective_restart_threshold()

    # =========================================================================
    # Voter State Persistence (Sprint 15.1 - Jan 3, 2026)
    # =========================================================================

    def _get_state_for_persistence(self) -> dict[str, Any]:
        """Return voter health state for persistence.

        Jan 3, 2026 (Sprint 15.1): Required by StatePersistenceMixin.
        Persists voter state for faster quorum recovery after restart.
        """
        return {
            "last_online_voters": list(self._last_online_voters),
            "voter_state_history": {
                voter_id: [(ts, online) for ts, online in history]
                for voter_id, history in self._voter_state_history.items()
            },
            "flapping_voters": list(self._flapping_voters),
            "quorum_recovery_attempts": self._quorum_recovery_attempts,
            "quorum_recovery_triggered": self._quorum_recovery_triggered,
            "last_voter_quorum_check": self._last_voter_quorum_check,
            # January 4, 2026: Persist quorum loss state for faster recovery after restart
            "quorum_lost": self._quorum_lost,
            # January 4, 2026: Early escalation counter
            "quorum_lost_healing_attempts": self._quorum_lost_healing_attempts,
            "persisted_at": time.time(),
        }

    def _restore_state_from_persistence(self, state: dict[str, Any]) -> None:
        """Restore voter health state from persistence.

        Jan 3, 2026 (Sprint 15.1): Required by StatePersistenceMixin.
        """
        self._last_online_voters = set(state.get("last_online_voters", []))

        # Restore voter state history (prune old entries)
        raw_history = state.get("voter_state_history", {})
        cutoff = time.time() - self.FLAP_WINDOW_SECONDS * 2
        self._voter_state_history = {}
        for voter_id, history in raw_history.items():
            pruned = [(ts, online) for ts, online in history if ts > cutoff]
            if pruned:
                self._voter_state_history[voter_id] = pruned

        self._flapping_voters = set(state.get("flapping_voters", []))
        self._quorum_recovery_attempts = state.get("quorum_recovery_attempts", 0)
        self._quorum_recovery_triggered = state.get("quorum_recovery_triggered", 0)
        self._last_voter_quorum_check = state.get("last_voter_quorum_check", {})
        # January 4, 2026: Restore quorum loss state for continued aggressive recovery
        self._quorum_lost = state.get("quorum_lost", False)
        # January 4, 2026: Restore early escalation counter
        self._quorum_lost_healing_attempts = state.get("quorum_lost_healing_attempts", 0)

        # Log restoration summary
        persisted_at = state.get("persisted_at", 0)
        age_seconds = time.time() - persisted_at if persisted_at else float("inf")
        logger.info(
            f"[P2PRecovery] Restored voter state: "
            f"{len(self._last_online_voters)} voters, "
            f"{len(self._flapping_voters)} flapping, "
            f"age={age_seconds:.1f}s"
        )

    def _restore_voter_state(self) -> None:
        """Restore voter state from last snapshot.

        Jan 3, 2026 (Sprint 15.1): Called during __init__.
        """
        try:
            if hasattr(self, "_restore_latest_snapshot"):
                self._restore_latest_snapshot()
        except Exception as e:
            logger.debug(f"[P2PRecovery] No voter state to restore: {e}")

    def _record_voter_state_change(self, voter_id: str, is_online: bool) -> None:
        """Record a voter state change for flapping detection.

        Jan 3, 2026 (Sprint 15.1): Tracks state changes to detect flapping.
        A voter is considered flapping if it changes state >= FLAP_THRESHOLD
        times within FLAP_WINDOW_SECONDS.
        """
        now = time.time()

        # Initialize history for new voters
        if voter_id not in self._voter_state_history:
            self._voter_state_history[voter_id] = []

        # Add new state change
        history = self._voter_state_history[voter_id]
        history.append((now, is_online))

        # Prune old entries outside the flapping window
        cutoff = now - self.FLAP_WINDOW_SECONDS
        self._voter_state_history[voter_id] = [
            (ts, state) for ts, state in history if ts > cutoff
        ]

        # Count state changes (transitions, not just entries)
        transitions = 0
        pruned_history = self._voter_state_history[voter_id]
        for i in range(1, len(pruned_history)):
            if pruned_history[i][1] != pruned_history[i-1][1]:
                transitions += 1

        # Update flapping status
        was_flapping = voter_id in self._flapping_voters
        is_flapping = transitions >= self.FLAP_THRESHOLD

        if is_flapping and not was_flapping:
            self._flapping_voters.add(voter_id)
            logger.warning(
                f"[P2PRecovery] Voter {voter_id} is flapping: "
                f"{transitions} state changes in {self.FLAP_WINDOW_SECONDS}s"
            )
            # Emit flapping event
            self._emit_voter_flapping_event(voter_id, transitions)
        elif not is_flapping and was_flapping:
            self._flapping_voters.discard(voter_id)
            logger.info(f"[P2PRecovery] Voter {voter_id} stabilized (no longer flapping)")

    def _emit_voter_flapping_event(self, voter_id: str, transitions: int) -> None:
        """Emit event when voter is detected as flapping.

        Jan 3, 2026 (Sprint 15.1): Alerts monitoring systems to unstable voters.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                DataEventType.VOTER_FLAPPING,
                {
                    "voter_id": voter_id,
                    "transitions": transitions,
                    "window_seconds": self.FLAP_WINDOW_SECONDS,
                    "threshold": self.FLAP_THRESHOLD,
                },
                context="P2PRecovery",
            )
        except ImportError as e:
            logger.debug(f"[P2PRecovery] Failed to emit VOTER_FLAPPING event: {e}")

    def get_flapping_voters(self) -> set[str]:
        """Get the set of currently flapping voters.

        Jan 3, 2026 (Sprint 15.1): Useful for excluding unstable voters
        from quorum calculations or prioritizing their investigation.
        """
        return self._flapping_voters.copy()


# =============================================================================
# Singleton Accessor
# =============================================================================


def get_p2p_recovery_daemon() -> P2PRecoveryDaemon:
    """Get the singleton P2PRecovery instance."""
    return P2PRecoveryDaemon.get_instance()


def reset_p2p_recovery_daemon() -> None:
    """Reset the singleton (for testing)."""
    P2PRecoveryDaemon.reset_instance()

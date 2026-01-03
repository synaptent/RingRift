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
from dataclasses import dataclass
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

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
        max_leader_gap_seconds: Maximum seconds without a leader before forcing election (default: 120)
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
    health_endpoint: str = "http://localhost:8770/status"
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
    # Dec 29, 2025: Self-healing for quorum and leader gaps
    max_leader_gap_seconds: int = 45  # Jan 2026: Reduced from 120s for faster leader recovery
    quorum_recovery_enabled: bool = True
    leader_election_endpoint: str = "http://localhost:8770/election/start"
    # Jan 2026: Proactive voter quorum monitoring
    voter_quorum_monitoring_enabled: bool = True
    min_voters_for_healthy_quorum: int = 5  # Trigger recovery if < 5 of 7 voters healthy
    voter_quorum_size: int = 4  # Quorum threshold (out of 7 total voters)

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


class P2PRecoveryDaemon(HandlerBase):
    """Daemon that monitors P2P cluster health and triggers auto-recovery.

    Workflow:
    1. Periodically check P2P /status endpoint
    2. Track consecutive failures
    3. After max_consecutive_failures, restart P2P orchestrator
    4. Respect cooldown period between restarts

    Events Emitted:
    - P2P_RESTART_TRIGGERED: When restart is initiated
    - P2P_HEALTH_RECOVERED: When P2P becomes healthy after being unhealthy
    """

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
        """Get the effective base cooldown based on NAT-blocked status.

        Jan 2026: NAT-blocked nodes use shorter cooldown for faster recovery.
        """
        if self._check_is_nat_blocked():
            return self.config.nat_blocked_cooldown_seconds
        return self.config.restart_cooldown_seconds

    # =========================================================================
    # Event Subscriptions (December 30, 2025)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions for voter health events.

        Dec 30, 2025: Subscribe to voter health events from VoterHealthMonitorDaemon
        to coordinate emergency quorum recovery.
        """
        return {
            "QUORUM_LOST": self._on_quorum_lost,
            "QUORUM_AT_RISK": self._on_quorum_at_risk,
            "VOTER_OFFLINE": self._on_voter_offline,
        }

    async def _on_quorum_lost(self, event: Any) -> None:
        """Handle QUORUM_LOST event - trigger emergency quorum restoration.

        Dec 30, 2025: Called when voter count drops below quorum threshold.
        Initiates aggressive recovery actions including:
        - Prioritized voter reconnection
        - Faster health check interval
        - Leader election if needed
        """
        logger.critical(
            f"QUORUM LOST - initiating emergency recovery "
            f"(online_voters={event.get('online_voters', '?')}, "
            f"quorum={event.get('quorum_size', '?')})"
        )

        # Track for health reporting
        self._quorum_recovery_attempts += 1

        # Trigger immediate quorum-aware reconnection prioritization
        await self._prioritize_quorum_reconnections()

        # Emit event for monitoring
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.QUORUM_RECOVERY_STARTED,
                trigger="quorum_lost_event",
                online_voters=event.get("online_voters", 0),
                quorum_size=event.get("quorum_size", 4),
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit QUORUM_RECOVERY_STARTED: {e}")

    async def _on_quorum_at_risk(self, event: Any) -> None:
        """Handle QUORUM_AT_RISK event - boost voter reconnection priority.

        Dec 30, 2025: Called when voter count is exactly at quorum threshold.
        Proactively attempts to reconnect offline voters before quorum is lost.
        """
        online_voters = event.get("online_voters", 0)
        offline_voters = event.get("offline_voters", [])
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
        voter_id = event.get("voter_id", "unknown")
        reason = event.get("reason", "unknown")
        logger.warning(f"Voter {voter_id} went offline: {reason}")

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

    async def _check_p2p_health(self) -> tuple[bool, dict[str, Any]]:
        """Check P2P orchestrator health via /status endpoint.

        Returns:
            Tuple of (is_healthy, status_dict)
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.health_endpoint,
                    timeout=aiohttp.ClientTimeout(total=self.config.health_timeout_seconds),
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

        return is_isolated, details

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
                if self._quorum_at_risk_consecutive > 0:
                    logger.info(
                        f"Voter quorum restored: {online_count}/{total_voters} voters online "
                        f"(was at risk for {self._quorum_at_risk_consecutive} checks)"
                    )
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
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.QUORUM_RECOVERY_STARTED if severity == "QUORUM_LOST"
                else DataEventType.QUORUM_AT_RISK,
                trigger="proactive_monitoring",
                offline_voters=list(offline_voters),
                severity=severity,
                consecutive_checks=self._quorum_at_risk_consecutive,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit quorum event: {e}")

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
        try:
            # Emit event that ClusterHealingLoop can subscribe to
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.VOTER_HEALING_REQUESTED,
                offline_voters=list(offline_voters),
                priority="urgent",
                source="P2PRecoveryDaemon",
            )
            logger.info(f"Requested urgent healing for {len(offline_voters)} offline voters")
        except Exception as e:
            logger.debug(f"Failed to emit VOTER_HEALING_REQUESTED: {e}")

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

    async def _restart_p2p(self) -> None:
        """Restart the P2P orchestrator process.

        December 2025: Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
        Dec 30, 2025: Tracks restart attempts for exponential backoff.
        """
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
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.QUORUM_PRIORITY_RECONNECT,
                prioritized_nodes=prioritized_nodes,
                voters_needed=len(prioritized_nodes),
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit QUORUM_PRIORITY_RECONNECT: {e}")

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
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.P2P_RESTART_TRIGGERED,
                consecutive_failures=self._consecutive_failures,
                last_status=self._last_status,
                total_restarts=self._total_restarts,
                unhealthy_duration_seconds=time.time() - self._last_healthy_time,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit P2P_RESTART_TRIGGERED: {e}")

    async def _emit_recovery_event(self, status: dict[str, Any]) -> None:
        """Emit event when P2P recovers."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.P2P_HEALTH_RECOVERED,
                alive_peers=status.get("alive_peers", 0),
                leader_id=status.get("leader_id"),
                recovery_duration_seconds=time.time() - self._last_healthy_time,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit P2P_HEALTH_RECOVERED: {e}")

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
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.NETWORK_ISOLATION_DETECTED,
                p2p_peers=isolation_details.get("p2p_peers", 0),
                tailscale_peers=isolation_details.get("tailscale_peers", 0),
                peer_ratio=isolation_details.get("peer_ratio", 0),
                consecutive_checks=self._consecutive_isolation_checks,
                isolation_triggered_restarts=self._isolation_triggered_restarts,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit NETWORK_ISOLATION_DETECTED: {e}")

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
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            leader_gap_seconds = time.time() - self._last_leader_seen_time
            emit_data_event(
                DataEventType.LEADER_ELECTION_TRIGGERED,
                reason="leader_gap",
                leader_gap_seconds=leader_gap_seconds,
                threshold_seconds=self.config.max_leader_gap_seconds,
                total_leader_gap_elections=self._leader_gap_elections_triggered,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit LEADER_ELECTION_TRIGGERED: {e}")

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
        }

        return base_status

    def is_p2p_healthy(self) -> bool:
        """Check if P2P is currently healthy based on last check.

        Jan 2026: Uses NAT-aware threshold for healthiness check.
        """
        return self._consecutive_failures < self._get_effective_restart_threshold()


# =============================================================================
# Singleton Accessor
# =============================================================================


def get_p2p_recovery_daemon() -> P2PRecoveryDaemon:
    """Get the singleton P2PRecovery instance."""
    return P2PRecoveryDaemon.get_instance()


def reset_p2p_recovery_daemon() -> None:
    """Reset the singleton (for testing)."""
    P2PRecoveryDaemon.reset_instance()

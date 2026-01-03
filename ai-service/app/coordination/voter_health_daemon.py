"""Voter Health Monitor Daemon for P2P Quorum Reliability.

Continuously monitors individual voter node health with multi-transport probing
to detect issues before quorum is lost.

December 30, 2025: Added for 48-hour autonomous operation reliability.

Features:
- Per-voter health state tracking
- Multi-transport fallback: P2P HTTP → Tailscale → SSH
- VOTER_OFFLINE/VOTER_ONLINE events for individual voters
- QUORUM_LOST/QUORUM_RESTORED events for quorum state changes
- QUORUM_AT_RISK early warning when quorum is marginal

Integration:
- Uses QuorumRecoveryManager for quorum state
- Uses cluster_config for voter IPs
- P2PRecoveryDaemon subscribes to quorum events for emergency recovery

Usage:
    from app.coordination.voter_health_daemon import (
        VoterHealthMonitorDaemon,
        get_voter_health_daemon,
    )

    daemon = get_voter_health_daemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VoterHealthConfig:
    """Configuration for voter health monitoring.

    Attributes:
        check_interval_seconds: How often to check voter health (default: 30s)
        consecutive_failures_before_offline: Failures before marking offline (default: 2)
        p2p_timeout_seconds: Timeout for P2P HTTP check (default: 5s)
        tailscale_timeout_seconds: Timeout for Tailscale ping (default: 10s)
        ssh_timeout_seconds: Timeout for SSH check (default: 15s)
        enable_ssh_fallback: Whether to try SSH as last resort (default: True)
        quorum_warning_threshold: Emit QUORUM_AT_RISK when only N voters online (default: quorum_size + 1)
        quorum_size: Minimum voters needed for quorum (default: 4)
        startup_grace_seconds: Grace period after startup before checking (default: 30s)
    """

    enabled: bool = True
    check_interval_seconds: int = 30  # Faster than P2P recovery (60s)
    # January 3, 2026: Increased from 2 to 3 to reduce false positives from
    # transient network hiccups. Slightly slower offline detection but fewer
    # spurious QUORUM_AT_RISK alerts.
    consecutive_failures_before_offline: int = 3
    p2p_timeout_seconds: float = 5.0
    tailscale_timeout_seconds: float = 10.0
    ssh_timeout_seconds: float = 15.0
    enable_ssh_fallback: bool = True
    quorum_size: int = 4
    quorum_warning_threshold: int = 5  # Warn when only 5 voters online (quorum + 1)
    # January 3, 2026: Reverted 10s→30s after Session 8 analysis showed
    # P2P initialization takes 8-12s; 10s caused false-positive restarts.
    startup_grace_seconds: int = 30
    p2p_port: int = 8770

    @classmethod
    def from_env(cls) -> "VoterHealthConfig":
        """Load configuration from environment variables."""
        config = cls()

        if os.environ.get("RINGRIFT_VOTER_HEALTH_ENABLED"):
            config.enabled = os.environ.get("RINGRIFT_VOTER_HEALTH_ENABLED", "1") == "1"

        if os.environ.get("RINGRIFT_VOTER_HEALTH_INTERVAL"):
            try:
                config.check_interval_seconds = int(
                    os.environ.get("RINGRIFT_VOTER_HEALTH_INTERVAL", "30")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_VOTER_HEALTH_FAILURES"):
            try:
                config.consecutive_failures_before_offline = int(
                    os.environ.get("RINGRIFT_VOTER_HEALTH_FAILURES", "2")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_VOTER_HEALTH_SSH_FALLBACK"):
            config.enable_ssh_fallback = (
                os.environ.get("RINGRIFT_VOTER_HEALTH_SSH_FALLBACK", "1") == "1"
            )

        if os.environ.get("RINGRIFT_P2P_PORT"):
            try:
                config.p2p_port = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))
            except ValueError:
                pass

        # Load quorum size from coordination defaults if available
        try:
            from app.config.coordination_defaults import P2PDefaults
            config.quorum_size = getattr(P2PDefaults, "DEFAULT_QUORUM", 4)
            config.quorum_warning_threshold = config.quorum_size + 1
        except ImportError:
            pass

        return config


# =============================================================================
# Per-Voter Health State
# =============================================================================


@dataclass
class VoterHealthState:
    """Per-voter health tracking.

    Tracks the health state of each individual voter node.
    """

    voter_id: str
    tailscale_ip: str = ""
    ssh_host: str = ""
    is_online: bool = True
    last_seen: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    last_successful_transport: str = "unknown"
    failure_reason: str = ""
    last_check_time: float = 0.0
    total_checks: int = 0
    total_failures: int = 0


# =============================================================================
# Voter Health Monitor Daemon
# =============================================================================


class VoterHealthMonitorDaemon(HandlerBase):
    """Monitors individual voter health with multi-transport probing.

    Workflow:
    1. Load voter list from cluster config
    2. Probe each voter using P2P HTTP → Tailscale → SSH fallback
    3. Track consecutive failures per voter
    4. Emit VOTER_OFFLINE when voter becomes unreachable
    5. Emit VOTER_ONLINE when voter recovers
    6. Emit QUORUM_LOST/QUORUM_RESTORED/QUORUM_AT_RISK for quorum state changes

    Events Emitted:
    - VOTER_OFFLINE: Individual voter became unreachable
    - VOTER_ONLINE: Individual voter recovered
    - QUORUM_LOST: Quorum threshold crossed (was OK, now lost)
    - QUORUM_RESTORED: Quorum threshold crossed (was lost, now OK)
    - QUORUM_AT_RISK: Quorum is marginal (e.g., exactly at threshold)
    """

    _event_source = "VoterHealthMonitor"

    def __init__(self, config: VoterHealthConfig | None = None):
        self._daemon_config = config or VoterHealthConfig.from_env()
        super().__init__(
            name="VoterHealthMonitor",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )
        self._voter_states: dict[str, VoterHealthState] = {}
        self._voter_ips: dict[str, str] = {}  # voter_id -> tailscale_ip
        self._voter_ssh: dict[str, str] = {}  # voter_id -> ssh_host
        self._startup_time = time.time()
        self._had_quorum = True  # Assume we start with quorum
        self._quorum_at_risk_emitted = False
        self._stats_extra = {
            "voters_online": 0,
            "voters_offline": 0,
            "quorum_lost_count": 0,
            "quorum_restored_count": 0,
        }

    @property
    def config(self) -> VoterHealthConfig:
        """Return daemon configuration."""
        return self._daemon_config

    # =========================================================================
    # Initialization
    # =========================================================================

    async def _on_start(self) -> None:
        """Load voter list from cluster config."""
        await super()._on_start()
        self._load_voters()

    def _load_voters(self) -> None:
        """Load voter list and IPs from cluster configuration."""
        try:
            from app.config.cluster_config import get_p2p_voters, load_cluster_config

            voters = get_p2p_voters()
            config = load_cluster_config()

            for voter_id in voters:
                host_info = config.hosts_raw.get(voter_id, {})
                tailscale_ip = host_info.get("tailscale_ip", "")
                ssh_host = host_info.get("ssh_host", "")

                self._voter_ips[voter_id] = tailscale_ip
                self._voter_ssh[voter_id] = ssh_host

                # Initialize state for each voter
                self._voter_states[voter_id] = VoterHealthState(
                    voter_id=voter_id,
                    tailscale_ip=tailscale_ip,
                    ssh_host=ssh_host,
                )

            logger.info(
                f"VoterHealthMonitor: Loaded {len(voters)} voters from config"
            )

        except ImportError as e:
            # cluster_config module not available
            logger.error(f"VoterHealthMonitor: Failed to import cluster_config: {e}")
        except (KeyError, AttributeError, TypeError, OSError) as e:
            # Config structure issues or file access problems
            logger.error(f"VoterHealthMonitor: Failed to load voters: {e}")

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main daemon cycle - probe all voters and emit events."""
        # Skip during startup grace period
        if time.time() - self._startup_time < self._daemon_config.startup_grace_seconds:
            logger.debug("VoterHealthMonitor: In startup grace period, skipping")
            return

        if not self._voter_states:
            self._load_voters()
            if not self._voter_states:
                logger.warning("VoterHealthMonitor: No voters configured")
                return

        # Probe all voters concurrently
        probe_tasks = [
            self._probe_and_update(voter_id)
            for voter_id in self._voter_states
        ]
        await asyncio.gather(*probe_tasks, return_exceptions=True)

        # Check quorum status after all probes
        self._check_quorum_status()

    async def _probe_and_update(self, voter_id: str) -> None:
        """Probe a voter and update its state.

        Args:
            voter_id: The voter node identifier.
        """
        state = self._voter_states[voter_id]
        state.last_check_time = time.time()
        state.total_checks += 1

        is_reachable, transport = await self._probe_voter(voter_id)

        if is_reachable:
            # Voter is reachable
            was_offline = not state.is_online
            state.is_online = True
            state.last_seen = time.time()
            state.consecutive_failures = 0
            state.last_successful_transport = transport
            state.failure_reason = ""

            if was_offline:
                # Voter came back online
                logger.info(
                    f"VoterHealthMonitor: Voter {voter_id} is ONLINE via {transport}"
                )
                await self._emit_voter_online(voter_id, transport)
        else:
            # Voter is unreachable
            state.consecutive_failures += 1
            state.total_failures += 1
            state.failure_reason = transport  # transport contains error on failure

            if state.consecutive_failures >= self._daemon_config.consecutive_failures_before_offline:
                if state.is_online:
                    # Voter just went offline
                    state.is_online = False
                    logger.warning(
                        f"VoterHealthMonitor: Voter {voter_id} is OFFLINE "
                        f"after {state.consecutive_failures} failures: {transport}"
                    )
                    await self._emit_voter_offline(voter_id, transport)
            else:
                logger.debug(
                    f"VoterHealthMonitor: Voter {voter_id} probe failed "
                    f"({state.consecutive_failures}/{self._daemon_config.consecutive_failures_before_offline})"
                )

    # =========================================================================
    # Multi-Transport Probing
    # =========================================================================

    async def _probe_voter(self, voter_id: str) -> tuple[bool, str]:
        """Probe voter using multi-transport fallback.

        Tries: P2P HTTP → Tailscale ping → SSH (if enabled)

        Args:
            voter_id: The voter node identifier.

        Returns:
            Tuple of (is_reachable, transport_or_error)
        """
        tailscale_ip = self._voter_ips.get(voter_id, "")
        ssh_host = self._voter_ssh.get(voter_id, "")

        # Try P2P HTTP first
        if tailscale_ip:
            if await self._check_p2p_reachable(tailscale_ip):
                return True, "p2p_http"

        # Try Tailscale ping
        if tailscale_ip:
            if await self._check_tailscale_reachable(tailscale_ip):
                return True, "tailscale_ping"

        # Try SSH as last resort
        if self._daemon_config.enable_ssh_fallback and ssh_host:
            if await self._check_ssh_reachable(ssh_host):
                return True, "ssh"

        return False, "all_transports_failed"

    async def _check_p2p_reachable(self, tailscale_ip: str) -> bool:
        """Check if voter is reachable via P2P /health endpoint.

        Args:
            tailscale_ip: The voter's Tailscale IP.

        Returns:
            True if voter responds to P2P health check.
        """
        try:
            import aiohttp

            url = f"http://{tailscale_ip}:{self._daemon_config.p2p_port}/health"
            timeout = aiohttp.ClientTimeout(
                total=self._daemon_config.p2p_timeout_seconds
            )

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    return resp.status == 200

        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def _check_tailscale_reachable(self, tailscale_ip: str) -> bool:
        """Check if voter is reachable via Tailscale ping.

        Args:
            tailscale_ip: The voter's Tailscale IP.

        Returns:
            True if Tailscale ping succeeds.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "ping", "--c", "1", "--timeout", "5s", tailscale_ip,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(
                    proc.wait(),
                    timeout=self._daemon_config.tailscale_timeout_seconds,
                )
                return proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False

        except FileNotFoundError:
            # Tailscale not installed
            return False
        except Exception:
            return False

    async def _check_ssh_reachable(self, ssh_host: str) -> bool:
        """Check if voter is reachable via SSH.

        Args:
            ssh_host: The SSH host string (may include user@host:port).

        Returns:
            True if SSH connection succeeds.
        """
        try:
            # Parse SSH host
            host = ssh_host
            port = "22"
            if ":" in ssh_host:
                parts = ssh_host.rsplit(":", 1)
                host = parts[0]
                port = parts[1]

            # Use ssh with connection timeout
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-p", port,
                host,
                "exit", "0",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(
                    proc.wait(),
                    timeout=self._daemon_config.ssh_timeout_seconds,
                )
                return proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False

        except FileNotFoundError:
            # SSH not installed
            return False
        except Exception:
            return False

    # =========================================================================
    # Quorum Checking
    # =========================================================================

    def _check_quorum_status(self) -> None:
        """Check quorum status and emit events on state changes."""
        online_voters = sum(1 for s in self._voter_states.values() if s.is_online)
        total_voters = len(self._voter_states)
        quorum_size = self._daemon_config.quorum_size

        # Update stats
        self._stats_extra["voters_online"] = online_voters
        self._stats_extra["voters_offline"] = total_voters - online_voters

        has_quorum = online_voters >= quorum_size
        quorum_at_risk = online_voters <= self._daemon_config.quorum_warning_threshold

        # Check for quorum state changes
        if self._had_quorum and not has_quorum:
            # Quorum lost
            logger.critical(
                f"VoterHealthMonitor: QUORUM LOST! "
                f"Only {online_voters}/{total_voters} voters online (need {quorum_size})"
            )
            self._emit_quorum_lost(online_voters, total_voters)
            self._stats_extra["quorum_lost_count"] = (
                self._stats_extra.get("quorum_lost_count", 0) + 1
            )
            self._quorum_at_risk_emitted = False  # Reset risk warning

        elif not self._had_quorum and has_quorum:
            # Quorum restored
            logger.info(
                f"VoterHealthMonitor: QUORUM RESTORED! "
                f"{online_voters}/{total_voters} voters online"
            )
            self._emit_quorum_restored(online_voters, total_voters)
            self._stats_extra["quorum_restored_count"] = (
                self._stats_extra.get("quorum_restored_count", 0) + 1
            )
            self._quorum_at_risk_emitted = False  # Reset risk warning

        elif has_quorum and quorum_at_risk and not self._quorum_at_risk_emitted:
            # Quorum at risk (marginal)
            logger.warning(
                f"VoterHealthMonitor: QUORUM AT RISK! "
                f"Only {online_voters}/{total_voters} voters online (threshold: {quorum_size})"
            )
            self._emit_quorum_at_risk(online_voters, total_voters)
            self._quorum_at_risk_emitted = True

        elif has_quorum and not quorum_at_risk:
            # Quorum healthy, reset risk warning
            self._quorum_at_risk_emitted = False

        self._had_quorum = has_quorum

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_voter_offline(self, voter_id: str, reason: str) -> None:
        """Emit VOTER_OFFLINE event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            state = self._voter_states.get(voter_id)
            emit_data_event(
                DataEventType.VOTER_OFFLINE,
                voter_id=voter_id,
                reason=reason,
                last_seen=state.last_seen if state else 0,
                consecutive_failures=state.consecutive_failures if state else 0,
                source=self._event_source,
            )
        except Exception as e:
            logger.debug(f"Failed to emit VOTER_OFFLINE: {e}")

    async def _emit_voter_online(self, voter_id: str, transport: str) -> None:
        """Emit VOTER_ONLINE event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            state = self._voter_states.get(voter_id)
            emit_data_event(
                DataEventType.VOTER_ONLINE,
                voter_id=voter_id,
                transport=transport,
                downtime_seconds=(
                    time.time() - state.last_seen if state else 0
                ),
                source=self._event_source,
            )
        except Exception as e:
            logger.debug(f"Failed to emit VOTER_ONLINE: {e}")

    def _emit_quorum_lost(self, online_voters: int, total_voters: int) -> None:
        """Emit QUORUM_LOST event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            offline_voters = [
                s.voter_id for s in self._voter_states.values() if not s.is_online
            ]
            emit_data_event(
                DataEventType.QUORUM_LOST,
                online_voters=online_voters,
                total_voters=total_voters,
                quorum_size=self._daemon_config.quorum_size,
                offline_voters=offline_voters,
                source=self._event_source,
            )
        except Exception as e:
            logger.debug(f"Failed to emit QUORUM_LOST: {e}")

    def _emit_quorum_restored(self, online_voters: int, total_voters: int) -> None:
        """Emit QUORUM_RESTORED event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.QUORUM_RESTORED,
                online_voters=online_voters,
                total_voters=total_voters,
                quorum_size=self._daemon_config.quorum_size,
                source=self._event_source,
            )
        except Exception as e:
            logger.debug(f"Failed to emit QUORUM_RESTORED: {e}")

    def _emit_quorum_at_risk(self, online_voters: int, total_voters: int) -> None:
        """Emit QUORUM_AT_RISK event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.QUORUM_AT_RISK,
                online_voters=online_voters,
                total_voters=total_voters,
                quorum_size=self._daemon_config.quorum_size,
                margin=online_voters - self._daemon_config.quorum_size,
                source=self._event_source,
            )
        except Exception as e:
            logger.debug(f"Failed to emit QUORUM_AT_RISK: {e}")

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="VoterHealthMonitor not running",
                details={},
            )

        online_voters = sum(1 for s in self._voter_states.values() if s.is_online)
        total_voters = len(self._voter_states)
        has_quorum = online_voters >= self._daemon_config.quorum_size

        if not has_quorum:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"QUORUM LOST: {online_voters}/{total_voters} voters online",
                details=self._get_status_details(),
            )

        if online_voters <= self._daemon_config.quorum_warning_threshold:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Quorum at risk: {online_voters}/{total_voters} voters online",
                details=self._get_status_details(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Quorum healthy: {online_voters}/{total_voters} voters online",
            details=self._get_status_details(),
        )

    def _get_status_details(self) -> dict[str, Any]:
        """Get detailed status for health check."""
        online_voters = []
        offline_voters = []

        for state in self._voter_states.values():
            voter_info = {
                "voter_id": state.voter_id,
                "last_seen": state.last_seen,
                "last_transport": state.last_successful_transport,
                "consecutive_failures": state.consecutive_failures,
            }
            if state.is_online:
                online_voters.append(voter_info)
            else:
                voter_info["failure_reason"] = state.failure_reason
                offline_voters.append(voter_info)

        return {
            "online_voters": online_voters,
            "offline_voters": offline_voters,
            "quorum_size": self._daemon_config.quorum_size,
            "has_quorum": len(online_voters) >= self._daemon_config.quorum_size,
            "stats": dict(self._stats_extra),
        }

    def get_status(self) -> dict[str, Any]:
        """Return detailed status."""
        base_status = super().get_status()
        base_status["voter_health"] = self._get_status_details()
        return base_status

    def get_voter_state(self, voter_id: str) -> VoterHealthState | None:
        """Get the health state of a specific voter."""
        return self._voter_states.get(voter_id)

    def get_online_voters(self) -> list[str]:
        """Get list of online voter IDs."""
        return [s.voter_id for s in self._voter_states.values() if s.is_online]

    def get_offline_voters(self) -> list[str]:
        """Get list of offline voter IDs."""
        return [s.voter_id for s in self._voter_states.values() if not s.is_online]


# =============================================================================
# Singleton Accessor
# =============================================================================


def get_voter_health_daemon() -> VoterHealthMonitorDaemon:
    """Get the singleton VoterHealthMonitor instance."""
    return VoterHealthMonitorDaemon.get_instance()


def reset_voter_health_daemon() -> None:
    """Reset the singleton (for testing)."""
    VoterHealthMonitorDaemon.reset_instance()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "VoterHealthConfig",
    "VoterHealthState",
    "VoterHealthMonitorDaemon",
    "get_voter_health_daemon",
    "reset_voter_health_daemon",
]

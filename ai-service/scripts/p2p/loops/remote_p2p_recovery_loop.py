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
    REMOTE_P2P_RECOVERY_SUCCESS: Emitted when a node successfully recovered and joined mesh
    REMOTE_P2P_RECOVERY_FAILED: Emitted when recovery attempt failed (SSH or verification)

Session 17.25 (Jan 5, 2026): Added success/failure events for feedback loop integration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import BaseLoop, LoopStats

logger = logging.getLogger(__name__)

# Jan 5, 2026 (Session 17.29): Circuit breaker integration for SSH recovery
# Prevents wasted effort on permanently dead nodes
try:
    from app.coordination.node_circuit_breaker import get_node_circuit_breaker

    HAS_CIRCUIT_BREAKER = True
except ImportError:
    get_node_circuit_breaker = None  # type: ignore
    HAS_CIRCUIT_BREAKER = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RemoteP2PRecoveryConfig:
    """Configuration for remote P2P recovery."""

    # Interval between recovery cycles (seconds) - default 60 seconds
    # Jan 2026: Reduced from 120s to 60s for faster recovery
    # Nodes typically connect within 30-60s if healthy
    check_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_REMOTE_P2P_RECOVERY_INTERVAL", "60")
        )
    )

    # Maximum nodes to recover per cycle (prevent thundering herd)
    # Jan 2026: Increased from 5 to 10 for faster cluster recovery
    max_nodes_per_cycle: int = 10

    # SSH timeout per node (seconds)
    # Jan 5, 2026: Increased from 30s to 60s - high-load nodes need more time
    ssh_timeout_seconds: float = 60.0

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
    # Jan 2026: Reduced from 180s to 90s - balances recovery speed with stability
    # Nodes typically need 30-60s to start and join mesh
    retry_cooldown_seconds: float = 90.0  # 1.5 minutes (base cooldown)

    # Exponential backoff configuration
    # Jan 2026: Prevent pathological restart loops by backing off exponentially
    # after repeated restart attempts on the same node
    backoff_multiplier: float = 2.0  # Each restart attempt doubles the cooldown
    backoff_max_seconds: float = 600.0  # Max 10 min between restart attempts (was 1h)
    backoff_reset_after_stable_seconds: float = 300.0  # Reset backoff after 5 min stable (was 10 min)
    backoff_max_restart_count: int = 5  # After this many restarts, use max backoff

    # Whether to emit events on recovery
    emit_events: bool = True

    # Post-recovery verification timeout (seconds) - wait for node to appear in peers
    # Jan 2026: Increased from 60s to 120s - nodes need time to load state and bootstrap
    verification_timeout_seconds: float = 120.0

    # Verification poll interval (seconds)
    verification_poll_interval: float = 5.0


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class RemoteP2PRecoveryStats(LoopStats):
    """Statistics for remote P2P recovery operations.

    Inherits from LoopStats (Jan 24, 2026) to ensure type consistency with
    BaseLoop expectations. This fixes potential AttributeError issues when
    BaseLoop.run_forever() accesses fields like last_run_time, last_error_time, etc.

    Inherited from LoopStats:
        - name, total_runs, successful_runs, failed_runs, consecutive_errors
        - last_run_time, last_success_time, last_error_time, last_error_message
        - total_run_duration, last_run_duration
        - success_rate (property), avg_run_duration (property)
    """

    # Provide default for inherited required field
    name: str = "remote_p2p_recovery"

    # Domain-specific fields for recovery tracking
    nodes_recovered: int = 0
    nodes_verified: int = 0  # Nodes confirmed in P2P mesh after recovery
    nodes_failed: int = 0
    nodes_verification_failed: int = 0  # Started but didn't appear in mesh
    nodes_skipped_unreachable: int = 0  # Nodes skipped due to failed pre-flight check
    last_recovery_time: float = 0.0
    nodes_skipped_cooldown: int = 0
    nodes_skipped_backoff: int = 0  # Jan 2026: Nodes skipped due to exponential backoff
    nodes_skipped_circuit_broken: int = 0  # Jan 5, 2026: Nodes skipped due to circuit breaker OPEN
    ssh_key_missing: bool = False  # SSH key validation failed
    backoff_resets: int = 0  # Jan 2026: Count of nodes that had backoff reset after stability

    @property
    def cycles_run(self) -> int:
        """Alias for total_runs for backward compatibility."""
        return self.total_runs

    def to_dict(self) -> dict:
        """Convert stats to dictionary for JSON serialization.

        Extends parent's to_dict() with domain-specific recovery fields.
        """
        base_dict = super().to_dict()
        base_dict.update({
            "nodes_recovered": self.nodes_recovered,
            "nodes_verified": self.nodes_verified,
            "nodes_failed": self.nodes_failed,
            "nodes_verification_failed": self.nodes_verification_failed,
            "nodes_skipped_unreachable": self.nodes_skipped_unreachable,
            "last_recovery_time": self.last_recovery_time,
            "cycles_run": self.cycles_run,  # Property alias for backward compat
            "nodes_skipped_cooldown": self.nodes_skipped_cooldown,
            "nodes_skipped_backoff": self.nodes_skipped_backoff,
            "nodes_skipped_circuit_broken": self.nodes_skipped_circuit_broken,
            "backoff_resets": self.backoff_resets,
            "ssh_key_missing": self.ssh_key_missing,
        })
        return base_dict


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
    - IMPORTANT: Only runs on the leader node to prevent restart cascades
    """

    def __init__(
        self,
        get_alive_peer_ids: Callable[[], list[str]],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        config: RemoteP2PRecoveryConfig | None = None,
        is_leader: Callable[[], bool] | None = None,
    ):
        """Initialize remote P2P recovery loop.

        Args:
            get_alive_peer_ids: Callback returning list of alive peer node IDs
            emit_event: Optional callback to emit events (event_name, event_data)
            config: Recovery configuration
            is_leader: Callback returning True if this node is the leader.
                       CRITICAL: Only the leader should run recovery to prevent
                       all nodes trying to restart each other simultaneously.
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
        # Jan 3, 2026: Validate is_leader is callable to prevent "'bool' object is not callable"
        # This can happen if someone passes is_leader=True instead of is_leader=lambda: True
        if is_leader is not None and not callable(is_leader):
            logger.warning(
                f"[RemoteP2PRecovery] is_leader must be callable, got {type(is_leader).__name__}. "
                f"Using default (always False). Pass a lambda like: is_leader=lambda: my_check()"
            )
            is_leader = None
        self._is_leader = is_leader if is_leader is not None else (lambda: False)

        # Statistics
        self._stats = RemoteP2PRecoveryStats()

        # Track last attempt time per node to implement cooldown
        self._last_attempt: dict[str, float] = {}

        # Jan 2026: Track restart counts for exponential backoff
        # Increment on failed verification, reset after stable period
        self._restart_count: dict[str, int] = {}

        # Track when nodes were first seen stable (in mesh continuously)
        # Used to reset backoff after stable period
        self._first_stable_time: dict[str, float] = {}

        # Paramiko client (lazy loaded)
        self._paramiko: Any = None

        # Validate SSH key on initialization
        if not self._validate_ssh_key():
            self._stats.ssh_key_missing = True

    def _safe_emit_p2p_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit P2P event with fallback to safe_emit_event.

        Jan 5, 2026: Unified event emission pattern. Uses callback if provided,
        falls back to safe_emit_event from event_emission_helpers.
        """
        # Try callback first (backward compatibility)
        if self._emit_event:
            try:
                self._emit_event(event_type, payload)
                return
            except Exception as e:
                logger.debug(f"[RemoteP2PRecovery] Callback emit failed: {e}")

        # Fallback to safe_emit_event
        try:
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                event_type,
                payload,
                context="RemoteP2PRecovery",
                source="remote_p2p_recovery_loop",
            )
        except ImportError:
            pass  # Event modules not available

    def _validate_ssh_key(self) -> bool:
        """Check SSH key exists and has correct permissions.

        Returns:
            True if SSH key is valid, False otherwise.
        """
        from pathlib import Path

        key_path = Path(self.config.ssh_key_path).expanduser()
        if not key_path.exists():
            logger.warning(f"[RemoteP2PRecovery] SSH key not found: {key_path}")
            return False

        # Check permissions (should be 600 or more restrictive)
        try:
            mode = key_path.stat().st_mode & 0o777
            if mode > 0o600:
                logger.warning(
                    f"[RemoteP2PRecovery] SSH key has insecure permissions: "
                    f"{oct(mode)} (should be 0600 or more restrictive)"
                )
                # Don't fail, just warn - some systems may have different requirements
        except OSError as e:
            logger.warning(f"[RemoteP2PRecovery] Cannot check SSH key permissions: {e}")

        return True

    def _get_effective_cooldown(self, node_id: str) -> float:
        """Calculate effective cooldown with exponential backoff.

        Args:
            node_id: Node identifier

        Returns:
            Cooldown in seconds, exponentially increasing with restart count.
        """
        restart_count = self._restart_count.get(node_id, 0)
        if restart_count == 0:
            return self.config.retry_cooldown_seconds

        # Exponential backoff: base * (multiplier ^ min(count, max_count))
        effective_count = min(restart_count, self.config.backoff_max_restart_count)
        cooldown = self.config.retry_cooldown_seconds * (
            self.config.backoff_multiplier ** effective_count
        )
        return min(cooldown, self.config.backoff_max_seconds)

    def _check_and_reset_stable_nodes(self, alive_peer_ids: set[str]) -> None:
        """Check if nodes have been stable long enough to reset their backoff.

        A node is considered stable if it has been in the alive peers set
        continuously for backoff_reset_after_stable_seconds.

        Args:
            alive_peer_ids: Set of currently alive peer node IDs
        """
        now = time.time()
        reset_threshold = self.config.backoff_reset_after_stable_seconds

        for node_id in list(self._restart_count.keys()):
            if node_id in alive_peer_ids:
                # Node is alive - track when we first saw it stable
                if node_id not in self._first_stable_time:
                    self._first_stable_time[node_id] = now
                    logger.debug(
                        f"[RemoteP2PRecovery] {node_id} now stable, "
                        f"will reset backoff after {reset_threshold}s"
                    )
                elif now - self._first_stable_time[node_id] >= reset_threshold:
                    # Been stable long enough - reset backoff
                    old_count = self._restart_count.pop(node_id, 0)
                    self._first_stable_time.pop(node_id, None)
                    self._last_attempt.pop(node_id, None)
                    self._stats.backoff_resets += 1
                    logger.info(
                        f"[RemoteP2PRecovery] {node_id} stable for {reset_threshold}s, "
                        f"reset backoff (was {old_count} restarts)"
                    )
            else:
                # Node is not alive - reset the stable timer
                if node_id in self._first_stable_time:
                    del self._first_stable_time[node_id]

    async def _verify_recovery(self, node_id: str) -> bool:
        """Wait for recovered node to appear in alive peers.

        Args:
            node_id: Node identifier to wait for

        Returns:
            True if node appeared in peers within timeout, False otherwise.
        """
        start = time.time()
        timeout = self.config.verification_timeout_seconds
        poll_interval = self.config.verification_poll_interval

        while time.time() - start < timeout:
            alive_peers = set(self._get_alive_peer_ids())
            if node_id in alive_peers:
                logger.info(
                    f"[RemoteP2PRecovery] Node {node_id} verified in mesh after "
                    f"{time.time() - start:.1f}s"
                )
                return True
            await asyncio.sleep(poll_interval)

        logger.warning(
            f"[RemoteP2PRecovery] Node {node_id} did not appear in mesh within "
            f"{timeout}s timeout"
        )
        return False

    async def _check_ssh_reachable(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Quick TCP check if SSH port is open.

        This pre-flight check avoids wasting time on nodes that are completely
        unreachable, allowing us to skip them quickly.

        Args:
            host: SSH host/IP to check
            port: SSH port (usually 22)
            timeout: Connection timeout in seconds

        Returns:
            True if SSH port is reachable, False otherwise.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
            return False

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

        # CRITICAL: Only leader should run recovery to prevent restart cascade
        # If every node ran this, they'd all try to restart each other
        if not self._is_leader():
            logger.debug(
                "[RemoteP2PRecovery] Skipping cycle - not leader"
            )
            return

        # Skip if SSH key is missing
        if self._stats.ssh_key_missing:
            logger.debug(
                "[RemoteP2PRecovery] Skipping cycle - SSH key missing or invalid"
            )
            return

        paramiko = self._get_paramiko()
        if paramiko is None:
            return

        # Jan 5, 2026 (Session 17.29): Get circuit breaker for SSH recovery
        # This prevents wasted SSH attempts on nodes that consistently fail
        ssh_circuit_breaker = None
        if HAS_CIRCUIT_BREAKER and get_node_circuit_breaker is not None:
            try:
                ssh_circuit_breaker = get_node_circuit_breaker("ssh_recovery")
            except Exception as e:
                logger.debug(f"[RemoteP2PRecovery] Could not get circuit breaker: {e}")

        now = time.time()
        # Jan 1, 2026: Removed self._stats.cycles_run += 1 as base class handles
        # total_runs incrementing in run_forever() at lines 256 and 281

        # Get current alive peers
        alive_peer_ids = set(self._get_alive_peer_ids())
        if not alive_peer_ids and not self.config.dry_run:
            logger.debug("[RemoteP2PRecovery] No alive peers yet, skipping")
            return

        # Check if any nodes with backoff have been stable long enough to reset
        self._check_and_reset_stable_nodes(alive_peer_ids)

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

            # Skip if P2P is disabled for this node
            if not node_info.get("p2p_enabled", True):
                continue

            # Skip if no SSH access configured
            if not node_info.get("tailscale_ip") and not node_info.get("ssh_host"):
                continue

            # Skip if in cooldown (with exponential backoff for repeat offenders)
            last_attempt = self._last_attempt.get(node_id, 0)
            effective_cooldown = self._get_effective_cooldown(node_id)
            if now - last_attempt < effective_cooldown:
                restart_count = self._restart_count.get(node_id, 0)
                if restart_count > 0:
                    # This is exponential backoff, not just regular cooldown
                    self._stats.nodes_skipped_backoff += 1
                    logger.debug(
                        f"[RemoteP2PRecovery] Skipping {node_id}: "
                        f"backoff {effective_cooldown:.0f}s (restart #{restart_count})"
                    )
                else:
                    self._stats.nodes_skipped_cooldown += 1
                continue

            # Jan 5, 2026 (Session 17.32): Reset circuit breaker before recovery attempt
            # instead of skipping nodes with open CBs. This allows the recovery loop
            # to give nodes a fresh chance after the CB TTL has been reduced from 6h to 1h.
            # Previous behavior (Session 17.29) skipped CB-open nodes entirely, which
            # caused nodes to be isolated for hours after transient failures.
            if ssh_circuit_breaker is not None:
                if not ssh_circuit_breaker.can_check(node_id):
                    # Reset CB to give node another chance
                    ssh_circuit_breaker.reset(node_id)
                    logger.info(
                        f"[RemoteP2PRecovery] Reset circuit breaker for {node_id} before recovery"
                    )

            nodes_to_recover.append((node_id, node_info))

        # Jan 9, 2026: Prioritize recovery order for faster quorum restoration
        # Order: 1. Voters (quorum critical) 2. NAT-blocked 3. Others
        voter_ids = self._get_voter_node_ids()

        # Log urgently if voters need recovery
        voters_needing_recovery = [n for n, _ in nodes_to_recover if n in voter_ids]
        if voters_needing_recovery:
            logger.warning(
                f"[RemoteP2PRecovery] URGENT: {len(voters_needing_recovery)} voter(s) need recovery: "
                f"{voters_needing_recovery}"
            )

        nodes_to_recover.sort(
            key=lambda x: (
                # Priority 0: Voters (quorum critical)
                0 if x[0] in voter_ids else (
                    # Priority 1: NAT-blocked (need more help staying connected)
                    1 if x[1].get("nat_blocked") or x[1].get("force_relay_mode") else 2
                ),
                x[0],  # Then alphabetical for determinism
            )
        )

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

        # Count successes/failures and verify recoveries
        nodes_to_verify = []
        for (node_id, _), result in zip(nodes_this_cycle, results):
            self._last_attempt[node_id] = now
            if isinstance(result, Exception):
                logger.warning(f"[RemoteP2PRecovery] Error recovering {node_id}: {result}")
                self._stats.nodes_failed += 1
            elif result:
                self._stats.nodes_recovered += 1
                nodes_to_verify.append(node_id)
            else:
                self._stats.nodes_failed += 1

        # Post-recovery verification: wait for nodes to appear in mesh
        if nodes_to_verify:
            logger.info(
                f"[RemoteP2PRecovery] Verifying {len(nodes_to_verify)} nodes joined mesh..."
            )
            verify_tasks = [self._verify_recovery(node_id) for node_id in nodes_to_verify]
            verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)

            for node_id, verified in zip(nodes_to_verify, verify_results):
                if isinstance(verified, Exception):
                    logger.warning(
                        f"[RemoteP2PRecovery] Verification error for {node_id}: {verified}"
                    )
                    self._stats.nodes_verification_failed += 1
                    # Increment restart count for exponential backoff
                    self._restart_count[node_id] = self._restart_count.get(node_id, 0) + 1
                    logger.info(
                        f"[RemoteP2PRecovery] {node_id} verification failed, "
                        f"restart count now {self._restart_count[node_id]} "
                        f"(next cooldown: {self._get_effective_cooldown(node_id):.0f}s)"
                    )
                    # Jan 5, 2026 (Session 17.29): Record failure to circuit breaker
                    if ssh_circuit_breaker is not None:
                        ssh_circuit_breaker.record_failure(node_id, verified)
                    # Session 17.25: Emit failure event for feedback loop integration
                    if self.config.emit_events:
                        self._safe_emit_p2p_event(
                            "REMOTE_P2P_RECOVERY_FAILED",
                            {
                                "node_id": node_id,
                                "recovery_type": "ssh_restart",
                                "failure_reason": f"verification_exception: {type(verified).__name__}",
                                "restart_count": self._restart_count.get(node_id, 0),
                                "timestamp": time.time(),
                            },
                        )
                elif verified:
                    self._stats.nodes_verified += 1
                    # Successful verification - start tracking stability
                    # (backoff will be reset after stable period)
                    if node_id in self._restart_count:
                        self._first_stable_time[node_id] = now
                    # Jan 5, 2026 (Session 17.29): Record success to circuit breaker
                    # This helps the circuit breaker transition from HALF_OPEN to CLOSED
                    if ssh_circuit_breaker is not None:
                        ssh_circuit_breaker.record_success(node_id)
                    # Session 17.25: Emit success event for feedback loop integration
                    if self.config.emit_events:
                        self._safe_emit_p2p_event(
                            "REMOTE_P2P_RECOVERY_SUCCESS",
                            {
                                "node_id": node_id,
                                "recovery_type": "ssh_restart",
                                "duration_seconds": time.time() - self._last_attempt.get(node_id, now),
                                "timestamp": time.time(),
                            },
                        )
                        # Session 17.28: Emit NODE_READY_FOR_WORK for work queue pre-population
                        # This signals that the recovered node is ready to accept work
                        # immediately, reducing idle time from 120s to ~10s
                        node_capacity = self._get_node_capacity(node_id)
                        self._safe_emit_p2p_event(
                            "NODE_READY_FOR_WORK",
                            {
                                "node_id": node_id,
                                "priority": "high",
                                "capacity": node_capacity,
                                "recovery_source": "remote_p2p_recovery",
                                "timestamp": time.time(),
                            },
                        )
                else:
                    self._stats.nodes_verification_failed += 1
                    # Increment restart count for exponential backoff
                    self._restart_count[node_id] = self._restart_count.get(node_id, 0) + 1
                    logger.info(
                        f"[RemoteP2PRecovery] {node_id} did not appear in mesh, "
                        f"restart count now {self._restart_count[node_id]} "
                        f"(next cooldown: {self._get_effective_cooldown(node_id):.0f}s)"
                    )
                    # Jan 5, 2026 (Session 17.29): Record failure to circuit breaker
                    # This helps circuit breaker track consistently failing nodes
                    if ssh_circuit_breaker is not None:
                        ssh_circuit_breaker.record_failure(node_id)
                    # Session 17.25: Emit failure event for feedback loop integration
                    if self.config.emit_events:
                        self._safe_emit_p2p_event(
                            "REMOTE_P2P_RECOVERY_FAILED",
                            {
                                "node_id": node_id,
                                "recovery_type": "ssh_restart",
                                "failure_reason": "verification_failed",
                                "restart_count": self._restart_count.get(node_id, 0),
                                "timestamp": time.time(),
                            },
                        )

        self._stats.last_recovery_time = now

        # Jan 9, 2026: Update interval based on voter status
        # Faster checks when voters are missing from the mesh
        self.interval = self._get_adaptive_check_interval()

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
        except FileNotFoundError:
            logger.warning("[RemoteP2PRecovery] Config file not found")
            return {}
        except PermissionError as e:
            logger.warning(f"[RemoteP2PRecovery] Permission denied reading config: {e}")
            return {}
        except (OSError, IOError) as e:
            logger.warning(f"[RemoteP2PRecovery] Failed to read config file: {e}")
            return {}
        except ImportError:
            logger.warning("[RemoteP2PRecovery] yaml module not available")
            return {}

    def _get_voter_node_ids(self) -> set[str]:
        """Get voter node IDs from distributed_hosts.yaml.

        Jan 9, 2026: Added for voter prioritization. Voter nodes are critical
        for quorum and should be recovered first.
        """
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config",
                "distributed_hosts.yaml",
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)
            voters = config.get("p2p_voters", [])
            return set(voters) if voters else set()
        except Exception:
            return set()

    def _get_adaptive_check_interval(self) -> float:
        """Get adaptive check interval - faster when voters need recovery.

        Jan 9, 2026: Added to reduce recovery time for voter nodes.
        When voters are missing, check every 30s instead of the default 60s.
        """
        base_interval = self.config.check_interval_seconds

        try:
            # Get alive peers
            alive_peer_ids = set(self._get_alive_peer_ids())

            # Get voter IDs
            voter_ids = self._get_voter_node_ids()

            # Check if any voters are missing
            missing_voters = voter_ids - alive_peer_ids
            if missing_voters:
                # Use faster interval when voters missing
                return min(base_interval, 30.0)
        except Exception:
            pass

        return base_interval

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

        # Pre-flight check: quickly verify SSH port is reachable
        if not await self._check_ssh_reachable(host, port):
            logger.debug(f"[RemoteP2PRecovery] Skipping {node_id} ({host}:{port}): SSH port unreachable")
            self._stats.nodes_skipped_unreachable += 1
            return False

        logger.info(f"[RemoteP2PRecovery] Starting P2P on {node_id} ({host})")

        try:
            # Run SSH operations in thread pool to not block event loop
            result = await asyncio.to_thread(
                self._ssh_start_p2p, node_id, host, port, user, node_info, paramiko
            )
            return result
        except Exception as e:
            logger.warning(f"[RemoteP2PRecovery] SSH failed for {node_id}: {e}")
            return False

    def _ssh_start_p2p(
        self, node_id: str, host: str, port: int, user: str, node_info: dict, paramiko: Any
    ) -> bool:
        """Execute SSH commands to start P2P on a node (runs in thread).

        Args:
            node_id: Node identifier
            host: SSH host/IP
            port: SSH port
            user: SSH user
            node_info: Node configuration from distributed_hosts.yaml
            paramiko: paramiko module

        Returns:
            True if P2P was started successfully

        January 13, 2026: Added relay failover for NAT-blocked nodes. When a node
        has force_relay_mode=true, we check which relays are healthy and use the
        first available one via --relay-peers.
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Use per-node SSH key if specified, otherwise fall back to default
        ssh_key = node_info.get("ssh_key", self.config.ssh_key_path)
        ssh_key = os.path.expanduser(ssh_key)

        # Get ringrift path from node config (supports Vast.ai /workspace, etc.)
        ringrift_path = node_info.get("ringrift_path", "~/ringrift/ai-service")

        try:
            client.connect(
                host,
                port=port,
                username=user,
                key_filename=ssh_key,
                timeout=self.config.ssh_timeout_seconds,
                banner_timeout=self.config.ssh_timeout_seconds,
            )

            # First kill any existing P2P process and clean up screen sessions
            # Jan 2026: Added screen cleanup to prevent dead session accumulation
            client.exec_command(
                "pkill -f p2p_orchestrator 2>/dev/null || true; "
                "screen -X -S p2p quit 2>/dev/null || true; "
                "screen -wipe 2>/dev/null || true"
            )
            time.sleep(2)  # Allow time for cleanup

            # Pull latest code and start P2P
            # Use python3 explicitly since 'python' may not exist on all nodes
            # January 2026: Always include --advertise-host for Tailscale nodes
            tailscale_ip = node_info.get("tailscale_ip", "")
            advertise_arg = f"--advertise-host {tailscale_ip} " if tailscale_ip else ""

            # January 13, 2026: Relay failover for NAT-blocked nodes
            # Check relay health and use first available healthy relay
            relay_arg = ""
            if node_info.get("nat_blocked") or node_info.get("force_relay_mode"):
                healthy_relays = self._get_healthy_relays_for_node(node_info)
                configured_primary = node_info.get("relay_primary", "")

                if healthy_relays:
                    chosen_relay = healthy_relays[0]
                    if chosen_relay != configured_primary:
                        logger.info(
                            f"[RemoteP2PRecovery] Relay FAILOVER for {node_id}: "
                            f"{configured_primary} -> {chosen_relay}"
                        )
                    relay_arg = f"--relay-peers {chosen_relay} "
                else:
                    # No healthy relays - try with configured primary anyway
                    if configured_primary:
                        logger.warning(
                            f"[RemoteP2PRecovery] No healthy relays for {node_id}, "
                            f"using configured primary: {configured_primary}"
                        )
                        relay_arg = f"--relay-peers {configured_primary} "

            cmd = f"""cd {ringrift_path} && \
git pull origin main 2>/dev/null || true && \
PYTHONPATH=. nohup python3 scripts/p2p_orchestrator.py --node-id {node_id} --port 8770 {advertise_arg}{relay_arg}> logs/p2p.log 2>&1 &"""

            stdin, stdout, stderr = client.exec_command(cmd)
            stdout.channel.recv_exit_status()  # Wait for completion

            # Wait for process to start
            time.sleep(3)  # Sync function running in thread - use time.sleep

            # Verify it started
            _, stdout2, _ = client.exec_command("pgrep -f p2p_orchestrator")
            pid = stdout2.read().decode().strip()

            client.close()

            if pid:
                logger.info(f"[RemoteP2PRecovery] Started P2P on {node_id} (PID: {pid})")

                # Emit event - Jan 5, 2026: Use safe_emit_event for consistent handling
                if self.config.emit_events:
                    self._safe_emit_p2p_event(
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

        except paramiko.AuthenticationException:
            logger.error(f"[RemoteP2PRecovery] Auth failed for {node_id}: SSH key rejected or wrong user")
            return False
        except paramiko.SSHException as e:
            logger.error(f"[RemoteP2PRecovery] SSH protocol error for {node_id}: {e}")
            return False
        except socket.timeout:
            logger.error(f"[RemoteP2PRecovery] Timeout connecting to {node_id} ({host}): host unreachable")
            return False
        except socket.error as e:
            logger.error(f"[RemoteP2PRecovery] Network error for {node_id} ({host}): {e}")
            return False
        except OSError as e:
            logger.error(f"[RemoteP2PRecovery] OS error for {node_id}: {e}")
            return False
        finally:
            try:
                client.close()
            except (OSError, AttributeError):
                # OSError: Socket already closed or network error
                # AttributeError: Client never fully initialized
                pass

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        # Include nodes currently in backoff with their cooldown times
        nodes_in_backoff = {
            node_id: {
                "restart_count": count,
                "effective_cooldown_seconds": self._get_effective_cooldown(node_id),
            }
            for node_id, count in self._restart_count.items()
            if count > 0
        }

        return {
            **self._stats.to_dict(),
            "nodes_in_backoff": nodes_in_backoff,
            "config": {
                "interval_seconds": self.config.check_interval_seconds,
                "max_per_cycle": self.config.max_nodes_per_cycle,
                "retry_cooldown_seconds": self.config.retry_cooldown_seconds,
                "backoff_multiplier": self.config.backoff_multiplier,
                "backoff_max_seconds": self.config.backoff_max_seconds,
                "backoff_reset_after_stable_seconds": self.config.backoff_reset_after_stable_seconds,
                "enabled": self.config.enabled,
                "dry_run": self.config.dry_run,
            },
        }

    # =========================================================================
    # Relay Failover Support (January 13, 2026)
    # =========================================================================

    def _tcp_probe(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """TCP probe to check if a host:port is reachable.

        January 13, 2026: Added for relay health checking. Uses a simple TCP
        connect to verify the P2P port is listening.

        Args:
            host: Hostname or IP to probe
            port: Port number to probe
            timeout: Connection timeout in seconds

        Returns:
            True if connection succeeded, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except socket.timeout:
            return False
        except socket.error:
            return False
        except Exception:
            return False

    def _get_relay_ip(self, relay_node_id: str) -> str | None:
        """Get the best IP address for a relay node.

        January 13, 2026: Added for relay failover support. Prefers Tailscale IP
        since Lambda nodes are on the Tailscale mesh.

        Args:
            relay_node_id: Node ID of the relay

        Returns:
            IP address string or None if not found
        """
        configured_nodes = self._get_configured_nodes()
        relay_info = configured_nodes.get(relay_node_id, {})

        # Prefer Tailscale IP (works for mesh connections)
        tailscale_ip = relay_info.get("tailscale_ip")
        if tailscale_ip:
            return tailscale_ip

        # Fallback to public IP
        public_ip = relay_info.get("ip")
        if public_ip:
            return public_ip

        # Try SSH host
        ssh_host = relay_info.get("ssh_host")
        if ssh_host:
            return ssh_host

        return None

    def _check_relay_health(self, relay_ids: list[str]) -> list[str]:
        """Return list of healthy relay node IDs.

        January 13, 2026: Added for relay failover in RemoteP2PRecoveryLoop.
        Checks each configured relay to find which ones are reachable via TCP
        probe to their P2P port (8770).

        Args:
            relay_ids: List of relay node IDs to check

        Returns:
            List of healthy relay node IDs (ordered by input order)
        """
        healthy = []
        for relay_id in relay_ids:
            if not relay_id:
                continue

            ip = self._get_relay_ip(relay_id)
            if not ip:
                logger.debug(f"[RemoteP2PRecovery] No IP found for relay {relay_id}")
                continue

            # Probe P2P port (8770)
            if self._tcp_probe(ip, 8770, timeout=5.0):
                logger.debug(f"[RemoteP2PRecovery] Relay {relay_id} ({ip}) is healthy")
                healthy.append(relay_id)
            else:
                logger.debug(f"[RemoteP2PRecovery] Relay {relay_id} ({ip}) is unreachable")

        return healthy

    def _get_healthy_relays_for_node(self, node_info: dict) -> list[str]:
        """Get ordered list of healthy relays for a NAT-blocked node.

        January 13, 2026: Added for relay failover. Checks relay_primary,
        relay_secondary, relay_tertiary, relay_quaternary in order and returns
        those that are healthy.

        Args:
            node_info: Node configuration dict

        Returns:
            List of healthy relay node IDs (in priority order)
        """
        # Collect all configured relays in priority order
        relay_keys = ["relay_primary", "relay_secondary", "relay_tertiary", "relay_quaternary"]
        relays = [node_info.get(key) for key in relay_keys if node_info.get(key)]

        if not relays:
            return []

        return self._check_relay_health(relays)

    def _get_node_capacity(self, node_id: str) -> dict[str, Any]:
        """Estimate node capacity for work queue pre-population.

        Session 17.28: Returns capacity info to help work dispatcher
        prioritize and allocate work appropriately to recovered nodes.

        Args:
            node_id: Node identifier

        Returns:
            Dict with capacity info: gpu_count, gpu_memory_gb, role, etc.
        """
        configured_nodes = self._get_configured_nodes()
        node_info = configured_nodes.get(node_id, {})

        # Extract GPU info from node configuration
        gpu_info = node_info.get("gpu", "")
        gpu_memory_gb = 0
        gpu_count = 1  # Default to 1 GPU

        # Parse GPU memory from strings like "GH200 96GB", "H100 80GB", etc.
        if gpu_info:
            import re
            memory_match = re.search(r"(\d+)\s*GB", gpu_info, re.IGNORECASE)
            if memory_match:
                gpu_memory_gb = int(memory_match.group(1))

            # Parse GPU count from strings like "8x RTX 4090"
            count_match = re.search(r"(\d+)\s*x\s*", gpu_info, re.IGNORECASE)
            if count_match:
                gpu_count = int(count_match.group(1))

        # Get role info
        role = node_info.get("role", "gpu_selfplay")
        can_train = role in ("gpu_training_primary", "training_backup", "gpu_both")
        can_selfplay = role not in ("coordinator", "voter", "cpu_only")

        return {
            "gpu_count": gpu_count,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_info": gpu_info,
            "role": role,
            "can_train": can_train,
            "can_selfplay": can_selfplay,
            "provider": node_info.get("provider", "unknown"),
        }

    def reset_stats(self) -> None:
        """Reset recovery statistics."""
        self._stats = RemoteP2PRecoveryStats()
        self._last_attempt.clear()
        self._restart_count.clear()
        self._first_stable_time.clear()
        logger.info("[RemoteP2PRecovery] Statistics and backoff state reset")

    def health_check(self) -> dict[str, Any]:
        """Return health status for daemon manager integration.

        January 2026: Added for HandlerBase compatibility and unified
        health monitoring across all P2P loops.

        Returns:
            Health check result with status, message, and details.
        """
        # Calculate health based on success/failure rates
        total_attempts = (
            self._stats.nodes_recovered
            + self._stats.nodes_failed
            + self._stats.nodes_skipped_unreachable
        )

        if total_attempts == 0:
            # No attempts yet - healthy by default
            return {
                "healthy": True,
                "status": "idle",
                "message": "No recovery attempts yet",
                "details": {
                    "enabled": self.config.enabled,
                    "is_leader": self._is_leader(),
                    "ssh_key_missing": self._stats.ssh_key_missing,
                    "cycles_run": self._stats.cycles_run,
                },
            }

        # Calculate success rate
        success_rate = self._stats.nodes_verified / max(self._stats.nodes_recovered, 1)
        failure_rate = self._stats.nodes_failed / max(total_attempts, 1)

        # Healthy if <50% failure rate and SSH key present
        healthy = failure_rate < 0.5 and not self._stats.ssh_key_missing

        # Determine status
        if self._stats.ssh_key_missing:
            status = "degraded"
            message = "SSH key missing or invalid"
        elif failure_rate >= 0.5:
            status = "degraded"
            message = f"High failure rate: {failure_rate:.0%}"
        elif not self._is_leader():
            status = "standby"
            message = "Not leader - recovery disabled"
        else:
            status = "healthy"
            message = f"Recovered {self._stats.nodes_recovered} nodes, {self._stats.nodes_verified} verified"

        return {
            "healthy": healthy,
            "status": status,
            "message": message,
            "details": {
                "enabled": self.config.enabled,
                "is_leader": self._is_leader(),
                "nodes_recovered": self._stats.nodes_recovered,
                "nodes_verified": self._stats.nodes_verified,
                "nodes_failed": self._stats.nodes_failed,
                "nodes_skipped_unreachable": self._stats.nodes_skipped_unreachable,
                "nodes_skipped_cooldown": self._stats.nodes_skipped_cooldown,
                "nodes_skipped_backoff": self._stats.nodes_skipped_backoff,
                "backoff_resets": self._stats.backoff_resets,
                "cycles_run": self._stats.cycles_run,
                "success_rate": success_rate,
                "failure_rate": failure_rate,
                "ssh_key_missing": self._stats.ssh_key_missing,
                "nodes_in_backoff": len(self._restart_count),
            },
        }

"""
Graceful Leader Step-Down Module.

Dec 30, 2025: Part of Phase 6 - Bully Algorithm Enhancements.

Provides graceful leadership transfer for planned maintenance or
voluntary step-down. Ensures minimal disruption to cluster operations.

Features:
- Work queue draining before step-down
- State transfer to successor
- Coordinated leadership handoff
- Event emission for cluster awareness

Usage:
    from scripts.p2p.graceful_stepdown import (
        GracefulStepDown,
        StepDownConfig,
        StepDownReason,
    )

    stepdown = GracefulStepDown(orchestrator)
    success = await stepdown.step_down_gracefully(
        reason=StepDownReason.MAINTENANCE,
        successor="node-2",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

logger = logging.getLogger(__name__)

# Step-down configuration - import centralized timeouts
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    DEFAULT_DRAIN_TIMEOUT = LoopTimeouts.DRAIN_TIMEOUT
    DEFAULT_STATE_TRANSFER_TIMEOUT = LoopTimeouts.STATE_TRANSFER
except ImportError:
    # Fallback for standalone usage
    DEFAULT_DRAIN_TIMEOUT = 30.0
    DEFAULT_STATE_TRANSFER_TIMEOUT = 10.0

DEFAULT_ANNOUNCEMENT_DELAY = 2.0  # seconds to wait after announcement


class StepDownReason(str, Enum):
    """Reasons for leader step-down."""

    MAINTENANCE = "maintenance"  # Planned maintenance
    SHUTDOWN = "shutdown"  # Node shutting down
    RESOURCE_EXHAUSTED = "resource_exhausted"  # Out of resources
    VOLUNTARY = "voluntary"  # Manual step-down
    QUORUM_LOST = "quorum_lost"  # Lost voter quorum
    SPLIT_BRAIN = "split_brain"  # Split-brain resolution
    HEALTH_DEGRADED = "health_degraded"  # Health check failing


class StepDownState(str, Enum):
    """States during step-down process."""

    NOT_STARTED = "not_started"
    ANNOUNCING = "announcing"  # Announcing intention to step down
    DRAINING = "draining"  # Waiting for work to complete
    TRANSFERRING = "transferring"  # Transferring state to successor
    RELEASING = "releasing"  # Releasing leader lease
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepDownConfig:
    """Configuration for graceful step-down."""

    drain_timeout: float = DEFAULT_DRAIN_TIMEOUT
    state_transfer_timeout: float = DEFAULT_STATE_TRANSFER_TIMEOUT
    announcement_delay: float = DEFAULT_ANNOUNCEMENT_DELAY
    force_on_timeout: bool = True  # Force step-down even if drain times out
    skip_state_transfer: bool = False  # Skip state transfer (for emergencies)


@dataclass
class StepDownResult:
    """Result of step-down operation."""

    success: bool
    reason: StepDownReason
    successor: str | None
    state: StepDownState
    drain_duration: float = 0.0
    transfer_duration: float = 0.0
    total_duration: float = 0.0
    jobs_drained: int = 0
    jobs_abandoned: int = 0
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "reason": self.reason.value,
            "successor": self.successor,
            "state": self.state.value,
            "drain_duration": round(self.drain_duration, 2),
            "transfer_duration": round(self.transfer_duration, 2),
            "total_duration": round(self.total_duration, 2),
            "jobs_drained": self.jobs_drained,
            "jobs_abandoned": self.jobs_abandoned,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class OrchestratorProtocol(Protocol):
    """Protocol defining required orchestrator methods."""

    node_id: str
    leader_id: str | None
    peers: dict[str, Any]
    peers_lock: Any

    def _save_state(self) -> None:
        """Save state to disk."""
        ...

    def _start_election(self) -> None:
        """Start a new election."""
        ...


class GracefulStepDown:
    """
    Manages graceful leader step-down for planned maintenance.

    Ensures minimal disruption by:
    1. Announcing step-down to cluster
    2. Stopping acceptance of new work
    3. Draining in-flight work
    4. Transferring state to successor
    5. Releasing leader lease
    6. Triggering new election
    """

    def __init__(
        self,
        orchestrator: Any,
        config: StepDownConfig | None = None,
    ) -> None:
        """Initialize graceful step-down manager.

        Args:
            orchestrator: P2P orchestrator instance
            config: Step-down configuration
        """
        self._orchestrator = orchestrator
        self._config = config or StepDownConfig()
        self._state = StepDownState.NOT_STARTED
        self._accepting_work = True
        self._step_down_in_progress = False

    @property
    def state(self) -> StepDownState:
        """Get current step-down state."""
        return self._state

    @property
    def is_stepping_down(self) -> bool:
        """Check if step-down is in progress."""
        return self._step_down_in_progress

    async def step_down_gracefully(
        self,
        reason: StepDownReason = StepDownReason.VOLUNTARY,
        successor: str | None = None,
    ) -> StepDownResult:
        """Gracefully transfer leadership before stepping down.

        Args:
            reason: Reason for step-down
            successor: Preferred successor node ID (optional)

        Returns:
            StepDownResult with details
        """
        start_time = time.time()

        # Check if we're actually the leader
        if getattr(self._orchestrator, "leader_id", None) != self._orchestrator.node_id:
            return StepDownResult(
                success=False,
                reason=reason,
                successor=successor,
                state=StepDownState.NOT_STARTED,
                error="not_leader",
            )

        if self._step_down_in_progress:
            return StepDownResult(
                success=False,
                reason=reason,
                successor=successor,
                state=self._state,
                error="already_in_progress",
            )

        self._step_down_in_progress = True
        jobs_drained = 0
        jobs_abandoned = 0

        try:
            # Step 1: Announce intention to step down
            self._state = StepDownState.ANNOUNCING
            await self._announce_step_down(reason, successor)
            await asyncio.sleep(self._config.announcement_delay)

            # Step 2: Stop accepting new work
            self._accepting_work = False
            self._set_orchestrator_accepting_work(False)

            # Step 3: Drain work queue
            self._state = StepDownState.DRAINING
            drain_start = time.time()
            jobs_drained, jobs_abandoned = await self._drain_work_queue()
            drain_duration = time.time() - drain_start

            # Step 4: Transfer state to successor
            transfer_duration = 0.0
            if not self._config.skip_state_transfer and successor:
                self._state = StepDownState.TRANSFERRING
                transfer_start = time.time()
                await self._transfer_state_to_successor(successor)
                transfer_duration = time.time() - transfer_start

            # Step 5: Release leader lease
            self._state = StepDownState.RELEASING
            await self._release_leader_lease()

            # Step 6: Emit step-down event
            await self._emit_step_down_event(reason, successor)

            # Step 7: Trigger new election
            self._trigger_election()

            self._state = StepDownState.COMPLETED
            total_duration = time.time() - start_time

            logger.info(
                f"Graceful step-down completed: reason={reason.value}, "
                f"successor={successor}, duration={total_duration:.1f}s, "
                f"drained={jobs_drained}, abandoned={jobs_abandoned}"
            )

            return StepDownResult(
                success=True,
                reason=reason,
                successor=successor,
                state=self._state,
                drain_duration=drain_duration,
                transfer_duration=transfer_duration,
                total_duration=total_duration,
                jobs_drained=jobs_drained,
                jobs_abandoned=jobs_abandoned,
            )

        except asyncio.TimeoutError as e:
            self._state = StepDownState.FAILED
            if self._config.force_on_timeout:
                # Force step-down anyway
                await self._release_leader_lease()
                self._trigger_election()
                return StepDownResult(
                    success=True,
                    reason=reason,
                    successor=successor,
                    state=self._state,
                    total_duration=time.time() - start_time,
                    jobs_abandoned=jobs_abandoned,
                    error="timeout_forced",
                )
            return StepDownResult(
                success=False,
                reason=reason,
                successor=successor,
                state=self._state,
                total_duration=time.time() - start_time,
                error=str(e),
            )

        except Exception as e:
            self._state = StepDownState.FAILED
            logger.error(f"Graceful step-down failed: {e}")
            return StepDownResult(
                success=False,
                reason=reason,
                successor=successor,
                state=self._state,
                total_duration=time.time() - start_time,
                error=str(e),
            )

        finally:
            self._step_down_in_progress = False
            self._accepting_work = True

    async def _announce_step_down(
        self,
        reason: StepDownReason,
        successor: str | None,
    ) -> None:
        """Announce step-down intention to cluster."""
        try:
            # Emit event for cluster awareness
            self._safe_emit_event("LEADER_STEP_DOWN_ANNOUNCED", {
                "leader_id": self._orchestrator.node_id,
                "reason": reason.value,
                "successor": successor,
                "timestamp": time.time(),
            })
        except Exception as e:
            logger.debug(f"Failed to announce step-down: {e}")

    def _set_orchestrator_accepting_work(self, accepting: bool) -> None:
        """Set whether orchestrator accepts new work."""
        if hasattr(self._orchestrator, "_accepting_work"):
            self._orchestrator._accepting_work = accepting
        if hasattr(self._orchestrator, "job_manager"):
            jm = self._orchestrator.job_manager
            if hasattr(jm, "set_accepting_jobs"):
                jm.set_accepting_jobs(accepting)

    async def _drain_work_queue(self) -> tuple[int, int]:
        """Drain the work queue, waiting for in-flight work.

        Returns:
            (jobs_drained, jobs_abandoned)
        """
        jobs_drained = 0
        jobs_abandoned = 0
        deadline = time.time() + self._config.drain_timeout

        try:
            # Get work queue or job manager
            job_manager = getattr(self._orchestrator, "job_manager", None)
            if not job_manager:
                return 0, 0

            # Poll for completion
            while time.time() < deadline:
                # Check active jobs
                active_jobs = getattr(job_manager, "active_jobs", {})
                if hasattr(active_jobs, "__len__"):
                    if len(active_jobs) == 0:
                        break

                # Check if work queue is empty
                if hasattr(job_manager, "get_pending_job_count"):
                    pending = job_manager.get_pending_job_count()
                    if pending == 0:
                        break

                await asyncio.sleep(1.0)
                jobs_drained += 1

            # Count abandoned jobs
            if hasattr(job_manager, "active_jobs"):
                jobs_abandoned = len(job_manager.active_jobs)

        except Exception as e:
            logger.debug(f"Error draining work queue: {e}")

        return jobs_drained, jobs_abandoned

    async def _transfer_state_to_successor(self, successor: str) -> None:
        """Transfer leader state to successor for fast takeover.

        Args:
            successor: Target node ID
        """
        try:
            # Gather state to transfer
            state = await self._gather_leader_state()
            if not state:
                return

            # Find successor address
            successor_addr = self._get_peer_address(successor)
            if not successor_addr:
                logger.warning(f"Cannot find address for successor {successor}")
                return

            # Send state transfer request
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.state_transfer_timeout)
            ) as session:
                url = f"http://{successor_addr}/leader-state-transfer"
                async with session.post(url, json=state) as resp:
                    if resp.status == 200:
                        logger.info(f"State transferred to successor {successor}")
                    else:
                        logger.warning(
                            f"State transfer to {successor} failed: HTTP {resp.status}"
                        )

        except Exception as e:
            logger.debug(f"State transfer failed: {e}")

    async def _gather_leader_state(self) -> dict[str, Any] | None:
        """Gather leader state for transfer."""
        try:
            state: dict[str, Any] = {
                "timestamp": time.time(),
                "from_leader": self._orchestrator.node_id,
            }

            # Include work queue state
            if hasattr(self._orchestrator, "job_manager"):
                jm = self._orchestrator.job_manager
                if hasattr(jm, "get_state_snapshot"):
                    state["job_manager"] = jm.get_state_snapshot()

            # Include selfplay scheduler state
            if hasattr(self._orchestrator, "selfplay_scheduler"):
                ss = self._orchestrator.selfplay_scheduler
                if hasattr(ss, "get_state_snapshot"):
                    state["selfplay_scheduler"] = ss.get_state_snapshot()

            return state

        except Exception as e:
            logger.debug(f"Error gathering leader state: {e}")
            return None

    def _get_peer_address(self, node_id: str) -> str | None:
        """Get peer's HTTP address."""
        with self._orchestrator.peers_lock:
            peer = self._orchestrator.peers.get(node_id)
            if peer:
                host = getattr(peer, "host", None) or getattr(peer, "address", None)
                port = getattr(peer, "port", 8770)
                if host:
                    return f"{host}:{port}"
        return None

    async def _release_leader_lease(self) -> None:
        """Release the leader lease early."""
        try:
            # Import NodeRole lazily
            try:
                from scripts.p2p.types import NodeRole
            except ImportError:
                from enum import Enum

                class NodeRole(str, Enum):
                    LEADER = "leader"
                    FOLLOWER = "follower"

            # Clear leadership state
            self._orchestrator.role = NodeRole.FOLLOWER
            self._orchestrator.leader_id = None
            self._orchestrator.leader_lease_id = ""
            self._orchestrator.leader_lease_expires = 0.0

            # Clear voter grant if we have one
            if hasattr(self._orchestrator, "voter_grant_leader_id"):
                if self._orchestrator.voter_grant_leader_id == self._orchestrator.node_id:
                    self._orchestrator.voter_grant_leader_id = ""
                    self._orchestrator.voter_grant_lease_id = ""
                    self._orchestrator.voter_grant_expires = 0.0

            # Save state
            if hasattr(self._orchestrator, "_save_state"):
                self._orchestrator._save_state()

            logger.debug("Leader lease released")

        except Exception as e:
            logger.error(f"Failed to release leader lease: {e}")
            raise

    async def _emit_step_down_event(
        self,
        reason: StepDownReason,
        successor: str | None,
    ) -> None:
        """Emit step-down completion event."""
        self._safe_emit_event("LEADER_STEP_DOWN_COMPLETED", {
            "leader_id": self._orchestrator.node_id,
            "reason": reason.value,
            "successor": successor,
            "timestamp": time.time(),
        })

    def _trigger_election(self) -> None:
        """Trigger new leader election."""
        # Jan 5, 2026: Check quorum before starting election
        # If quorum is LOST, election cannot succeed - skip to avoid election storms
        check_quorum = getattr(self._orchestrator, "_check_quorum_health", None)
        if check_quorum:
            try:
                from scripts.p2p.leader_election import QuorumHealthLevel
                quorum_health = check_quorum()
                if quorum_health == QuorumHealthLevel.LOST:
                    logger.warning(
                        "[GracefulStepDown] Quorum LOST - cannot hold election. "
                        "Wait for voters to recover."
                    )
                    self._safe_emit_event("ELECTION_BLOCKED_QUORUM_LOST", {
                        "reason": "graceful_stepdown",
                        "quorum_health": quorum_health.value,
                    })
                    return
            except ImportError:
                pass  # QuorumHealthLevel not available, proceed without check

        if hasattr(self._orchestrator, "_start_election"):
            try:
                self._orchestrator._start_election()
            except Exception as e:
                logger.debug(f"Failed to trigger election: {e}")

    def _safe_emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Safely emit an event."""
        try:
            if hasattr(self._orchestrator, "_safe_emit_event"):
                self._orchestrator._safe_emit_event(event_type, payload)
            else:
                from app.coordination.data_events import DataEventType, emit_event

                emit_event(DataEventType(event_type.lower()), payload)
        except Exception:
            pass  # Event emission should not block step-down


# Convenience functions
async def step_down_leader(
    orchestrator: Any,
    reason: StepDownReason = StepDownReason.VOLUNTARY,
    successor: str | None = None,
    config: StepDownConfig | None = None,
) -> StepDownResult:
    """Convenience function for graceful step-down.

    Args:
        orchestrator: P2P orchestrator instance
        reason: Reason for step-down
        successor: Preferred successor
        config: Step-down configuration

    Returns:
        StepDownResult
    """
    stepdown = GracefulStepDown(orchestrator, config)
    return await stepdown.step_down_gracefully(reason, successor)


def select_best_successor(orchestrator: Any) -> str | None:
    """Select the best successor for leadership.

    Criteria:
    1. Must be a voter
    2. Must be alive
    3. Highest uptime (stability)
    4. Lowest node_id (tiebreaker)

    Args:
        orchestrator: P2P orchestrator instance

    Returns:
        Best successor node ID or None
    """
    voter_ids = getattr(orchestrator, "voter_node_ids", []) or []
    if not voter_ids:
        return None

    candidates = []
    with orchestrator.peers_lock:
        for node_id in voter_ids:
            if node_id == orchestrator.node_id:
                continue  # Don't select self

            peer = orchestrator.peers.get(node_id)
            if peer and peer.is_alive():
                uptime = getattr(peer, "uptime", 0) or 0
                candidates.append((node_id, uptime))

    if not candidates:
        return None

    # Sort by uptime (descending), then node_id (ascending)
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[0][0]

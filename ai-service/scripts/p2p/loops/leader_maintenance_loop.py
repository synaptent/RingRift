"""Leader Maintenance Loop - Periodic Leadership Refresh.

Jan 25, 2026 - Fixes leadership stability by periodically refreshing
leadership on the designated coordinator node.

Problem: Gossip protocol can overwrite forced leadership state, causing the
designated coordinator to lose leadership even after force_leader was called.
This breaks work queue dispatch since is_leader returns False.

Solution: LeaderMaintenanceLoop runs on the designated coordinator and
periodically checks if:
1. This node is in the voter list (can be a leader)
2. This node is supposed to be the primary leader (first voter)
3. This node has lost leadership (gossip overwrote it)

If all conditions are true, it forces leadership back using the existing
_forced_leader_override mechanism.

Usage:
    from scripts.p2p.loops.leader_maintenance_loop import LeaderMaintenanceLoop

    loop = LeaderMaintenanceLoop(orchestrator)
    loop.start_background()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from .base import BaseLoop

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# How often to check and refresh leadership (seconds)
MAINTENANCE_INTERVAL = 30.0

# Grace period after startup before forcing leadership
STARTUP_GRACE_PERIOD = 60.0


class LeaderMaintenanceLoop(BaseLoop):
    """Periodically refresh leadership on the designated coordinator.

    This loop prevents gossip protocol from permanently overwriting forced
    leadership state. It only runs on nodes that are configured as the
    primary coordinator (first in the voter list).

    Key features:
    - Checks if this node is the designated primary leader
    - Refreshes leadership if gossip has overwritten it
    - Emits events for observability
    - Has startup grace period to avoid interfering with normal elections
    """

    depends_on: list[str] = []  # No dependencies

    def __init__(
        self,
        orchestrator: Any,
        *,
        maintenance_interval: float = MAINTENANCE_INTERVAL,
        startup_grace_period: float = STARTUP_GRACE_PERIOD,
    ) -> None:
        """Initialize the leader maintenance loop.

        Args:
            orchestrator: P2POrchestrator instance
            maintenance_interval: Seconds between maintenance checks (default: 30s)
            startup_grace_period: Seconds to wait after startup before
                forcing leadership (default: 60s)
        """
        super().__init__(
            name="leader_maintenance",
            interval=maintenance_interval,
            enabled=True,
            depends_on=[],
        )

        self._orchestrator = orchestrator
        self._startup_grace_period = startup_grace_period
        self._startup_time = time.time()
        self._leadership_refreshes = 0
        self._last_refresh_time: float = 0.0
        self._is_primary_voter: bool | None = None  # Cached flag

    def _is_in_startup_phase(self) -> bool:
        """Check if we're still in the startup grace period."""
        elapsed = time.time() - self._startup_time
        return elapsed < self._startup_grace_period

    def _get_node_id(self) -> str | None:
        """Get this node's ID."""
        try:
            return getattr(self._orchestrator, "node_id", None)
        except (AttributeError, TypeError):
            return None

    def _get_voter_list(self) -> list[str]:
        """Get the voter node IDs from orchestrator."""
        try:
            voters = getattr(self._orchestrator, "voter_node_ids", None) or []
            return list(voters)
        except (AttributeError, TypeError):
            return []

    def _is_primary_leader_node(self) -> bool:
        """Check if this node is the designated primary leader.

        The primary leader is the first node in the voter list.
        This is typically the main coordinator (local-mac).

        Returns:
            True if this node is first in voter list
        """
        if self._is_primary_voter is not None:
            return self._is_primary_voter

        node_id = self._get_node_id()
        if not node_id:
            return False

        voters = self._get_voter_list()
        if not voters:
            return False

        # First voter is the designated primary leader
        self._is_primary_voter = (voters[0] == node_id)
        if self._is_primary_voter:
            logger.info(
                f"[LeaderMaintenance] This node ({node_id}) is the primary "
                f"leader (first in voter list: {voters[:3]}...)"
            )
        return self._is_primary_voter

    def _is_currently_leader(self) -> bool:
        """Check if this node is currently the leader."""
        try:
            # Use the orchestrator's _is_leader method which now checks
            # _forced_leader_override first
            is_leader = getattr(self._orchestrator, "_is_leader", None)
            if callable(is_leader):
                return bool(is_leader())

            # Fallback to checking the role directly
            from scripts.p2p.types import NodeRole
            role = getattr(self._orchestrator, "role", None)
            return role == NodeRole.LEADER
        except (AttributeError, TypeError, ImportError):
            return False

    def _get_leader_id(self) -> str | None:
        """Get the current leader ID."""
        try:
            return getattr(self._orchestrator, "leader_id", None)
        except (AttributeError, TypeError):
            return None

    async def _force_leadership(self) -> bool:
        """Force this node to become leader.

        Uses the same mechanism as the force_leader HTTP endpoint.

        Returns:
            True if leadership was successfully forced
        """
        try:
            import uuid
            from app.p2p.constants import LEADER_LEASE_DURATION
            from scripts.p2p.types import NodeRole

            node_id = self._get_node_id()
            if not node_id:
                logger.warning("[LeaderMaintenance] Cannot force leadership: no node_id")
                return False

            # Generate new lease
            lease_id = f"{node_id}_{int(time.time())}_maint_{uuid.uuid4().hex[:8]}"

            # Set leader state
            self._orchestrator.role = NodeRole.LEADER
            self._orchestrator.leader_id = node_id
            self._orchestrator.leader_lease_id = lease_id
            self._orchestrator.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
            self._orchestrator.last_leader_seen = time.time()
            self._orchestrator.election_in_progress = False

            # Set forced leader override (critical for is_leader checks)
            self._orchestrator._forced_leader_override = True

            # Increment epoch and save state
            if hasattr(self._orchestrator, "_increment_cluster_epoch"):
                self._orchestrator._increment_cluster_epoch()
            if hasattr(self._orchestrator, "_save_state"):
                self._orchestrator._save_state()

            # Broadcast to peers for fast propagation
            if hasattr(self._orchestrator, "_broadcast_leader_to_all_peers"):
                epoch = getattr(self._orchestrator, "cluster_epoch", 0)
                if hasattr(self._orchestrator, "_leadership_sm"):
                    sm = self._orchestrator._leadership_sm
                    if sm:
                        epoch = getattr(sm, "epoch", epoch)
                asyncio.create_task(
                    self._orchestrator._broadcast_leader_to_all_peers(
                        node_id,
                        epoch,
                        self._orchestrator.leader_lease_expires,
                    )
                )

            self._leadership_refreshes += 1
            self._last_refresh_time = time.time()

            logger.warning(
                f"[LeaderMaintenance] Refreshed leadership for {node_id} "
                f"(refresh #{self._leadership_refreshes})"
            )

            self._emit_event("LEADER_MAINTENANCE_REFRESH", {
                "node_id": node_id,
                "lease_id": lease_id,
                "refresh_count": self._leadership_refreshes,
                "reason": "gossip_overwrite_recovery",
            })

            return True

        except Exception as e:
            logger.error(f"[LeaderMaintenance] Failed to force leadership: {e}")
            return False

    async def _run_once(self) -> None:
        """Single maintenance iteration - check and refresh leadership if needed."""
        # Skip during startup grace period
        if self._is_in_startup_phase():
            remaining = self._startup_grace_period - (time.time() - self._startup_time)
            logger.debug(
                f"[LeaderMaintenance] Startup grace period active "
                f"({remaining:.0f}s remaining)"
            )
            return

        # Only run on the designated primary leader node
        if not self._is_primary_leader_node():
            logger.debug("[LeaderMaintenance] Not the primary leader node, skipping")
            return

        # Check if we're currently the leader
        if self._is_currently_leader():
            # Renew our lease to prevent it from expiring
            # Without this, the lease expires after LEADER_LEASE_DURATION
            # and gossip can then override our leadership
            try:
                from app.p2p.constants import LEADER_LEASE_DURATION
                lease_expires = getattr(self._orchestrator, "leader_lease_expires", 0)
                remaining = lease_expires - time.time()
                # Renew when less than 50% of lease remaining
                if remaining < LEADER_LEASE_DURATION * 0.5:
                    self._orchestrator.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                    self._orchestrator.last_leader_seen = time.time()
                    if hasattr(self._orchestrator, "_save_state"):
                        self._orchestrator._save_state()
                    logger.debug(
                        f"[LeaderMaintenance] Renewed lease "
                        f"(was {remaining:.0f}s remaining, now {LEADER_LEASE_DURATION}s)"
                    )
            except (ImportError, AttributeError) as e:
                logger.debug(f"[LeaderMaintenance] Lease renewal skipped: {e}")
            return

        # We should be leader but we're not - gossip has overwritten us
        current_leader = self._get_leader_id()
        node_id = self._get_node_id()

        logger.warning(
            f"[LeaderMaintenance] Leadership lost! Current leader: {current_leader}, "
            f"this node: {node_id}. Refreshing leadership..."
        )

        self._emit_event("LEADER_MAINTENANCE_NEEDED", {
            "node_id": node_id,
            "current_leader": current_leader,
            "reason": "gossip_overwrite",
        })

        # Force leadership back
        success = await self._force_leadership()
        if not success:
            logger.error("[LeaderMaintenance] Failed to refresh leadership")

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event for observability."""
        try:
            from app.coordination.event_router import emit_event

            emit_event(event_type, {
                "source": "leader_maintenance_loop",
                "timestamp": time.time(),
                **payload,
            })
        except Exception as e:
            logger.debug(f"[LeaderMaintenance] Failed to emit event {event_type}: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get loop status for health checks."""
        return {
            "name": self.name,
            "running": self.running,
            "enabled": self.enabled,
            "is_primary_voter": self._is_primary_voter,
            "is_leader": self._is_currently_leader(),
            "leadership_refreshes": self._leadership_refreshes,
            "last_refresh_time": self._last_refresh_time,
            "in_startup_grace": self._is_in_startup_phase(),
        }

    def health_check(self) -> Any:
        """Check loop health for DaemonManager integration."""
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"LeaderMaintenanceLoop {'running' if self.running else 'stopped'}",
                "details": self.get_status(),
            }

        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="LeaderMaintenanceLoop is stopped",
                details={"running": False},
            )

        # Not the primary voter - healthy but inactive
        if not self._is_primary_leader_node():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="Not primary leader node (inactive)",
                details={"is_primary_voter": False},
            )

        # Primary voter - check if we're maintaining leadership
        if self._is_currently_leader():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Leadership maintained ({self._leadership_refreshes} refreshes)",
                details={
                    "is_leader": True,
                    "refreshes": self._leadership_refreshes,
                },
            )

        # We should be leader but aren't - degraded
        return HealthCheckResult(
            healthy=True,  # Still running, just needs refresh
            status=CoordinatorStatus.DEGRADED,
            message="Leadership needs refresh",
            details={
                "is_leader": False,
                "current_leader": self._get_leader_id(),
                "refreshes": self._leadership_refreshes,
            },
        )

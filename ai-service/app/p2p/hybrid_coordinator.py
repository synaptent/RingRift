"""Hybrid Coordinator for P2P Protocol Integration.

This module provides the glue layer that coordinates between:
- SWIM-based membership (gossip protocol for failure detection)
- Raft-based consensus (for work queue and leader election)
- Existing HTTP-based P2P orchestrator

The HybridCoordinator enables gradual migration from the current Bully/HTTP
system to SWIM/Raft, with instant fallback capability via feature flags.

Feature Flags (from app.p2p.constants):
- MEMBERSHIP_MODE: "http" | "swim" | "hybrid"
- CONSENSUS_MODE: "bully" | "raft" | "hybrid"

Usage:
    from app.p2p.hybrid_coordinator import HybridCoordinator

    coordinator = HybridCoordinator(p2p_orchestrator)
    await coordinator.start()

    # Routes to SWIM or HTTP based on MEMBERSHIP_MODE
    alive_peers = coordinator.get_alive_peers()

    # Routes to Raft or SQLite based on CONSENSUS_MODE
    success = coordinator.claim_work(work_id, node_id)

    # Check leader status (Raft or Bully)
    if coordinator.is_leader():
        # Do leader things

    await coordinator.stop()

Phase 2.4 - Dec 26, 2025
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    # Avoid circular imports at runtime
    pass

logger = logging.getLogger(__name__)

# ============================================
# Import feature flags from P2P constants
# ============================================

from app.p2p.constants import (
    CONSENSUS_MODE,
    MEMBERSHIP_MODE,
    RAFT_ENABLED,
    SWIM_ENABLED,
)

# ============================================
# Import SWIM adapter
# ============================================

try:
    from app.p2p.swim_adapter import (
        SWIM_AVAILABLE,
        HybridMembershipManager,
        SwimMembershipManager,
    )
except ImportError:
    SWIM_AVAILABLE = False
    SwimMembershipManager = None  # type: ignore
    HybridMembershipManager = None  # type: ignore

# ============================================
# Import Raft state machines
# ============================================

try:
    from app.p2p.raft_state import (
        PYSYNCOBJ_AVAILABLE,
        ReplicatedWorkQueue,
        create_replicated_work_queue,
    )
except ImportError:
    PYSYNCOBJ_AVAILABLE = False
    ReplicatedWorkQueue = None  # type: ignore
    create_replicated_work_queue = None  # type: ignore

# ============================================
# Import SQLite work queue for fallback
# ============================================

try:
    from app.coordination.work_queue import WorkQueue, get_work_queue
except ImportError:
    WorkQueue = None  # type: ignore
    get_work_queue = None  # type: ignore


# ============================================
# Status Dataclass
# ============================================


@dataclass
class HybridStatus:
    """Comprehensive status of hybrid coordinator protocols."""

    # Overall state
    started: bool = False
    node_id: str = ""
    membership_mode: str = "http"
    consensus_mode: str = "bully"

    # SWIM status
    swim_enabled: bool = False
    swim_available: bool = False
    swim_started: bool = False
    swim_alive_count: int = 0
    swim_suspected_count: int = 0
    swim_failed_count: int = 0

    # Raft status
    raft_enabled: bool = False
    raft_available: bool = False
    raft_ready: bool = False
    raft_is_leader: bool = False
    raft_leader_address: str = ""
    raft_pending_work: int = 0
    raft_claimed_work: int = 0

    # HTTP fallback status
    http_alive_peers: int = 0
    http_leader_id: str = ""

    # Work queue stats
    work_queue_pending: int = 0
    work_queue_running: int = 0
    work_queue_completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "started": self.started,
            "node_id": self.node_id,
            "membership_mode": self.membership_mode,
            "consensus_mode": self.consensus_mode,
            "swim": {
                "enabled": self.swim_enabled,
                "available": self.swim_available,
                "started": self.swim_started,
                "alive_count": self.swim_alive_count,
                "suspected_count": self.swim_suspected_count,
                "failed_count": self.swim_failed_count,
            },
            "raft": {
                "enabled": self.raft_enabled,
                "available": self.raft_available,
                "ready": self.raft_ready,
                "is_leader": self.raft_is_leader,
                "leader_address": self.raft_leader_address,
                "pending_work": self.raft_pending_work,
                "claimed_work": self.raft_claimed_work,
            },
            "http": {
                "alive_peers": self.http_alive_peers,
                "leader_id": self.http_leader_id,
            },
            "work_queue": {
                "pending": self.work_queue_pending,
                "running": self.work_queue_running,
                "completed": self.work_queue_completed,
            },
        }


# ============================================
# HybridCoordinator
# ============================================


class HybridCoordinator:
    """Coordinates SWIM membership, Raft consensus, and HTTP P2P.

    This class provides a unified interface for:
    - Peer discovery and failure detection (SWIM or HTTP heartbeats)
    - Leader election (Raft or Bully algorithm)
    - Work queue management (Raft replicated or SQLite)

    The active protocols are determined by feature flags:
    - MEMBERSHIP_MODE: Controls peer discovery source
    - CONSENSUS_MODE: Controls leader election and work distribution

    All operations include automatic fallback when primary protocol
    is unavailable, ensuring cluster resilience during migration.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        node_id: str = "",
        on_leader_change: Callable[[str | None], None] | None = None,
        on_member_alive: Callable[[str], None] | None = None,
        on_member_failed: Callable[[str], None] | None = None,
    ):
        """Initialize the hybrid coordinator.

        Args:
            orchestrator: P2POrchestrator instance for HTTP fallback
            node_id: This node's unique identifier (auto-detected if not provided)
            on_leader_change: Callback when leader changes (receives leader ID or None)
            on_member_alive: Callback when a member becomes alive
            on_member_failed: Callback when a member is detected as failed
        """
        self._orchestrator = orchestrator
        self._node_id = node_id or self._detect_node_id()
        self._on_leader_change = on_leader_change
        self._on_member_alive = on_member_alive
        self._on_member_failed = on_member_failed

        # Protocol instances (created on start)
        self._swim_manager: SwimMembershipManager | HybridMembershipManager | None = None
        self._raft_queue: ReplicatedWorkQueue | None = None
        self._sqlite_queue: WorkQueue | None = None

        # State tracking
        self._started = False
        self._membership_mode = MEMBERSHIP_MODE
        self._consensus_mode = CONSENSUS_MODE

        # Fallback tracking
        self._swim_fallback_active = False
        self._raft_fallback_active = False
        self._last_fallback_log = 0.0

        logger.info(
            f"HybridCoordinator initialized: node_id={self._node_id}, "
            f"membership_mode={self._membership_mode}, "
            f"consensus_mode={self._consensus_mode}"
        )

    def _detect_node_id(self) -> str:
        """Auto-detect node ID from orchestrator or environment."""
        # Try orchestrator
        if self._orchestrator and hasattr(self._orchestrator, "node_id"):
            return self._orchestrator.node_id

        # Try environment
        import os
        env_node_id = os.environ.get("RINGRIFT_NODE_ID", "")
        if env_node_id:
            return env_node_id

        # Fall back to hostname
        import socket
        try:
            return socket.gethostname()
        except (OSError, socket.error) as e:
            # Dec 2025: Socket errors rare but possible in containers
            logger.debug(f"[HybridCoordinator] Could not get hostname: {e}")
            return "unknown"

    # ========================================
    # Lifecycle Methods
    # ========================================

    async def start(self) -> bool:
        """Start SWIM and Raft if enabled by feature flags.

        Returns:
            True if at least one protocol started successfully
        """
        if self._started:
            logger.warning("HybridCoordinator already started")
            return True

        success = False

        # Start SWIM if enabled
        if self._membership_mode in ("swim", "hybrid"):
            swim_started = await self._start_swim()
            if swim_started:
                success = True
                logger.info("SWIM membership started")
            else:
                logger.warning("SWIM membership failed to start, using HTTP fallback")
                self._swim_fallback_active = True

        # Start Raft if enabled
        if self._consensus_mode in ("raft", "hybrid"):
            raft_started = await self._start_raft()
            if raft_started:
                success = True
                logger.info("Raft consensus started")
            else:
                logger.warning("Raft consensus failed to start, using Bully/SQLite fallback")
                self._raft_fallback_active = True

        # Initialize SQLite fallback (always available)
        self._init_sqlite_fallback()

        self._started = True
        logger.info(
            f"HybridCoordinator started: swim={'active' if not self._swim_fallback_active else 'fallback'}, "
            f"raft={'active' if not self._raft_fallback_active else 'fallback'}"
        )

        return success or True  # SQLite fallback is always available

    async def _start_swim(self) -> bool:
        """Start SWIM membership protocol."""
        if not SWIM_AVAILABLE:
            logger.warning("swim-p2p not installed, cannot start SWIM")
            return False

        if not SWIM_ENABLED:
            logger.info("SWIM disabled by RINGRIFT_SWIM_ENABLED=false")
            return False

        try:
            if self._membership_mode == "hybrid":
                # Use hybrid manager that runs both SWIM and HTTP
                self._swim_manager = HybridMembershipManager(
                    node_id=self._node_id,
                )
            else:
                # Pure SWIM mode
                self._swim_manager = SwimMembershipManager.from_distributed_hosts(
                    node_id=self._node_id,
                )

            # Set up callbacks
            if hasattr(self._swim_manager, "swim_manager") and self._swim_manager.swim_manager:
                # Hybrid mode - callbacks on inner manager
                inner = self._swim_manager.swim_manager
                inner.on_member_alive = self._handle_member_alive
                inner.on_member_failed = self._handle_member_failed
            elif isinstance(self._swim_manager, SwimMembershipManager):
                self._swim_manager.on_member_alive = self._handle_member_alive
                self._swim_manager.on_member_failed = self._handle_member_failed

            return await self._swim_manager.start()

        except Exception as e:
            logger.error(f"Failed to start SWIM: {e}", exc_info=True)
            return False

    async def _start_raft(self) -> bool:
        """Start Raft consensus protocol.

        Jan 24, 2026: Wrapped create_replicated_work_queue in asyncio.to_thread()
        to prevent event loop blocking during Raft initialization. PySyncObj
        initialization can take several seconds on slow networks.
        """
        if not PYSYNCOBJ_AVAILABLE:
            logger.warning("pysyncobj not installed, cannot start Raft")
            return False

        if not RAFT_ENABLED:
            logger.info("Raft disabled by RINGRIFT_RAFT_ENABLED=false")
            return False

        try:
            # Run Raft initialization in thread pool to avoid blocking event loop
            # create_replicated_work_queue() calls PySyncObj.__init__() which
            # performs network operations and can block for seconds
            logger.info("Starting Raft work queue in background thread...")

            # Use functools.partial to pass keyword arguments to asyncio.to_thread
            import functools
            create_raft = functools.partial(
                create_replicated_work_queue,
                node_id=self._node_id,
                on_ready=self._handle_raft_ready,
                on_leader_change=self._handle_raft_leader_change,
            )
            self._raft_queue = await asyncio.to_thread(create_raft)

            if self._raft_queue is None:
                logger.warning("Failed to create Raft work queue")
                return False

            # Wait briefly for cluster formation
            await asyncio.sleep(1.0)

            logger.info("Raft work queue started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start Raft: {e}", exc_info=True)
            return False

    def _init_sqlite_fallback(self) -> None:
        """Initialize SQLite work queue fallback."""
        try:
            if get_work_queue is not None:
                self._sqlite_queue = get_work_queue()
                logger.debug("SQLite work queue fallback initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite fallback: {e}")

    async def stop(self) -> None:
        """Graceful shutdown of all protocols."""
        if not self._started:
            return

        logger.info("Stopping HybridCoordinator...")

        # Stop SWIM
        if self._swim_manager:
            try:
                await self._swim_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping SWIM: {e}")
            self._swim_manager = None

        # Stop Raft
        if self._raft_queue:
            try:
                self._raft_queue.destroy()
            except Exception as e:
                logger.warning(f"Error stopping Raft: {e}")
            self._raft_queue = None

        self._started = False
        logger.info("HybridCoordinator stopped")

    # ========================================
    # Callback Handlers
    # ========================================

    def _handle_member_alive(self, member_id: str) -> None:
        """Handle SWIM member becoming alive."""
        logger.info(f"Member alive: {member_id}")
        if self._on_member_alive:
            try:
                self._on_member_alive(member_id)
            except Exception as e:
                logger.error(f"Error in on_member_alive callback: {e}")

    def _handle_member_failed(self, member_id: str) -> None:
        """Handle SWIM member failure detection."""
        logger.warning(f"Member failed: {member_id}")
        if self._on_member_failed:
            try:
                self._on_member_failed(member_id)
            except Exception as e:
                logger.error(f"Error in on_member_failed callback: {e}")

    def _handle_raft_ready(self) -> None:
        """Handle Raft cluster ready event."""
        logger.info("Raft cluster ready")

    def _handle_raft_leader_change(self, leader_address: str | None) -> None:
        """Handle Raft leader change event."""
        leader_id = leader_address or ""
        logger.info(f"Raft leader changed: {leader_id}")
        if self._on_leader_change:
            try:
                self._on_leader_change(leader_id)
            except Exception as e:
                logger.error(f"Error in on_leader_change callback: {e}")

    # ========================================
    # Membership Methods (SWIM or HTTP)
    # ========================================

    def get_alive_peers(self) -> list[str]:
        """Get list of alive peer node IDs.

        Routes to SWIM or HTTP based on MEMBERSHIP_MODE.
        Falls back automatically if primary protocol unavailable.

        Returns:
            List of alive peer node IDs
        """
        # Try SWIM first if configured
        if self._membership_mode in ("swim", "hybrid") and not self._swim_fallback_active:
            if self._swim_manager:
                try:
                    peers = self._swim_manager.get_alive_peers()
                    if peers or self._membership_mode == "swim":
                        return peers
                except Exception as e:
                    self._log_fallback("SWIM", "get_alive_peers", str(e))

        # Fall back to HTTP (via orchestrator)
        return self._get_http_alive_peers()

    def _get_http_alive_peers(self) -> list[str]:
        """Get alive peers from HTTP P2P orchestrator."""
        if not self._orchestrator:
            return []

        try:
            # Try P2P orchestrator's peer tracking
            if hasattr(self._orchestrator, "get_alive_peers"):
                return self._orchestrator.get_alive_peers()

            # Alternative: get from peers dict
            if hasattr(self._orchestrator, "peers"):
                alive = []
                for peer_id, peer_info in self._orchestrator.peers.items():
                    if isinstance(peer_info, dict):
                        # Check if peer is considered alive
                        last_seen = peer_info.get("last_seen", 0)
                        if time.time() - last_seen < 120:  # 2 minute timeout
                            alive.append(peer_id)
                    else:
                        alive.append(peer_id)
                return alive

        except Exception as e:
            logger.debug(f"Failed to get HTTP alive peers: {e}")

        return []

    def is_peer_alive(self, peer_id: str) -> bool:
        """Check if a specific peer is alive.

        Args:
            peer_id: Node ID to check

        Returns:
            True if peer is alive
        """
        if self._membership_mode in ("swim", "hybrid") and not self._swim_fallback_active:
            if self._swim_manager:
                try:
                    return self._swim_manager.is_peer_alive(peer_id)
                except Exception as e:
                    self._log_fallback("SWIM", "is_peer_alive", str(e))

        # Fall back to checking alive peers list
        return peer_id in self.get_alive_peers()

    # ========================================
    # Consensus Methods (Raft or Bully)
    # ========================================

    def claim_work(self, work_id: str, node_id: str | None = None) -> bool:
        """Atomically claim a work item.

        Routes to Raft or SQLite based on CONSENSUS_MODE.

        Args:
            work_id: ID of work item to claim
            node_id: Node claiming the work (defaults to this node)

        Returns:
            True if claimed successfully
        """
        claimer = node_id or self._node_id

        # Try Raft first if configured
        if self._consensus_mode in ("raft", "hybrid") and not self._raft_fallback_active:
            if self._raft_queue and self._raft_queue.is_ready:
                try:
                    return self._raft_queue.claim_work(work_id, claimer)
                except Exception as e:
                    self._log_fallback("Raft", "claim_work", str(e))

        # Fall back to SQLite work queue
        return self._claim_work_sqlite(work_id, claimer)

    def _claim_work_sqlite(self, work_id: str, node_id: str) -> bool:
        """Claim work via SQLite fallback."""
        if not self._sqlite_queue:
            logger.warning("SQLite work queue not available")
            return False

        try:
            # SQLite claim_work takes node_id and capabilities
            item = self._sqlite_queue.claim_work(node_id=node_id)
            if item and item.work_id == work_id:
                return True
            return False
        except Exception as e:
            logger.error(f"SQLite claim_work failed: {e}")
            return False

    def is_leader(self) -> bool:
        """Check if this node is the current leader.

        Routes to Raft or Bully based on CONSENSUS_MODE.

        Returns:
            True if this node is the leader
        """
        # Try Raft first if configured
        if self._consensus_mode in ("raft", "hybrid") and not self._raft_fallback_active:
            if self._raft_queue and self._raft_queue.is_ready:
                try:
                    return self._raft_queue.is_leader
                except Exception as e:
                    self._log_fallback("Raft", "is_leader", str(e))

        # Fall back to Bully (via orchestrator)
        return self._is_bully_leader()

    def _is_bully_leader(self) -> bool:
        """Check leader status via Bully algorithm."""
        if not self._orchestrator:
            return False

        try:
            if hasattr(self._orchestrator, "is_leader"):
                return self._orchestrator.is_leader()
            if hasattr(self._orchestrator, "leader_id"):
                return self._orchestrator.leader_id == self._node_id
        except Exception as e:
            logger.debug(f"Failed to check Bully leader: {e}")

        return False

    def get_leader_id(self) -> str:
        """Get the current leader's node ID.

        Returns:
            Leader node ID, or empty string if unknown
        """
        # Try Raft first
        if self._consensus_mode in ("raft", "hybrid") and not self._raft_fallback_active:
            if self._raft_queue and self._raft_queue.is_ready:
                try:
                    addr = self._raft_queue.leader_address
                    if addr:
                        return addr
                except Exception as e:
                    self._log_fallback("Raft", "get_leader_id", str(e))

        # Fall back to Bully
        if self._orchestrator:
            try:
                if hasattr(self._orchestrator, "leader_id"):
                    return self._orchestrator.leader_id or ""
                if hasattr(self._orchestrator, "get_leader_id"):
                    return self._orchestrator.get_leader_id() or ""
            except (AttributeError, TypeError, RuntimeError) as e:
                # Dec 2025: Orchestrator may be in inconsistent state
                logger.debug(f"[HybridCoordinator] Bully leader lookup failed: {e}")

        return ""

    # ========================================
    # Status Methods
    # ========================================

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status combining all protocols.

        Returns:
            Dict with swim, raft, http, and work_queue sections
        """
        status = HybridStatus(
            started=self._started,
            node_id=self._node_id,
            membership_mode=self._membership_mode,
            consensus_mode=self._consensus_mode,
        )

        # SWIM status
        status.swim_enabled = SWIM_ENABLED
        status.swim_available = SWIM_AVAILABLE
        if self._swim_manager:
            try:
                summary = self._swim_manager.get_membership_summary()
                status.swim_started = summary.get("started", False)
                status.swim_alive_count = summary.get("alive", 0)
                status.swim_suspected_count = summary.get("suspected", 0)
                status.swim_failed_count = summary.get("failed", 0)
            except Exception as e:
                logger.debug(f"Failed to get SWIM status: {e}")

        # Raft status
        status.raft_enabled = RAFT_ENABLED
        status.raft_available = PYSYNCOBJ_AVAILABLE
        if self._raft_queue:
            try:
                status.raft_ready = self._raft_queue.is_ready
                status.raft_is_leader = self._raft_queue.is_leader
                status.raft_leader_address = self._raft_queue.leader_address or ""
                queue_stats = self._raft_queue.get_queue_stats()
                status.raft_pending_work = queue_stats.get("pending", 0)
                status.raft_claimed_work = queue_stats.get("claimed", 0) + queue_stats.get("running", 0)
            except Exception as e:
                logger.debug(f"Failed to get Raft status: {e}")

        # HTTP status
        try:
            http_peers = self._get_http_alive_peers()
            status.http_alive_peers = len(http_peers)
            status.http_leader_id = self.get_leader_id() if not self._raft_queue else ""
        except Exception as e:
            logger.debug(f"Failed to get HTTP status: {e}")

        # Work queue stats (from SQLite or Raft)
        if self._sqlite_queue:
            try:
                status.work_queue_pending = self._sqlite_queue.get_pending_count()
                status.work_queue_running = self._sqlite_queue.get_running_count()
                queue_status = self._sqlite_queue.get_queue_status()
                status.work_queue_completed = queue_status.get("stats", {}).get("total_completed", 0)
            except Exception as e:
                logger.debug(f"Failed to get SQLite queue status: {e}")

        return status.to_dict()

    # ========================================
    # Helper Methods
    # ========================================

    def _log_fallback(self, protocol: str, operation: str, error: str) -> None:
        """Log fallback activation with rate limiting."""
        now = time.time()
        if now - self._last_fallback_log > 60:  # Log at most once per minute
            logger.warning(f"{protocol} {operation} failed, using fallback: {error}")
            self._last_fallback_log = now

    @property
    def node_id(self) -> str:
        """Get this node's ID."""
        return self._node_id

    @property
    def started(self) -> bool:
        """Check if coordinator is started."""
        return self._started

    @property
    def membership_mode(self) -> str:
        """Get current membership mode."""
        return self._membership_mode

    @property
    def consensus_mode(self) -> str:
        """Get current consensus mode."""
        return self._consensus_mode


# ============================================
# Factory Functions
# ============================================


def create_hybrid_coordinator(
    orchestrator: Any = None,
    node_id: str = "",
    **kwargs,
) -> HybridCoordinator:
    """Create a HybridCoordinator with sensible defaults.

    Args:
        orchestrator: P2POrchestrator instance
        node_id: Node ID (auto-detected if not provided)
        **kwargs: Additional arguments passed to HybridCoordinator

    Returns:
        Configured HybridCoordinator instance
    """
    return HybridCoordinator(
        orchestrator=orchestrator,
        node_id=node_id,
        **kwargs,
    )


# ============================================
# Module exports
# ============================================

__all__ = [
    # Feature flags (re-exported for convenience)
    "CONSENSUS_MODE",
    "MEMBERSHIP_MODE",
    "PYSYNCOBJ_AVAILABLE",
    "RAFT_ENABLED",
    "SWIM_AVAILABLE",
    "SWIM_ENABLED",
    # Classes
    "HybridCoordinator",
    "HybridStatus",
    # Factory functions
    "create_hybrid_coordinator",
]

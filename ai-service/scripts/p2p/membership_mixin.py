"""SWIM Membership Mixin for P2POrchestrator.

Provides hybrid membership management using SWIM gossip protocol for fast failure
detection (<5s) while maintaining HTTP heartbeat compatibility.

Usage:
    class P2POrchestrator(MembershipMixin, ...):
        pass

Features:
- SWIM-based failure detection (5s vs 60+ seconds with HTTP)
- Hybrid mode: SWIM when available, HTTP fallback
- Event emission for membership changes
- Graceful handling of missing swim-p2p dependency

Refactored to use P2PMixinBase - Dec 27, 2025
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from threading import RLock

    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Import SWIM adapter with graceful fallback using protocol_utils
from scripts.p2p.protocol_utils import safe_import

_, SwimConfig, SwimMembershipManager, SWIM_ADAPTER_AVAILABLE = safe_import(
    "app.p2p.swim_adapter", "SwimConfig", "SwimMembershipManager"
)
if not SWIM_ADAPTER_AVAILABLE:
    logger.debug("swim_adapter not available - SWIM features disabled")

# December 29, 2025: Import from canonical constants for consistency
# Canonical constants handle auto-detection of swim-p2p availability and hybrid mode
try:
    from app.p2p.constants import (
        SWIM_ENABLED,
        SWIM_BIND_PORT,
        SWIM_FAILURE_TIMEOUT,
        SWIM_SUSPICION_TIMEOUT,
        SWIM_PING_INTERVAL,
        SWIM_INDIRECT_PING_COUNT,
        MEMBERSHIP_MODE,
        PEER_TIMEOUT,
    )
    logger.debug(f"Loaded SWIM constants from canonical source: SWIM_ENABLED={SWIM_ENABLED}, MEMBERSHIP_MODE={MEMBERSHIP_MODE}")
except ImportError:
    # Fallback to local defaults if canonical constants not available
    logger.debug("Canonical constants not available, using local defaults")
    _CONSTANTS = P2PMixinBase._load_config_constants({
        "SWIM_ENABLED": False,
        "SWIM_BIND_PORT": 7947,
        "SWIM_FAILURE_TIMEOUT": 10.0,
        "SWIM_SUSPICION_TIMEOUT": 6.0,
        "SWIM_PING_INTERVAL": 1.0,
        "SWIM_INDIRECT_PING_COUNT": 7,
        "MEMBERSHIP_MODE": "hybrid",  # Changed from "http" to "hybrid"
        "PEER_TIMEOUT": 90,  # Jan 19, 2026: Increased from 60 to reduce false disconnections for NAT-blocked nodes
    })
    SWIM_ENABLED = _CONSTANTS["SWIM_ENABLED"]
    SWIM_BIND_PORT = _CONSTANTS["SWIM_BIND_PORT"]
    SWIM_FAILURE_TIMEOUT = _CONSTANTS["SWIM_FAILURE_TIMEOUT"]
    SWIM_SUSPICION_TIMEOUT = _CONSTANTS["SWIM_SUSPICION_TIMEOUT"]
    SWIM_PING_INTERVAL = _CONSTANTS["SWIM_PING_INTERVAL"]
    SWIM_INDIRECT_PING_COUNT = _CONSTANTS["SWIM_INDIRECT_PING_COUNT"]
    MEMBERSHIP_MODE = _CONSTANTS["MEMBERSHIP_MODE"]
    PEER_TIMEOUT = _CONSTANTS["PEER_TIMEOUT"]


class MembershipMixin(P2PMixinBase):
    """Mixin providing SWIM-based membership management.

    Inherits from P2PMixinBase for shared event emission helpers.

    Requires the implementing class to have:
    State:
    - node_id: str - This node's ID
    - peers: dict[str, NodeInfo] - Active peers
    - peers_lock: RLock - Lock for peers dict

    Methods:
    - _emit_event(event_type, payload) - Optional event emission
    """

    MIXIN_TYPE = "membership"

    # Type hints for IDE support (implemented by P2POrchestrator)
    node_id: str
    peers: dict[str, Any]  # dict[str, NodeInfo]
    peers_lock: "RLock"

    # SWIM membership manager (set by _init_swim_membership)
    _swim_manager: Optional[Any] = None  # SwimMembershipManager
    _swim_started: bool = False

    def _init_swim_membership(self) -> bool:
        """Initialize SWIM membership protocol if enabled.

        Returns:
            True if SWIM was initialized successfully, False otherwise
        """
        if not SWIM_ENABLED:
            self._log_debug("SWIM disabled via RINGRIFT_SWIM_ENABLED")
            return False

        if not SWIM_ADAPTER_AVAILABLE:
            self._log_warning("SWIM enabled but swim_adapter not available")
            return False

        if SwimMembershipManager is None or SwimConfig is None:
            self._log_warning("SWIM enabled but SwimMembershipManager/SwimConfig not importable")
            return False

        try:
            # Create SWIM configuration
            config = SwimConfig(
                bind_port=SWIM_BIND_PORT,
                failure_timeout=SWIM_FAILURE_TIMEOUT,
                suspicion_timeout=SWIM_SUSPICION_TIMEOUT,
                ping_interval=SWIM_PING_INTERVAL,
                ping_request_group_size=SWIM_INDIRECT_PING_COUNT,
            )

            # Create membership manager with callbacks
            self._swim_manager = SwimMembershipManager(
                node_id=self.node_id,
                bind_port=SWIM_BIND_PORT,
                config=config,
                on_member_alive=self._on_swim_member_alive,
                on_member_failed=self._on_swim_member_failed,
            )

            self._log_info(
                f"SWIM membership initialized on port {SWIM_BIND_PORT} "
                f"(failure_timeout={SWIM_FAILURE_TIMEOUT}s)"
            )
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize SWIM membership: {e}")
            self._swim_manager = None
            return False

    async def _start_swim_membership(self) -> bool:
        """Start SWIM membership protocol (async).

        Call this after _init_swim_membership during orchestrator startup.

        Returns:
            True if started successfully, False otherwise
        """
        if self._swim_manager is None:
            return False

        try:
            started = await self._swim_manager.start()
            self._swim_started = started

            if started:
                self._log_info(f"SWIM membership started for node {self.node_id}")
            else:
                self._log_warning("SWIM membership failed to start")

            return started

        except Exception as e:
            self._log_error(f"Error starting SWIM membership: {e}")
            self._swim_started = False
            return False

    async def _stop_swim_membership(self) -> None:
        """Stop SWIM membership protocol (async).

        Call this during orchestrator shutdown.
        """
        if self._swim_manager is not None:
            try:
                await self._swim_manager.stop()
                self._log_info("SWIM membership stopped")
            except Exception as e:
                self._log_warning(f"Error stopping SWIM membership: {e}")
            finally:
                self._swim_started = False

    def _on_swim_member_alive(self, member_id: str) -> None:
        """Handle SWIM member becoming alive.

        Called by SwimMembershipManager when a member transitions to alive state.
        This provides ~5s failure detection vs 60+ seconds with HTTP heartbeats.

        Args:
            member_id: Node ID of the member that became alive
        """
        self._log_info(f"SWIM: member {member_id} is now ALIVE")

        # Update peer last_heartbeat to mark as alive
        with self.peers_lock:
            if member_id in self.peers:
                self.peers[member_id].last_heartbeat = time.time()
                self.peers[member_id].consecutive_failures = 0

        # Emit host online event using base class helper
        self._safe_emit_event(
            "HOST_ONLINE",
            {
                "node_id": member_id,
                "timestamp": time.time(),
                "source": "swim" if self._swim_started else "http",
            },
        )

    def _on_swim_member_failed(self, member_id: str) -> None:
        """Handle SWIM member failure detection.

        Called by SwimMembershipManager when a member is detected as failed.
        SWIM provides ~5s failure detection with suspicion mechanism to reduce
        false positives.

        Args:
            member_id: Node ID of the member that failed
        """
        self._log_warning(f"SWIM: member {member_id} is now FAILED")

        # Update peer state
        with self.peers_lock:
            if member_id in self.peers:
                peer = self.peers[member_id]
                peer.consecutive_failures += 1
                peer.last_failure_time = time.time()

        # Emit host offline event using base class helper
        self._safe_emit_event(
            "HOST_OFFLINE",
            {
                "node_id": member_id,
                "timestamp": time.time(),
                "source": "swim" if self._swim_started else "http",
            },
        )

    def is_peer_alive_hybrid(self, peer_id: str) -> bool:
        """Check if a peer is alive using SWIM or HTTP based on MEMBERSHIP_MODE.

        December 29, 2025: Fixed to use true hybrid logic - if EITHER SWIM or HTTP
        reports the peer as alive, consider it alive. This prevents SWIM startup
        delays (when alive_count=0) from blocking quorum detection.

        Args:
            peer_id: Node ID to check

        Returns:
            True if peer is alive (by SWIM or HTTP), False otherwise
        """
        swim_alive = False
        http_alive = False

        # Check SWIM if available and enabled
        if MEMBERSHIP_MODE in ("swim", "hybrid") and self._swim_started and self._swim_manager:
            try:
                swim_alive = self._swim_manager.is_peer_alive(peer_id)
            except Exception as e:
                self._log_debug(f"SWIM is_peer_alive error for {peer_id}: {e}")

        # Always check HTTP for hybrid mode (or primary if MEMBERSHIP_MODE == "http")
        with self.peers_lock:
            peer = self.peers.get(peer_id)
            if peer is not None:
                http_alive = peer.is_alive()

        # In hybrid mode: peer is alive if EITHER source reports it alive
        # This prevents SWIM startup delays from blocking leader election
        if MEMBERSHIP_MODE == "hybrid":
            return swim_alive or http_alive
        elif MEMBERSHIP_MODE == "swim":
            return swim_alive
        else:  # http mode
            return http_alive

    def get_alive_peers_hybrid(self) -> list[str]:
        """Get list of alive peer node IDs using best available method.

        December 29, 2025: Fixed to use true hybrid logic - returns union of
        SWIM and HTTP alive peers. This prevents SWIM startup delays from
        returning empty lists when HTTP shows many alive peers.

        Returns:
            List of node IDs that are currently alive (union of sources)
        """
        swim_alive: set[str] = set()
        http_alive: set[str] = set()

        # Get SWIM alive peers if available
        if MEMBERSHIP_MODE in ("swim", "hybrid") and self._swim_started and self._swim_manager:
            try:
                result = self._swim_manager.get_alive_peers()
                if result:
                    swim_alive = set(result)
            except Exception as e:
                self._log_debug(f"SWIM get_alive_peers error: {e}")

        # Get HTTP alive peers
        with self.peers_lock:
            http_alive = {
                node_id
                for node_id, peer in self.peers.items()
                if peer.is_alive()
            }

        # In hybrid mode: return union of both sources
        if MEMBERSHIP_MODE == "hybrid":
            return list(swim_alive | http_alive)
        elif MEMBERSHIP_MODE == "swim":
            return list(swim_alive)
        else:  # http mode
            return list(http_alive)

    def get_swim_membership_summary(self) -> dict[str, Any]:
        """Get SWIM membership status summary.

        Returns:
            Dict with SWIM status and membership info
        """
        if self._swim_manager is None:
            return {
                "swim_enabled": SWIM_ENABLED,
                "swim_available": SWIM_ADAPTER_AVAILABLE,
                "swim_started": False,
                "membership_mode": MEMBERSHIP_MODE,
            }

        try:
            swim_summary = self._swim_manager.get_membership_summary()
        except Exception as e:
            swim_summary = {"error": str(e)}

        return {
            "swim_enabled": SWIM_ENABLED,
            "swim_available": SWIM_ADAPTER_AVAILABLE,
            "swim_started": self._swim_started,
            "membership_mode": MEMBERSHIP_MODE,
            "swim": swim_summary,
        }

    def membership_health_check(self) -> dict[str, Any]:
        """Return health status for membership subsystem.

        Returns:
            dict with is_healthy based on membership mode and status
        """
        summary = self.get_swim_membership_summary()
        peer_count = len(getattr(self, "peers", {}))
        has_peers = peer_count > 0
        no_bootstrap = not getattr(self, "bootstrap_seeds", [])

        # Determine health based on mode
        if SWIM_ENABLED and MEMBERSHIP_MODE == "swim":
            # Pure SWIM mode - SWIM must be started
            is_healthy = summary.get("swim_started", False)
        elif SWIM_ENABLED and MEMBERSHIP_MODE == "hybrid":
            # Hybrid mode - healthy if SWIM is started OR we have HTTP peers (fallback)
            swim_ok = summary.get("swim_started", False)
            http_ok = has_peers or no_bootstrap
            is_healthy = swim_ok or http_ok
        else:
            # HTTP mode - healthy if we have peers
            is_healthy = has_peers or no_bootstrap

        return {
            "is_healthy": is_healthy,
            **summary,
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for membership mixin (DaemonManager integration).

        December 2025: Added for unified health check interface.
        Uses base class helper for standardized response format.

        Returns:
            dict with healthy status, message, and details
        """
        status = self.membership_health_check()
        is_healthy = status.get("is_healthy", False)
        mode = status.get("membership_mode", "unknown")
        message = f"Membership ({mode})" if is_healthy else f"Membership unhealthy ({mode})"
        return self._build_health_response(is_healthy, message, status)

    async def _check_and_recover_swim(self) -> bool:
        """Check SWIM health and attempt recovery if unhealthy.

        December 29, 2025: Added for auto-recovery of SWIM membership.
        Should be called periodically (e.g., every 60 seconds) from main loop.

        Returns:
            True if SWIM is healthy (or recovered), False otherwise
        """
        if self._swim_manager is None:
            # SWIM not configured, nothing to recover
            return True

        # Get health status from SWIM manager
        if hasattr(self._swim_manager, "get_health_status"):
            health = self._swim_manager.get_health_status()
        else:
            # Fallback for older versions
            health = {"healthy": self._swim_started}

        if health.get("healthy", False):
            return True

        # SWIM unhealthy, attempt recovery
        reason = health.get("reason", "unknown")
        self._log_warning(f"SWIM membership unhealthy ({reason}), attempting recovery")

        # Track recovery attempts to avoid infinite loops
        if not hasattr(self, "_swim_recovery_attempts"):
            self._swim_recovery_attempts = 0
            self._swim_last_recovery_time = 0.0

        # Rate limit recovery attempts (max 1 per 5 minutes)
        import time
        now = time.time()
        if now - self._swim_last_recovery_time < 300:
            self._log_debug("SWIM recovery rate-limited, skipping")
            return False

        self._swim_recovery_attempts += 1
        self._swim_last_recovery_time = now

        # Attempt restart
        if hasattr(self._swim_manager, "restart"):
            try:
                success = await self._swim_manager.restart()
                if success:
                    self._swim_started = True
                    self._swim_recovery_attempts = 0  # Reset on success
                    self._log_info("SWIM membership recovered successfully")
                    self._safe_emit_event(
                        "SWIM_RECOVERED",
                        {
                            "node_id": self.node_id,
                            "timestamp": now,
                            "recovery_attempts": self._swim_recovery_attempts,
                        },
                    )
                    return True
                else:
                    self._log_error(
                        f"SWIM recovery failed (attempt {self._swim_recovery_attempts})"
                    )
                    return False
            except Exception as e:
                self._log_error(f"SWIM recovery exception: {e}")
                return False
        else:
            # Fallback: stop and start
            await self._stop_swim_membership()
            success = await self._start_swim_membership()
            if success:
                self._log_info("SWIM membership recovered via stop/start")
                return True
            return False

    def get_swim_health(self) -> dict[str, Any]:
        """Get detailed SWIM health information.

        December 29, 2025: Added for observability.

        Returns:
            Dict with SWIM health status and metrics
        """
        if self._swim_manager is None:
            return {
                "swim_configured": False,
                "swim_enabled": SWIM_ENABLED,
                "swim_available": SWIM_ADAPTER_AVAILABLE,
                "membership_mode": MEMBERSHIP_MODE,
            }

        # Get health from manager if available
        if hasattr(self._swim_manager, "get_health_status"):
            health = self._swim_manager.get_health_status()
        else:
            health = self.get_swim_membership_summary()

        # Add mixin-level info
        health["swim_configured"] = True
        health["swim_started_mixin"] = self._swim_started
        health["membership_mode"] = MEMBERSHIP_MODE

        # Add recovery stats if tracked
        if hasattr(self, "_swim_recovery_attempts"):
            health["recovery_attempts"] = self._swim_recovery_attempts
            health["last_recovery_time"] = self._swim_last_recovery_time

        return health

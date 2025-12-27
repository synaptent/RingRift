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
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from threading import RLock

    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Import SWIM adapter with graceful fallback
try:
    from app.p2p.swim_adapter import SwimConfig, SwimMembershipManager

    SWIM_ADAPTER_AVAILABLE = True
except ImportError:
    SWIM_ADAPTER_AVAILABLE = False
    SwimMembershipManager = None
    SwimConfig = None
    logger.debug("swim_adapter not available - SWIM features disabled")

# Import constants with fallbacks
try:
    from scripts.p2p.constants import (
        MEMBERSHIP_MODE,
        PEER_TIMEOUT,
        SWIM_BIND_PORT,
        SWIM_ENABLED,
        SWIM_FAILURE_TIMEOUT,
        SWIM_INDIRECT_PING_COUNT,
        SWIM_PING_INTERVAL,
        SWIM_SUSPICION_TIMEOUT,
    )
except ImportError:
    SWIM_ENABLED = False
    SWIM_BIND_PORT = 7947
    SWIM_FAILURE_TIMEOUT = 5.0
    SWIM_SUSPICION_TIMEOUT = 3.0
    SWIM_PING_INTERVAL = 1.0
    SWIM_INDIRECT_PING_COUNT = 3
    MEMBERSHIP_MODE = "http"
    PEER_TIMEOUT = 90


class MembershipMixin:
    """Mixin providing SWIM-based membership management.

    Requires the implementing class to have:
    State:
    - node_id: str - This node's ID
    - peers: dict[str, NodeInfo] - Active peers
    - peers_lock: RLock - Lock for peers dict

    Methods:
    - _emit_event(event_type, payload) - Optional event emission
    """

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
            logger.debug("SWIM disabled via RINGRIFT_SWIM_ENABLED")
            return False

        if not SWIM_ADAPTER_AVAILABLE:
            logger.warning("SWIM enabled but swim_adapter not available")
            return False

        if SwimMembershipManager is None or SwimConfig is None:
            logger.warning("SWIM enabled but SwimMembershipManager/SwimConfig not importable")
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

            logger.info(
                f"SWIM membership initialized on port {SWIM_BIND_PORT} "
                f"(failure_timeout={SWIM_FAILURE_TIMEOUT}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SWIM membership: {e}", exc_info=True)
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
                logger.info(f"SWIM membership started for node {self.node_id}")
            else:
                logger.warning("SWIM membership failed to start")

            return started

        except Exception as e:
            logger.error(f"Error starting SWIM membership: {e}", exc_info=True)
            self._swim_started = False
            return False

    async def _stop_swim_membership(self) -> None:
        """Stop SWIM membership protocol (async).

        Call this during orchestrator shutdown.
        """
        if self._swim_manager is not None:
            try:
                await self._swim_manager.stop()
                logger.info("SWIM membership stopped")
            except Exception as e:
                logger.warning(f"Error stopping SWIM membership: {e}")
            finally:
                self._swim_started = False

    def _on_swim_member_alive(self, member_id: str) -> None:
        """Handle SWIM member becoming alive.

        Called by SwimMembershipManager when a member transitions to alive state.
        This provides ~5s failure detection vs 60+ seconds with HTTP heartbeats.

        Args:
            member_id: Node ID of the member that became alive
        """
        logger.info(f"SWIM: member {member_id} is now ALIVE")

        # Update peer last_heartbeat to mark as alive
        with self.peers_lock:
            if member_id in self.peers:
                self.peers[member_id].last_heartbeat = time.time()
                self.peers[member_id].consecutive_failures = 0

        # Emit host online event
        self._emit_host_online(member_id)

    def _on_swim_member_failed(self, member_id: str) -> None:
        """Handle SWIM member failure detection.

        Called by SwimMembershipManager when a member is detected as failed.
        SWIM provides ~5s failure detection with suspicion mechanism to reduce
        false positives.

        Args:
            member_id: Node ID of the member that failed
        """
        logger.warning(f"SWIM: member {member_id} is now FAILED")

        # Update peer state
        with self.peers_lock:
            if member_id in self.peers:
                peer = self.peers[member_id]
                peer.consecutive_failures += 1
                peer.last_failure_time = time.time()

        # Emit host offline event
        self._emit_host_offline(member_id)

    def is_peer_alive_hybrid(self, peer_id: str) -> bool:
        """Check if a peer is alive using SWIM or HTTP based on MEMBERSHIP_MODE.

        Args:
            peer_id: Node ID to check

        Returns:
            True if peer is alive, False otherwise
        """
        # Use SWIM if available and enabled
        if MEMBERSHIP_MODE in ("swim", "hybrid") and self._swim_started and self._swim_manager:
            try:
                return self._swim_manager.is_peer_alive(peer_id)
            except Exception as e:
                logger.debug(f"SWIM is_peer_alive error for {peer_id}: {e}")
                # Fall through to HTTP check

        # HTTP fallback (or primary if MEMBERSHIP_MODE == "http")
        with self.peers_lock:
            peer = self.peers.get(peer_id)
            if peer is None:
                return False
            return peer.is_alive()

    def get_alive_peers_hybrid(self) -> list[str]:
        """Get list of alive peer node IDs using best available method.

        Uses SWIM when available in swim/hybrid mode, falls back to HTTP heartbeats.

        Returns:
            List of node IDs that are currently alive
        """
        # Use SWIM if available and enabled
        if MEMBERSHIP_MODE in ("swim", "hybrid") and self._swim_started and self._swim_manager:
            try:
                swim_alive = self._swim_manager.get_alive_peers()
                if swim_alive:
                    return swim_alive
            except Exception as e:
                logger.debug(f"SWIM get_alive_peers error: {e}")
                # Fall through to HTTP check

        # HTTP fallback
        with self.peers_lock:
            return [
                node_id
                for node_id, peer in self.peers.items()
                if peer.is_alive()
            ]

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

    def _emit_host_online(self, node_id: str) -> None:
        """Emit event when a host comes online.

        Args:
            node_id: Node ID that came online
        """
        try:
            # Check if _emit_event exists (optional method)
            emit_fn = getattr(self, "_emit_event", None)
            if callable(emit_fn):
                emit_fn(
                    "HOST_ONLINE",
                    {
                        "node_id": node_id,
                        "timestamp": time.time(),
                        "source": "swim" if self._swim_started else "http",
                    },
                )
        except Exception as e:
            logger.debug(f"Error emitting HOST_ONLINE event: {e}")

    def _emit_host_offline(self, node_id: str) -> None:
        """Emit event when a host goes offline.

        Args:
            node_id: Node ID that went offline
        """
        try:
            # Check if _emit_event exists (optional method)
            emit_fn = getattr(self, "_emit_event", None)
            if callable(emit_fn):
                emit_fn(
                    "HOST_OFFLINE",
                    {
                        "node_id": node_id,
                        "timestamp": time.time(),
                        "source": "swim" if self._swim_started else "http",
                    },
                )
        except Exception as e:
            logger.debug(f"Error emitting HOST_OFFLINE event: {e}")

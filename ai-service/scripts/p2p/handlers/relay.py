"""Relay HTTP Handlers Mixin.

Provides HTTP endpoints for NAT-blocked node communication.
Acts as a relay hub allowing nodes behind NAT/firewalls to participate
in the cluster by polling for commands from the leader.

Inherits from BaseP2PHandler for consistent response formatting and
authentication utilities.

Usage:
    class P2POrchestrator(RelayHandlersMixin, ...):
        pass

Endpoints:
    POST /relay/heartbeat - NAT-blocked node sends heartbeat to leader
    GET /relay/commands - NAT-blocked node polls for pending commands
    POST /relay/command - Leader enqueues command for NAT-blocked node
    POST /relay/response - NAT-blocked node sends command response

Relay Architecture:
    NAT-blocked nodes cannot receive incoming connections, so they:
    1. Send heartbeats to leader via /relay/heartbeat
    2. Poll for commands via /relay/commands
    3. Execute commands locally and respond via /relay/response

    Leader queues commands for NAT-blocked nodes and delivers them
    when the node polls. Max batch size: RELAY_COMMAND_MAX_BATCH (16).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_GOSSIP,
    HANDLER_TIMEOUT_TOURNAMENT,
)

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Import constants
try:
    from scripts.p2p.constants import RELAY_COMMAND_MAX_BATCH
except ImportError:
    RELAY_COMMAND_MAX_BATCH = 16

# Import models
try:
    from scripts.p2p.models import NodeInfo
    from scripts.p2p.types import NodeRole
except ImportError:
    NodeInfo = None
    NodeRole = None

# NonBlockingAsyncLockWrapper helper (Jan 2026 - fix lock contention)
try:
    from scripts.p2p.network import NonBlockingAsyncLockWrapper
except ImportError:
    # Fallback - use synchronous lock acquisition to ensure correct RLock semantics
    # January 12, 2026: FIXED - Cannot use asyncio.to_thread() because threading.RLock
    # requires the same thread to acquire and release. The thread pool doesn't guarantee
    # thread affinity, causing "cannot release un-acquired lock" errors.
    import asyncio
    import contextlib

    class NonBlockingAsyncLockWrapper:
        """Fallback lock wrapper for async contexts with timeout support."""
        def __init__(self, lock, lock_name: str = "unknown", timeout: float = 5.0):
            self._lock = lock
            self._lock_name = lock_name
            self._timeout = timeout
            self._acquired = False

        async def __aenter__(self):
            # Synchronous acquire ensures same-thread semantics for RLock
            acquired = self._lock.acquire(blocking=True, timeout=self._timeout)
            if not acquired:
                raise asyncio.TimeoutError(
                    f"Lock {self._lock_name} acquisition timed out after {self._timeout}s"
                )
            self._acquired = True
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self._acquired:
                self._lock.release()
                self._acquired = False
            return False


class RelayHandlersMixin(BaseP2PHandler):
    """Mixin providing relay HTTP handlers for NAT-blocked nodes.

    Inherits from BaseP2PHandler for consistent response formatting
    (json_response, error_response) and authentication utilities.

    Requires the implementing class to have:
    - node_id: str
    - leader_id: str | None
    - auth_token: str | None
    - self_info: NodeInfo
    - peers: dict[str, NodeInfo]
    - peers_lock: threading.RLock
    - relay_lock: threading.RLock
    - relay_command_queue: dict[str, list]
    - _update_self_info() method
    - _get_leader_peer() method
    - _has_voter_quorum() method
    - _is_request_authorized() method
    - _enqueue_relay_command() method
    """

    # Type hints for IDE support
    node_id: str
    leader_id: str | None
    auth_token: str | None

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_relay_heartbeat(self, request: web.Request) -> web.Response:
        """POST /relay/heartbeat - Accept heartbeat from NAT-blocked node.

        NAT-blocked nodes (e.g., Vast.ai behind carrier NAT) can't receive
        incoming connections. They use this endpoint to:
        1. Send their status to the leader
        2. Get back the full cluster peer list
        3. Mark themselves as nat_blocked so leader doesn't try to reach them

        Request body: Same as regular heartbeat (NodeInfo dict)
        Response: {
            "self": NodeInfo,  # Leader's info
            "peers": {node_id: NodeInfo},  # All known peers including NAT-blocked
            "leader_id": str
        }
        """
        try:
            data = await request.json()
            relay_ack = data.get("relay_ack") or []
            relay_results = data.get("relay_results") or []
            peer_info = NodeInfo.from_dict(data)
            if not peer_info.reported_host:
                peer_info.reported_host = peer_info.host
            if not peer_info.reported_port:
                peer_info.reported_port = peer_info.port
            peer_info.last_heartbeat = time.time()
            peer_info.nat_blocked = True  # Mark as NAT-blocked
            peer_info.nat_blocked_since = (
                peer_info.nat_blocked_since or time.time()
            )  # Track when blocked
            peer_info.relay_via = self.node_id  # This node is their relay

            # Get their real IP from the request (for logging/debugging)
            forwarded_for = (
                request.headers.get("X-Forwarded-For")
                or request.headers.get("X-Real-IP")
                or request.headers.get("CF-Connecting-IP")
            )
            real_ip = (
                forwarded_for.split(",")[0].strip() if forwarded_for else request.remote
            )
            if real_ip:
                peer_info.host = real_ip

            # STABILITY FIX: Correct stale leader role claims
            if peer_info.role == NodeRole.LEADER and peer_info.node_id != self.node_id:
                actual_leader = self.leader_id
                if actual_leader and actual_leader != peer_info.node_id:
                    peer_info.role = NodeRole.FOLLOWER

            # Store in peers list (they're part of the cluster even if not directly reachable)
            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                self.peers[peer_info.node_id] = peer_info

            logger.info(f"Relay heartbeat from {peer_info.node_id} (real IP: {real_ip})")

            # Apply relay ACKs/results and return any queued commands.
            commands_to_send: list[dict[str, Any]] = []
            async with NonBlockingAsyncLockWrapper(self.relay_lock, "relay_lock", timeout=5.0):
                queue = list(self.relay_command_queue.get(peer_info.node_id, []))
                now = time.time()
                queue = [
                    cmd
                    for cmd in queue
                    if float(cmd.get("expires_at", 0.0) or 0.0) > now
                ]

                if relay_ack:
                    ack_set = {str(c) for c in relay_ack if c}
                    queue = [
                        cmd for cmd in queue if str(cmd.get("id", "")) not in ack_set
                    ]

                if relay_results:
                    for item in relay_results:
                        try:
                            cmd_id = str(item.get("id") or "")
                            ok = bool(item.get("ok", False))
                            err = str(item.get("error") or "")
                            if not cmd_id:
                                continue
                            if ok:
                                logger.info(
                                    f"Relay command {cmd_id} on {peer_info.node_id}: ok"
                                )
                            else:
                                logger.info(
                                    f"Relay command {cmd_id} on {peer_info.node_id}: failed {err[:200]}"
                                )
                        except AttributeError:
                            continue

                self.relay_command_queue[peer_info.node_id] = queue
                commands_to_send = queue[:RELAY_COMMAND_MAX_BATCH]

            # Return cluster state so they can see all peers
            # Feb 2026: Fire-and-forget self_info refresh to prevent event loop blocking
            # under high CPU load (was causing 56-64s timeouts on /relay/peers)
            try:
                safe_create_task(self._update_self_info_async(), name="relay-self-info-refresh")
            except Exception:
                pass  # Fire-and-forget, don't block on errors

            # Feb 2026: Use lock-free peer snapshot for read-only access
            if hasattr(self, "_peer_snapshot"):
                peers_snapshot = self._peer_snapshot.get_snapshot()
                peers = {k: v.to_dict() for k, v in peers_snapshot.items()}
            else:
                async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                    peers = {k: v.to_dict() for k, v in self.peers.items()}

            effective_leader = self._get_leader_peer()
            effective_leader_id = effective_leader.node_id if effective_leader else None
            return self.json_response(
                {
                    "success": True,
                    "self": self.self_info.to_dict(),
                    "peers": peers,
                    # IMPORTANT: only advertise a leader_id when it is actually reachable
                    # and currently reporting itself as leader. Persisted/stale leader_id
                    # values are surfaced separately so bootstrapping nodes don't get
                    # stuck pointing at a non-leader.
                    "leader_id": effective_leader_id,
                    "effective_leader_id": effective_leader_id,
                    "last_known_leader_id": self.leader_id,
                    "relay_node": self.node_id,
                    # Propagate the stable voter set so nodes that boot without local
                    # config still enable quorum gating and avoid split-brain.
                    "voter_node_ids": list(
                        getattr(self, "voter_node_ids", []) or []
                    ),
                    "voter_quorum_size": int(
                        getattr(self, "voter_quorum_size", 0) or 0
                    ),
                    "voter_quorum_ok": self._has_voter_quorum(),
                    "voter_config_source": str(
                        getattr(self, "voter_config_source", "") or ""
                    ),
                    "commands": commands_to_send,
                }
            )

        except Exception as e:
            return self.error_response(str(e), status=400)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_relay_enqueue(self, request: web.Request) -> web.Response:
        """POST /relay/enqueue - Enqueue a command for a NAT-blocked node on this relay.

        This enables multi-hop operation when NAT-blocked nodes can reach a
        public relay hub (e.g., AWS) but cannot reach the cluster leader
        directly (e.g., TUN-less Tailscale inside some containers).

        Request body:
          {
            "target_node_id": "node-id",
            "type": "start_job" | "cleanup" | ...,
            "payload": { ... }
          }

        Response:
          { "success": true, "id": "<cmd_id>" }
        """
        try:
            data = await request.json()
        except AttributeError:
            data = {}

        try:
            target_node_id = str(
                data.get("target_node_id") or data.get("node_id") or ""
            ).strip()
            cmd_type = str(data.get("type") or data.get("cmd_type") or "").strip()
            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}
        except AttributeError:
            target_node_id = ""
            cmd_type = ""
            payload = {}

        if not target_node_id or not cmd_type:
            return self.error_response(
                "target_node_id and type are required",
                status=400,
                error_code="invalid_request",
            )

        cmd_id = self._enqueue_relay_command(target_node_id, cmd_type, payload)
        if not cmd_id:
            return self.error_response(
                "queue_full",
                status=429,
                error_code="queue_full",
            )

        return self.json_response({"success": True, "id": cmd_id})

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_relay_peers(self, request: web.Request) -> web.Response:
        """GET /relay/peers - Get list of all peers including NAT-blocked ones.

        Used by nodes to discover the full cluster including NAT-blocked members.

        Feb 2026: Fixed event loop blocking under high CPU load (load avg 219).
        Three changes to prevent 56-64s timeouts:
        1. Fire-and-forget _update_self_info_async instead of awaiting it
        2. Use lock-free _peer_snapshot instead of peers_lock
        3. Cache response for 5s TTL to avoid repeated work under load
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return self.error_response("unauthorized", status=401)

            # Check response cache (5s TTL) to avoid repeated work under load
            now = time.time()
            cache_attr = "_relay_peers_cache"
            cache_time_attr = "_relay_peers_cache_time"
            cached_response = getattr(self, cache_attr, None)
            cached_time = getattr(self, cache_time_attr, 0.0)
            if cached_response is not None and (now - cached_time) < 5.0:
                return self.json_response(cached_response)

            # Fire-and-forget: schedule background refresh of self_info
            # Don't await - prevents blocking when subprocess resource detection
            # is slow under high CPU load
            try:
                safe_create_task(self._update_self_info_async(), name="relay-self-info-refresh")
            except Exception:
                pass  # Fire-and-forget, don't block on errors

            effective_leader = self._get_leader_peer()

            # Use lock-free peer snapshot instead of peers_lock to avoid blocking
            if hasattr(self, "_peer_snapshot"):
                peers_snapshot = self._peer_snapshot.get_snapshot()
                all_peers = {k: v.to_dict() for k, v in peers_snapshot.items()}
            else:
                # Fallback to lock-based access if snapshot not available
                async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                    all_peers = {k: v.to_dict() for k, v in self.peers.items()}

            # Separate NAT-blocked and directly reachable
            nat_blocked = {k: v for k, v in all_peers.items() if v.get("nat_blocked")}
            direct = {k: v for k, v in all_peers.items() if not v.get("nat_blocked")}

            response_data = {
                "success": True,
                "leader_id": (
                    effective_leader.node_id if effective_leader else self.leader_id
                ),
                "effective_leader_id": (
                    effective_leader.node_id if effective_leader else None
                ),
                "total_peers": len(all_peers),
                "direct_peers": len(direct),
                "nat_blocked_peers": len(nat_blocked),
                "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                "voter_quorum_size": int(
                    getattr(self, "voter_quorum_size", 0) or 0
                ),
                "voter_quorum_ok": self._has_voter_quorum(),
                "voter_config_source": str(
                    getattr(self, "voter_config_source", "") or ""
                ),
                "peers": all_peers,
            }

            # Cache the response for 5s TTL
            try:
                setattr(self, cache_attr, response_data)
                setattr(self, cache_time_attr, now)
            except Exception:
                pass  # Don't fail if caching fails

            return self.json_response(response_data)

        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_relay_status(self, request: web.Request) -> web.Response:
        """GET /relay/status - Get relay queue status for debugging.

        Shows pending commands per NAT-blocked node including command ages.
        Useful for diagnosing relay delivery issues.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return self.error_response("unauthorized", status=401)

            now = time.time()
            queue_status = {}
            total_pending = 0

            for node_id, commands in self.relay_command_queue.items():
                if not commands:
                    continue
                cmd_info = []
                for cmd in commands:
                    age_secs = now - cmd.get("ts", now)
                    cmd_info.append(
                        {
                            "id": cmd.get("id", ""),
                            "type": cmd.get("cmd", ""),
                            "age_secs": round(age_secs, 1),
                            "stale": age_secs > 300,  # >5 min is stale
                        }
                    )
                queue_status[node_id] = {
                    "pending_count": len(commands),
                    "commands": cmd_info,
                    "oldest_age_secs": round(
                        max((now - c.get("ts", now)) for c in commands), 1
                    )
                    if commands
                    else 0,
                }
                total_pending += len(commands)

            # Get NAT-blocked nodes for context
            # Feb 2026: Use lock-free peer snapshot to avoid blocking event loop
            if hasattr(self, "_peer_snapshot"):
                peers_snapshot = self._peer_snapshot.get_snapshot()
                nat_blocked_nodes = [
                    nid
                    for nid, p in peers_snapshot.items()
                    if getattr(p, "nat_blocked", False)
                ]
            else:
                with self.peers_lock:
                    nat_blocked_nodes = [
                        nid
                        for nid, p in self.peers.items()
                        if getattr(p, "nat_blocked", False)
                    ]

            return self.json_response(
                {
                    "success": True,
                    "total_pending_commands": total_pending,
                    "nat_blocked_nodes": nat_blocked_nodes,
                    "nodes_with_pending": list(queue_status.keys()),
                    "queues": queue_status,
                }
            )

        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_relay_health(self, request: web.Request) -> web.Response:
        """GET /relay/health - Get relay transport health for failover monitoring.

        Jan 3, 2026: Added for multi-relay failover monitoring. Shows health
        of all configured relay nodes and their success rates.

        Returns:
            JSON with relay health summary including:
            - total_relays: Number of configured relays
            - healthy_relays: Number of healthy relays
            - unhealthy_relays: Number of unhealthy relays
            - relays: Per-relay health details
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return self.error_response("unauthorized", status=401)

            # Get relay health summary from failover integration
            if hasattr(self, "get_relay_health_summary"):
                health_summary = self.get_relay_health_summary()
            else:
                health_summary = {"error": "Relay health tracking not available"}

            return self.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "relay_health": health_summary,
                }
            )

        except Exception as e:
            return self.error_response(str(e), status=500)

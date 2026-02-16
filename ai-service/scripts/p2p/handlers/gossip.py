"""Gossip Protocol HTTP Handlers Mixin.

Provides HTTP endpoints for decentralized state sharing via gossip protocol.
Nodes exchange state with random peers, propagating information cluster-wide
without leader coordination.

Usage:
    class P2POrchestrator(GossipHandlersMixin, ...):
        pass

Endpoints:
    POST /gossip - Exchange state with peer (bi-directional gossip)
    POST /gossip/push - Receive one-way state push from peer
    GET /gossip/manifest - Get local data manifest for sync planning
    GET /gossip/cluster-manifest - Get aggregated cluster manifest

Compression:
    Supports gzip-compressed requests/responses for bandwidth efficiency.
    Magic byte detection (0x1f 0x8b) ensures graceful handling of
    clients that set Content-Encoding: gzip but send raw JSON.

December 2025: Migrated to use handlers_base.py utilities for event bridge.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

# Dec 2025: Use consolidated handler utilities
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.handlers_base import get_event_bridge
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_GOSSIP,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Event bridge manager for safe event emission (Dec 2025 consolidation)
_event_bridge = get_event_bridge()


class GossipHandlersMixin(BaseP2PHandler):
    """Mixin providing gossip protocol HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting and
    error handling across all P2P handler mixins.

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - leader_id: str | None
    - self_info: NodeInfo
    - auth_token: str | None
    - _gossip_peer_states: dict
    - local_data_manifest: DataManifest | None
    - _is_request_authorized() method
    - _process_gossip_response() method
    - _record_gossip_metrics() method
    - _update_self_info() method
    - _get_gossip_known_states() method
    - _has_voter_quorum() method
    """

    # Type hints for IDE support
    node_id: str
    leader_id: str | None
    auth_token: str | None

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_gossip(self, request: web.Request) -> web.Response:
        """POST /gossip - Receive gossip from peer and respond with our state.

        GOSSIP PROTOCOL: Decentralized state sharing between nodes.
        Each node shares its state with random peers, and information
        propagates through the cluster without leader coordination.

        GOSSIP COMPRESSION: Supports gzip-compressed requests and responses
        to reduce network bandwidth. Check Content-Encoding header.

        Request body:
        {
            "sender": "node-id",
            "sender_state": { state dict },
            "known_states": { node_id -> state dict }
        }

        Response:
        {
            "sender_state": { our state },
            "known_states": { our known states },
            "peer_manifests": { node_id -> manifest summary }
        }
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return self.error_response("unauthorized", status=401)

            # GOSSIP COMPRESSION: Handle gzip-compressed requests
            # Dec 2025: Enhanced to check gzip magic bytes (0x1f 0x8b) before decompression
            # to handle clients that set Content-Encoding: gzip but send raw JSON
            content_encoding = request.headers.get("Content-Encoding", "")
            if content_encoding == "gzip":
                body = await request.read()
                # Check for gzip magic bytes before attempting decompression
                is_gzip = len(body) >= 2 and body[:2] == b"\x1f\x8b"
                if is_gzip:
                    try:
                        # Dec 30, 2025: Use asyncio.to_thread() to avoid blocking event loop
                        # Large payloads (100KB+) can take 50-200ms to decompress
                        decompressed = await asyncio.to_thread(gzip.decompress, body)
                        data = json.loads(decompressed.decode("utf-8"))
                    except (gzip.BadGzipFile, OSError) as e:
                        # Decompression failed despite magic bytes - treat as raw JSON
                        logger.warning(f"Gzip decompression failed despite magic bytes: {e}")
                        data = json.loads(body.decode("utf-8"))
                else:
                    # Header claimed gzip but no magic bytes - client sent raw JSON
                    logger.debug(
                        f"Sender {request.remote} claimed gzip but sent raw JSON (no magic bytes)"
                    )
                    data = json.loads(body.decode("utf-8"))
            else:
                data = await request.json()
        except (json.JSONDecodeError, AttributeError, ConnectionResetError, aiohttp.ClientError) as e:
            # Jan 2026: Handle connection reset and client errors during request read
            logger.debug(f"Gossip request read failed: {type(e).__name__}: {e}")
            data = {}

        try:
            # Process incoming gossip
            self._process_gossip_response(data)
            self._record_gossip_metrics("received")

            # Prepare our response
            now = time.time()
            # Feb 2026: Fire-and-forget self_info refresh to prevent event loop blocking.
            # Under high CPU (90%+), awaiting _update_self_info_async blocks for 50+ seconds
            # due to subprocess-based resource detection competing for CPU time.
            # Use cached self_info data for the response instead.
            try:
                asyncio.create_task(self._update_self_info_async())
            except Exception:
                pass  # Don't block gossip on self_info refresh failures

            our_state = {
                "node_id": self.node_id,
                "timestamp": now,
                "version": int(now * 1000),
                "role": self.role.value if hasattr(self.role, "value") else str(self.role),
                "leader_id": self.leader_id,
                "leader_lease_expires": getattr(self, "leader_lease_expires", 0),
                "selfplay_jobs": getattr(self.self_info, "selfplay_jobs", 0),
                "training_jobs": getattr(self.self_info, "training_jobs", 0),
                "gpu_percent": getattr(self.self_info, "gpu_percent", 0),
                "cpu_percent": getattr(self.self_info, "cpu_percent", 0),
                "memory_percent": getattr(self.self_info, "memory_percent", 0),
                "disk_percent": getattr(self.self_info, "disk_percent", 0),
                "has_gpu": getattr(self.self_info, "has_gpu", False),
                "gpu_name": getattr(self.self_info, "gpu_name", ""),
                "voter_quorum_ok": self._has_voter_quorum(),
            }

            # Include manifest summary
            local_manifest = getattr(self, "local_data_manifest", None)
            if local_manifest:
                our_state["manifest_summary"] = {
                    "total_files": getattr(local_manifest, "total_files", 0),
                    "selfplay_games": getattr(local_manifest, "selfplay_games", 0),
                    "collected_at": getattr(local_manifest, "collected_at", 0),
                }

            # Get known states to propagate
            known_states = self._get_gossip_known_states()

            # Include peer manifests for P2P sync
            peer_manifests = {}
            local_manifest = getattr(self, "local_data_manifest", None)
            if local_manifest and hasattr(local_manifest, "to_dict"):
                peer_manifests[self.node_id] = local_manifest.to_dict()

            response_data = {
                "sender_state": our_state,
                "known_states": known_states,
                "peer_manifests": peer_manifests,
            }

            # GOSSIP COMPRESSION: Send compressed response if client accepts it
            # Always compress responses for efficiency
            # Dec 30, 2025: Use asyncio.to_thread() to avoid blocking event loop
            response_json = json.dumps(response_data).encode("utf-8")
            compressed_response = await asyncio.to_thread(
                gzip.compress, response_json, 6  # compresslevel=6
            )
            return web.Response(
                body=compressed_response,
                content_type="application/json",
                headers={"Content-Encoding": "gzip"},
            )

        except Exception as e:
            # Dec 2025: Added logging for debugging gossip failures
            logger.error(f"Error handling gossip request: {e}", exc_info=True)
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_gossip_anti_entropy(self, request: web.Request) -> web.Response:
        """POST /gossip/anti-entropy - Full state exchange for consistency repair.

        ANTI-ENTROPY: Unlike regular gossip (which shares recent state only),
        this endpoint exchanges ALL known states to ensure eventual consistency.
        Used periodically to catch any missed updates from network issues.

        Request body:
        {
            "anti_entropy": true,
            "sender": "node-id",
            "timestamp": <float>,
            "all_known_states": { node_id -> state dict }
        }

        Response: Same format with our full state knowledge.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return self.error_response("unauthorized", status=401)

            data = await request.json()
        except (json.JSONDecodeError, AttributeError, ConnectionResetError, aiohttp.ClientError) as e:
            # Jan 2026: Handle connection reset and client errors during request read
            logger.debug(f"Anti-entropy request read failed: {type(e).__name__}: {e}")
            data = {}

        # Jan 3, 2026 Sprint 13.3: Extract sender for per-peer lock
        sender_id = data.get("sender", "")

        try:
            self._record_gossip_metrics("received")

            # Initialize gossip state storage if needed
            if not hasattr(self, "_gossip_peer_states"):
                self._gossip_peer_states = {}

            # Jan 3, 2026 Sprint 13.3: Acquire per-peer lock to serialize messages from same sender
            # This prevents concurrent message handling from the same peer corrupting state
            peer_lock = None
            if sender_id and hasattr(self, "_get_peer_lock"):
                peer_lock = self._get_peer_lock(sender_id)
                try:
                    await asyncio.wait_for(peer_lock.acquire(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._log_warning(f"[Gossip] Per-peer lock timeout for {sender_id}, proceeding without lock")
                    peer_lock = None  # Proceed without lock on timeout

            # Process ALL states from peer
            peer_states = data.get("all_known_states", {})
            updates = 0
            new_peers = []

            # Jan 3, 2026: Use sync lock to prevent race with cleanup
            # Feb 2026: Use timeout to prevent deadlock between event loop thread
            # and asyncio.to_thread() thread (both trying to acquire the RLock)
            sync_lock = getattr(self, "_gossip_state_sync_lock", None)
            lock_acquired = False
            if sync_lock is not None:
                lock_acquired = await asyncio.to_thread(
                    sync_lock.acquire, True, 2.0  # blocking=True, timeout=2.0
                )
            try:
                if sync_lock is None or lock_acquired:
                    for node_id, state in peer_states.items():
                        if node_id == self.node_id:
                            continue
                        existing = self._gossip_peer_states.get(node_id, {})
                        is_new_peer = not existing
                        if state.get("version", 0) > existing.get("version", 0):
                            self._gossip_peer_states[node_id] = state
                            updates += 1
                            self._record_gossip_metrics("update", node_id)
                            if is_new_peer:
                                new_peers.append((node_id, state))
            finally:
                if sync_lock is not None and lock_acquired:
                    sync_lock.release()

            # Emit node online events for newly discovered peers (Dec 2025 consolidation)
            for peer_id, peer_state in new_peers:
                await _event_bridge.emit("p2p_node_online", {
                    "node_id": peer_id,
                    "host_type": peer_state.get("role", ""),
                    "capabilities": {
                        "has_gpu": peer_state.get("has_gpu", False),
                        "gpu_name": peer_state.get("gpu_name", ""),
                    },
                })

            if updates > 0:
                self._record_gossip_metrics("anti_entropy")

            # Prepare our full state response
            now = time.time()
            # Feb 2026: Use async version to prevent event loop blocking
            await self._update_self_info_async()

            all_known_states = {}

            # Include all our known peer states
            for node_id, state in self._gossip_peer_states.items():
                all_known_states[node_id] = state

            # Include our own state
            all_known_states[self.node_id] = {
                "node_id": self.node_id,
                "timestamp": now,
                "version": int(now * 1000),
                "role": self.role.value if hasattr(self.role, "value") else str(self.role),
                "leader_id": self.leader_id,
                "selfplay_jobs": getattr(self.self_info, "selfplay_jobs", 0),
                "training_jobs": getattr(self.self_info, "training_jobs", 0),
                "gpu_percent": getattr(self.self_info, "gpu_percent", 0),
                "cpu_percent": getattr(self.self_info, "cpu_percent", 0),
                "memory_percent": getattr(self.self_info, "memory_percent", 0),
                "disk_percent": getattr(self.self_info, "disk_percent", 0),
            }

            return self.json_response({
                "anti_entropy": True,
                "sender": self.node_id,
                "timestamp": now,
                "all_known_states": all_known_states,
                "updates_applied": updates,
            })

        except Exception as e:
            # Dec 2025: Added logging for debugging anti-entropy failures
            logger.error(f"Error handling anti-entropy request: {e}", exc_info=True)
            return self.error_response(str(e), status=500)

        finally:
            # Jan 3, 2026 Sprint 13.3: Release per-peer lock
            if peer_lock is not None and peer_lock.locked():
                peer_lock.release()

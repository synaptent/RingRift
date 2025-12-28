"""Gossip Protocol Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides the core gossip protocol for decentralized state sharing.

Usage:
    class P2POrchestrator(GossipProtocolMixin, GossipMetricsMixin, ...):
        pass

The gossip protocol enables:
- Decentralized state propagation (O(log N) instead of O(N))
- Works without a leader
- Resilient to network partitions
- Reduces load on leader

Phase 3 extraction - Dec 26, 2025
Phase 4 consolidation - Dec 28, 2025: Now inherits from P2PMixinBase
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import time
from typing import TYPE_CHECKING, Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from .constants import DEFAULT_PORT, GOSSIP_MAX_PEER_ENDPOINTS
from .p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from .models import NodeInfo, NodeRole


class GossipProtocolMixin(P2PMixinBase):
    """Mixin providing core gossip protocol functionality.

    Inherits from P2PMixinBase (Phase 4 consolidation - Dec 28, 2025) to use:
    - _log_info/_log_debug/_log_warning/_log_error for consistent logging
    - _safe_emit_event for event emission
    - _ensure_multiple_state_attrs for state initialization
    - _get_timestamp/_is_expired for timing utilities

    Requires the implementing class to have:
    - node_id: str - This node's ID
    - peers: dict[str, NodeInfo] - Known peers
    - peers_lock: threading.RLock - Lock for peers dict
    - leader_id: Optional[str] - Current leader ID
    - role: NodeRole - This node's role
    - self_info: NodeInfo - This node's info
    - verbose: bool - Enable verbose logging
    - _cluster_epoch: int - Cluster epoch for split-brain resolution

    And methods:
    - _update_self_info() - Update self node info
    - _urls_for_peer(peer, path) - Get URLs to try for a peer
    - _auth_headers() - Get auth headers for requests
    - _has_voter_quorum() - Check voter quorum status
    - _save_cluster_epoch() - Persist cluster epoch
    - _send_heartbeat_to_peer(host, port) - Send heartbeat
    - _save_peer_to_cache(node_id, host, port, tailscale_ip) - Cache peer

    From GossipMetricsMixin:
    - _record_gossip_metrics(event, peer_id, latency_ms)
    - _record_gossip_compression(original_size, compressed_size)

    Optional methods (graceful degradation if missing):
    - _get_local_active_training_configs() -> list[dict]
    - _get_local_elo_summary() -> dict
    - _get_leader_hint() -> str | None
    - _get_peer_reputation_summary() -> dict
    - _get_tournament_gossip_state() -> dict
    - _process_tournament_gossip(node_id, tournament_state)
    - _check_tournament_consensus()
    """

    # MIXIN_TYPE for P2PMixinBase logging prefix (Phase 4 consolidation)
    MIXIN_TYPE = "GOSSIP"

    # Type hints for IDE support
    node_id: str
    peers: dict[str, Any]
    peers_lock: Any
    leader_id: str | None
    role: Any  # NodeRole
    self_info: Any  # NodeInfo
    verbose: bool
    last_leader_seen: float
    _cluster_epoch: int
    _gossip_peer_states: dict[str, dict]
    _gossip_peer_manifests: dict[str, Any]
    _gossip_learned_endpoints: dict[str, dict]

    # Dec 28, 2025: Limits to prevent unbounded memory growth
    GOSSIP_MAX_PEER_STATES = 200  # Max peer states to keep
    GOSSIP_MAX_MANIFESTS = 100  # Max manifests to keep
    GOSSIP_MAX_ENDPOINTS = 100  # Max learned endpoints to keep
    GOSSIP_STATE_TTL = 3600  # 1 hour TTL for stale states
    GOSSIP_ENDPOINT_TTL = 1800  # 30 min TTL for learned endpoints

    def _init_gossip_protocol(self) -> None:
        """Initialize gossip protocol state and metrics.

        Call this in __init__ to set up gossip storage.
        Uses P2PMixinBase._ensure_multiple_state_attrs() for cleaner initialization.

        Phase 4 consolidation: Now includes GossipMetricsMixin state (previously separate).
        """
        # Phase 4 consolidation: Use base class helper instead of manual hasattr checks
        self._ensure_multiple_state_attrs({
            # Gossip protocol state
            "_gossip_peer_states": {},
            "_gossip_peer_manifests": {},
            "_gossip_learned_endpoints": {},
            "_last_gossip_time": 0.0,
            "_last_anti_entropy_repair": 0.0,
            "_last_gossip_cleanup": 0.0,
            # Gossip metrics state (merged from GossipMetricsMixin)
            "_gossip_metrics": {
                "message_sent": 0,
                "message_received": 0,
                "state_updates": 0,
                "propagation_delay_ms": [],
                "anti_entropy_repairs": 0,
                "stale_states_detected": 0,
                "last_reset": time.time(),
            },
            "_gossip_compression_stats": {
                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
                "messages_compressed": 0,
            },
        })

    def _cleanup_gossip_state(self) -> None:
        """Dec 28, 2025: Clean up stale gossip state to prevent memory growth.

        Called periodically to:
        1. Remove entries older than TTL
        2. Enforce max size limits with LRU eviction
        """
        now = time.time()

        # Rate limit cleanup to every 5 minutes
        if now - getattr(self, "_last_gossip_cleanup", 0) < 300:
            return
        self._last_gossip_cleanup = now

        cleaned_states = 0
        cleaned_manifests = 0
        cleaned_endpoints = 0

        # 1. Clean stale peer states (older than TTL)
        cutoff = now - self.GOSSIP_STATE_TTL
        stale_state_ids = [
            node_id for node_id, state in self._gossip_peer_states.items()
            if state.get("timestamp", 0) < cutoff
        ]
        for node_id in stale_state_ids:
            del self._gossip_peer_states[node_id]
            cleaned_states += 1

        # 2. Enforce max size with LRU eviction (oldest first)
        if len(self._gossip_peer_states) > self.GOSSIP_MAX_PEER_STATES:
            # Sort by timestamp, keep newest
            sorted_states = sorted(
                self._gossip_peer_states.items(),
                key=lambda x: x[1].get("timestamp", 0),
                reverse=True,
            )
            # Keep only max entries
            self._gossip_peer_states = dict(sorted_states[:self.GOSSIP_MAX_PEER_STATES])
            cleaned_states += len(sorted_states) - self.GOSSIP_MAX_PEER_STATES

        # 3. Clean stale manifests (no timestamp, so just enforce size)
        if len(self._gossip_peer_manifests) > self.GOSSIP_MAX_MANIFESTS:
            # Keep first N (arbitrary but bounded)
            items = list(self._gossip_peer_manifests.items())
            self._gossip_peer_manifests = dict(items[:self.GOSSIP_MAX_MANIFESTS])
            cleaned_manifests = len(items) - self.GOSSIP_MAX_MANIFESTS

        # 4. Clean stale learned endpoints
        endpoint_cutoff = now - self.GOSSIP_ENDPOINT_TTL
        stale_endpoint_ids = [
            node_id for node_id, ep in self._gossip_learned_endpoints.items()
            if ep.get("learned_at", 0) < endpoint_cutoff
        ]
        for node_id in stale_endpoint_ids:
            del self._gossip_learned_endpoints[node_id]
            cleaned_endpoints += 1

        # 5. Enforce max endpoints
        if len(self._gossip_learned_endpoints) > self.GOSSIP_MAX_ENDPOINTS:
            sorted_endpoints = sorted(
                self._gossip_learned_endpoints.items(),
                key=lambda x: x[1].get("learned_at", 0),
                reverse=True,
            )
            self._gossip_learned_endpoints = dict(sorted_endpoints[:self.GOSSIP_MAX_ENDPOINTS])
            cleaned_endpoints += len(sorted_endpoints) - self.GOSSIP_MAX_ENDPOINTS

        # Log if significant cleanup occurred
        total_cleaned = cleaned_states + cleaned_manifests + cleaned_endpoints
        if total_cleaned > 10:
            # Phase 4: Use base class logging helper
            self._log_info(
                f"Cleanup: removed {cleaned_states} stale states, "
                f"{cleaned_manifests} manifests, {cleaned_endpoints} endpoints"
            )

    async def _gossip_state_to_peers(self) -> None:
        """DECENTRALIZED: Share node state with random peers using gossip protocol.

        GOSSIP PROTOCOL: Instead of relying solely on leader to collect state,
        nodes share information with neighbors, and it propagates through the cluster.

        Benefits:
        - Faster state propagation (O(log N) instead of O(N))
        - Works without a leader
        - Resilient to network partitions (state eventually converges)
        - Reduces load on leader

        Implementation:
        1. Each node maintains local state (jobs, resources, health)
        2. Periodically send state to K random peers (fanout)
        3. Receive state from peers and update local view
        4. Include version/timestamp to handle conflicts (last-write-wins)
        """
        if aiohttp is None:
            return

        # Dec 28, 2025: Clean up stale gossip state to prevent OOM
        self._cleanup_gossip_state()

        now = time.time()

        # Rate limit: gossip every 30 seconds
        last_gossip = getattr(self, "_last_gossip_time", 0)
        if now - last_gossip < 30:
            return
        self._last_gossip_time = now

        # Prepare our state to share
        self._update_self_info()
        local_state = self._build_local_gossip_state(now)

        # Select K random peers to gossip with (fanout = 3)
        GOSSIP_FANOUT = 3
        with self.peers_lock:
            alive_peers = [
                p for p in self.peers.values()
                if p.is_alive() and not getattr(p, "retired", False)
            ]

        if not alive_peers:
            return

        import random
        peers_to_gossip = random.sample(alive_peers, min(GOSSIP_FANOUT, len(alive_peers)))

        # Import session helper
        try:
            from .network import get_client_session
        except ImportError:
            # Fallback for legacy imports
            get_client_session = None

        # Send gossip to selected peers
        timeout = ClientTimeout(total=5)

        for peer in peers_to_gossip:
            await self._send_gossip_to_peer(peer, local_state, timeout, get_client_session)

    def _build_local_gossip_state(self, now: float) -> dict[str, Any]:
        """Build local state dict to share via gossip."""
        local_state = {
            "node_id": self.node_id,
            "timestamp": now,
            "version": int(now * 1000),  # Millisecond version for conflict resolution
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

        # DISTRIBUTED TRAINING COORDINATION: Include active training configs
        if hasattr(self, "_get_local_active_training_configs"):
            local_state["active_training_configs"] = self._get_local_active_training_configs()

        # DISTRIBUTED ELO: Include ELO summary for cluster-wide visibility
        if hasattr(self, "_get_local_elo_summary"):
            local_state["elo_summary"] = self._get_local_elo_summary()

        # GOSSIP-BASED LEADER HINTS: Share leader preference for faster elections
        if hasattr(self, "_get_leader_hint"):
            local_state["leader_hint"] = self._get_leader_hint()

        # PEER REPUTATION: Share peer reliability scores
        if hasattr(self, "_get_peer_reputation_summary"):
            local_state["peer_reputation"] = self._get_peer_reputation_summary()

        # DISTRIBUTED TOURNAMENT: Share tournament proposals and active tournaments
        if hasattr(self, "_get_tournament_gossip_state"):
            local_state["tournament"] = self._get_tournament_gossip_state()

        # Include manifest summary if available
        local_manifest = getattr(self, "local_data_manifest", None)
        if local_manifest:
            local_state["manifest_summary"] = {
                "total_files": getattr(local_manifest, "total_files", 0),
                "selfplay_games": getattr(local_manifest, "selfplay_games", 0),
                "collected_at": getattr(local_manifest, "collected_at", 0),
            }

        return local_state

    async def _send_gossip_to_peer(
        self,
        peer: Any,
        local_state: dict[str, Any],
        timeout: Any,
        get_client_session: Any,
    ) -> None:
        """Send gossip message to a single peer."""
        try:
            # Build gossip payload
            gossip_payload = {
                "sender": self.node_id,
                "sender_state": local_state,
                "known_states": self._get_gossip_known_states(),
                # Phase 28: Peer-of-peer discovery - share peer endpoints
                "peer_endpoints": self._get_peer_endpoints_for_gossip(),
                # Phase 29: Cluster epoch for split-brain resolution
                "cluster_epoch": self._cluster_epoch,
            }

            # GOSSIP COMPRESSION: Compress payload with gzip to reduce network transfer
            json_bytes = json.dumps(gossip_payload).encode("utf-8")
            original_size = len(json_bytes)
            compressed_bytes = gzip.compress(json_bytes, compresslevel=6)
            compressed_size = len(compressed_bytes)

            # Track compression metrics (method now in this class)
            self._record_gossip_compression(original_size, compressed_size)

            start_time = time.time()

            # Try each URL for the peer
            if get_client_session:
                async with get_client_session(timeout) as session:
                    await self._try_gossip_urls(
                        session, peer, compressed_bytes, start_time
                    )
            else:
                # Fallback without session helper
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await self._try_gossip_urls(
                        session, peer, compressed_bytes, start_time
                    )

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, AttributeError):
            pass

    async def _try_gossip_urls(
        self,
        session: Any,
        peer: Any,
        compressed_bytes: bytes,
        start_time: float,
    ) -> None:
        """Try gossip to peer via multiple URLs."""
        for url in self._urls_for_peer(peer, "/gossip"):
            try:
                headers = self._auth_headers()
                headers["Content-Encoding"] = "gzip"
                headers["Content-Type"] = "application/json"

                async with session.post(url, data=compressed_bytes, headers=headers) as resp:
                    if resp.status == 200:
                        # Process response (peer shares their state back)
                        response_data = await self._read_gossip_response(resp)
                        self._process_gossip_response(response_data)

                        # Record metrics (methods now in this class)
                        latency_ms = (time.time() - start_time) * 1000
                        self._record_gossip_metrics("sent", peer.node_id)
                        self._record_gossip_metrics("latency", peer.node_id, latency_ms)
                        break

            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, AttributeError):
                continue

    async def _read_gossip_response(self, resp: Any) -> dict:
        """Read and decompress gossip response."""
        content_encoding = resp.headers.get("Content-Encoding", "")
        if content_encoding == "gzip":
            response_bytes = await resp.read()
            decompressed = gzip.decompress(response_bytes)
            return json.loads(decompressed.decode("utf-8"))
        else:
            return await resp.json()

    def _get_gossip_known_states(self) -> dict[str, dict]:
        """Get known states about other nodes to propagate via gossip."""
        known = {}
        gossip_states = getattr(self, "_gossip_peer_states", {})
        # Only share recent states (last 5 minutes)
        cutoff = time.time() - 300
        for node_id, state in gossip_states.items():
            if state.get("timestamp", 0) > cutoff:
                known[node_id] = state
        return known

    def _get_peer_endpoints_for_gossip(self) -> list[dict[str, Any]]:
        """Phase 28: Get peer endpoints to share via gossip for peer-of-peer discovery.

        Returns a list of alive peer endpoints with connection info.
        This enables nodes to discover peers they can't reach directly.
        """
        endpoints = []
        with self.peers_lock:
            # Get alive, non-retired peers
            alive_peers = [
                p for p in self.peers.values()
                if p.node_id != self.node_id and p.is_alive() and not getattr(p, "retired", False)
            ]

        # Limit to top N peers to avoid payload bloat
        for peer in alive_peers[:GOSSIP_MAX_PEER_ENDPOINTS]:
            endpoint = {
                "node_id": peer.node_id,
                "host": str(getattr(peer, "host", "") or ""),
                "port": int(getattr(peer, "port", DEFAULT_PORT) or DEFAULT_PORT),
                "tailscale_ip": str(getattr(peer, "tailscale_ip", "") or ""),
                "is_alive": True,
                "last_heartbeat": float(getattr(peer, "last_heartbeat", 0) or 0),
            }
            endpoints.append(endpoint)

        return endpoints

    def _process_gossip_response(self, response: dict) -> None:
        """Process gossip response from a peer, updating our view of the cluster."""
        if not response:
            return

        # Initialize gossip state storage if needed
        self._init_gossip_protocol()

        # Process sender's state
        sender_state = response.get("sender_state", {})
        if sender_state:
            self._process_sender_state(sender_state)

        # Process known states (propagation)
        known_states = response.get("known_states", {})
        self._process_known_states(known_states)

        # Process manifest info for P2P sync
        peer_manifests = response.get("peer_manifests", {})
        self._process_peer_manifests(peer_manifests)

        # Process tournament gossip for distributed scheduling
        self._process_tournament_states(known_states)

        # Phase 28: Process peer endpoints for peer-of-peer discovery
        peer_endpoints = response.get("peer_endpoints") or []
        if peer_endpoints:
            self._process_gossip_peer_endpoints(peer_endpoints)

        # Phase 29: Process cluster epoch for split-brain resolution
        incoming_epoch = response.get("cluster_epoch")
        if incoming_epoch is not None:
            self._handle_incoming_cluster_epoch(incoming_epoch, response)

    def _process_sender_state(self, sender_state: dict) -> None:
        """Process the sender's state from a gossip response."""
        sender_id = sender_state.get("node_id")
        if not sender_id or sender_id == self.node_id:
            return

        existing = self._gossip_peer_states.get(sender_id, {})
        # Last-write-wins conflict resolution
        if sender_state.get("version", 0) > existing.get("version", 0):
            self._gossip_peer_states[sender_id] = sender_state

            # Update leader info if sender claims to know a leader
            if sender_state.get("leader_id") and not self.leader_id:
                claimed_leader = sender_state.get("leader_id")
                lease_expires = sender_state.get("leader_lease_expires", 0)
                if lease_expires > time.time():
                    self.leader_id = claimed_leader
                    self.last_leader_seen = time.time()

    def _process_known_states(self, known_states: dict[str, dict]) -> None:
        """Process known states from gossip propagation."""
        for node_id, state in known_states.items():
            if node_id == self.node_id:
                continue
            existing = self._gossip_peer_states.get(node_id, {})
            if state.get("version", 0) > existing.get("version", 0):
                self._gossip_peer_states[node_id] = state

    def _process_peer_manifests(self, peer_manifests: dict) -> None:
        """Process peer manifest info for P2P sync."""
        try:
            from ..app.coordination.sync_planner import NodeDataManifest
        except ImportError:
            # Try alternative import path
            try:
                from app.coordination.sync_planner import NodeDataManifest
            except ImportError:
                return  # Skip if manifest class not available

        for node_id, manifest_data in peer_manifests.items():
            if node_id != self.node_id:
                with contextlib.suppress(Exception):
                    self._gossip_peer_manifests[node_id] = NodeDataManifest.from_dict(manifest_data)

    def _process_tournament_states(self, known_states: dict[str, dict]) -> None:
        """Process tournament gossip for distributed scheduling."""
        if not hasattr(self, "_process_tournament_gossip"):
            return

        for node_id, state in known_states.items():
            if node_id == self.node_id:
                continue
            tournament_state = state.get("tournament")
            if tournament_state:
                with contextlib.suppress(Exception):
                    self._process_tournament_gossip(node_id, tournament_state)

        # Check for tournament consensus after processing gossip
        if hasattr(self, "_check_tournament_consensus"):
            with contextlib.suppress(Exception):
                self._check_tournament_consensus()

    def _process_gossip_peer_endpoints(self, peer_endpoints: list[dict]) -> None:
        """Phase 28: Process peer endpoints learned via gossip.

        Enables discovery of peers we can't reach directly through intermediaries.
        """
        for endpoint in peer_endpoints:
            node_id = endpoint.get("node_id")
            if not node_id or node_id == self.node_id:
                continue

            # Store in gossip-learned endpoints for later connection attempts
            host = endpoint.get("tailscale_ip") or endpoint.get("host")
            port = endpoint.get("port", DEFAULT_PORT)

            if host and port:
                self._gossip_learned_endpoints[node_id] = {
                    "host": host,
                    "port": port,
                    "tailscale_ip": endpoint.get("tailscale_ip", ""),
                    "last_heartbeat": endpoint.get("last_heartbeat", 0),
                    "learned_at": time.time(),
                }

                # If this is an unknown peer, try to connect
                if node_id not in self.peers:
                    # Queue for async connection attempt
                    asyncio.create_task(self._try_connect_gossip_peer(node_id, host, port))

    async def _try_connect_gossip_peer(self, node_id: str, host: str, port: int) -> None:
        """Phase 28: Attempt to connect to a peer learned via gossip."""
        try:
            # Check if already connected
            if node_id in self.peers and self.peers[node_id].is_alive():
                return

            # Phase 4: Use base class logging helper
            self._log_info(f"Attempting connection to gossip-learned peer: {node_id} at {host}:{port}")

            # Try to send heartbeat
            info = await self._send_heartbeat_to_peer(host, port)
            if info:
                with self.peers_lock:
                    self.peers[info.node_id] = info
                self._log_info(f"Successfully connected to gossip-learned peer: {info.node_id}")

                # Save to cache for future restarts
                self._save_peer_to_cache(
                    info.node_id, host, port,
                    str(getattr(info, "tailscale_ip", "") or "")
                )
        except Exception as e:
            if self.verbose:
                self._log_debug(f"Failed to connect to gossip-learned peer {node_id}: {e}")

    def _handle_incoming_cluster_epoch(self, incoming_epoch: Any, response: dict) -> None:
        """Phase 29: Handle incoming cluster epoch for split-brain resolution."""
        try:
            epoch = int(incoming_epoch)
        except (ValueError, TypeError):
            return

        if epoch > self._cluster_epoch:
            # Accept higher epoch - this cluster partition is more authoritative
            # Phase 4: Use base class logging helper
            self._log_info(f"Adopting higher cluster epoch: {epoch} (was {self._cluster_epoch})")
            self._cluster_epoch = epoch
            self._save_cluster_epoch()

            # If response includes a leader, adopt it
            sender_state = response.get("sender_state", {})
            incoming_leader = sender_state.get("leader_id")
            if incoming_leader and incoming_leader != self.node_id:
                # Import NodeRole for comparison
                try:
                    from .models import NodeRole
                except ImportError:
                    NodeRole = None

                if NodeRole and self.role == NodeRole.LEADER:
                    self._log_info(f"Stepping down: higher epoch cluster has leader {incoming_leader}")
                    self.role = NodeRole.FOLLOWER
                self.leader_id = incoming_leader

    async def _gossip_anti_entropy_repair(self) -> None:
        """DECENTRALIZED: Periodic full state reconciliation with random peer.

        ANTI-ENTROPY REPAIR: Gossip protocols can miss updates due to:
        - Network partitions
        - Message loss
        - Node restarts

        Solution: Periodically do full state exchange with a random peer to
        ensure eventual consistency. This catches any missed updates.

        Frequency: Every 5 minutes with a random healthy peer
        """
        if aiohttp is None:
            return

        now = time.time()

        # Rate limit: anti-entropy every 5 minutes
        last_repair = getattr(self, "_last_anti_entropy_repair", 0)
        if now - last_repair < 300:
            return
        self._last_anti_entropy_repair = now

        # Select a random healthy peer for full state exchange
        with self.peers_lock:
            alive_peers = [
                p for p in self.peers.values()
                if p.is_alive() and not getattr(p, "retired", False)
            ]

        if not alive_peers:
            return

        import random
        peer = random.choice(alive_peers)

        # Prepare full state dump (not just recent states)
        full_state = self._build_anti_entropy_state(now)

        # Send anti-entropy request
        await self._send_anti_entropy_request(peer, full_state, now)

    def _build_anti_entropy_state(self, now: float) -> dict[str, Any]:
        """Build full state for anti-entropy repair."""
        full_state: dict[str, Any] = {
            "anti_entropy": True,  # Flag for full state exchange
            "sender": self.node_id,
            "timestamp": now,
            "all_known_states": {},
        }

        # Include all known peer states (not just recent)
        gossip_states = getattr(self, "_gossip_peer_states", {})
        for node_id, state in gossip_states.items():
            full_state["all_known_states"][node_id] = state

        # Include our own state
        self._update_self_info()
        full_state["all_known_states"][self.node_id] = {
            "node_id": self.node_id,
            "timestamp": now,
            "version": int(now * 1000),
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "leader_id": self.leader_id,
            "selfplay_jobs": getattr(self.self_info, "selfplay_jobs", 0),
            "training_jobs": getattr(self.self_info, "training_jobs", 0),
        }

        return full_state

    async def _send_anti_entropy_request(
        self,
        peer: Any,
        full_state: dict[str, Any],
        now: float,
    ) -> None:
        """Send anti-entropy repair request to a peer."""
        start_time = time.time()
        timeout = ClientTimeout(total=10)  # Longer timeout for full exchange

        try:
            # Import session helper
            try:
                from .network import get_client_session
            except ImportError:
                get_client_session = None

            if get_client_session:
                async with get_client_session(timeout) as session:
                    await self._try_anti_entropy_urls(session, peer, full_state, start_time, now)
            else:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await self._try_anti_entropy_urls(session, peer, full_state, start_time, now)

        except (AttributeError, KeyError, ValueError, TypeError):
            pass  # Silent failure, will retry next cycle

    async def _try_anti_entropy_urls(
        self,
        session: Any,
        peer: Any,
        full_state: dict[str, Any],
        start_time: float,
        now: float,
    ) -> None:
        """Try anti-entropy repair via multiple URLs."""
        for url in self._urls_for_peer(peer, "/gossip/anti-entropy"):
            try:
                async with session.post(url, json=full_state, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        latency = (time.time() - start_time) * 1000
                        self._record_gossip_metrics("latency", peer.node_id, latency)

                        # Process peer's full state
                        updates = self._process_anti_entropy_response(response_data, now)

                        if updates > 0:
                            self._record_gossip_metrics("anti_entropy")
                            self._log_debug(f"Anti-entropy repair: {updates} state updates from {peer.node_id}")

                        return

            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, KeyError, ValueError):
                continue

    def _process_anti_entropy_response(self, response_data: dict, now: float) -> int:
        """Process anti-entropy response and return number of updates."""
        peer_states = response_data.get("all_known_states", {})
        updates = 0

        for node_id, state in peer_states.items():
            if node_id == self.node_id:
                continue
            existing = self._gossip_peer_states.get(node_id, {})
            if state.get("version", 0) > existing.get("version", 0):
                self._gossip_peer_states[node_id] = state
                updates += 1
                self._record_gossip_metrics("update", node_id)

        # Check for stale states we have that peer doesn't know
        our_nodes = set(self._gossip_peer_states.keys())
        peer_nodes = set(peer_states.keys())
        stale_candidates = our_nodes - peer_nodes - {self.node_id}

        for stale_node in stale_candidates:
            stale_state = self._gossip_peer_states.get(stale_node, {})
            # If state is older than 10 minutes and peer doesn't know it,
            # the node might be offline - mark as stale
            if stale_state.get("timestamp", 0) < now - 600:
                self._record_gossip_metrics("stale", stale_node)

        return updates

    def get_gossip_peer_states(self) -> dict[str, dict]:
        """Get a copy of current gossip peer states.

        Public API for accessing gossip-learned peer states.
        """
        return dict(self._gossip_peer_states)

    def get_gossip_learned_endpoints(self) -> dict[str, dict]:
        """Get a copy of gossip-learned peer endpoints.

        Public API for accessing endpoints discovered via peer-of-peer gossip.
        """
        return dict(self._gossip_learned_endpoints)

    # =========================================================================
    # Gossip Metrics (Phase 4: Merged from GossipMetricsMixin - Dec 28, 2025)
    # =========================================================================

    def _record_gossip_metrics(
        self,
        event: str,
        peer_id: str | None = None,
        latency_ms: float = 0,
    ) -> None:
        """Record gossip protocol metrics for monitoring.

        GOSSIP METRICS: Track propagation efficiency and protocol health.
        - message_sent: Gossip messages sent
        - message_received: Gossip messages received
        - state_updates: Number of state updates from gossip
        - propagation_delay_ms: Average latency for gossip messages
        - anti_entropy_repairs: Full state reconciliations triggered

        Args:
            event: Event type (sent, received, update, anti_entropy, stale, latency)
            peer_id: Optional peer ID for context
            latency_ms: Latency in milliseconds (for latency events)
        """
        # Ensure metrics state exists
        self._ensure_state_attr("_gossip_metrics", {
            "message_sent": 0,
            "message_received": 0,
            "state_updates": 0,
            "propagation_delay_ms": [],
            "anti_entropy_repairs": 0,
            "stale_states_detected": 0,
            "last_reset": time.time(),
        })
        metrics = self._gossip_metrics

        if event == "sent":
            metrics["message_sent"] += 1
        elif event == "received":
            metrics["message_received"] += 1
        elif event == "update":
            metrics["state_updates"] += 1
        elif event == "anti_entropy":
            metrics["anti_entropy_repairs"] += 1
        elif event == "stale":
            metrics["stale_states_detected"] += 1
        elif event == "latency":
            # Keep last 100 latency measurements
            metrics["propagation_delay_ms"].append(latency_ms)
            if len(metrics["propagation_delay_ms"]) > 100:
                metrics["propagation_delay_ms"] = metrics["propagation_delay_ms"][-100:]

        # Reset metrics every hour
        if time.time() - metrics.get("last_reset", 0) > 3600:
            self._reset_gossip_metrics_hourly()

    def _reset_gossip_metrics_hourly(self) -> dict[str, Any]:
        """Reset gossip metrics and return old values.

        Called automatically after 1 hour. Returns old metrics for logging.
        """
        self._ensure_state_attr("_gossip_metrics", {})
        old_metrics = self._gossip_metrics.copy()

        self._gossip_metrics = {
            "message_sent": 0,
            "message_received": 0,
            "state_updates": 0,
            "propagation_delay_ms": [],
            "anti_entropy_repairs": 0,
            "stale_states_detected": 0,
            "last_reset": time.time(),
        }

        # Log metrics before reset using base class helper
        delays = old_metrics.get("propagation_delay_ms", [])
        avg_latency = sum(delays) / max(1, len(delays)) if delays else 0

        self._log_debug(
            f"Hourly: sent={old_metrics.get('message_sent', 0)} "
            f"recv={old_metrics.get('message_received', 0)} "
            f"updates={old_metrics.get('state_updates', 0)} "
            f"repairs={old_metrics.get('anti_entropy_repairs', 0)} "
            f"stale={old_metrics.get('stale_states_detected', 0)} "
            f"avg_latency={avg_latency:.1f}ms"
        )

        return old_metrics

    def _record_gossip_compression(
        self,
        original_size: int,
        compressed_size: int,
    ) -> None:
        """Record gossip compression metrics.

        COMPRESSION METRICS: Track how effective compression is for gossip messages.
        Typical JSON gossip payloads compress 60-80% with gzip level 6.

        Args:
            original_size: Original message size in bytes
            compressed_size: Compressed message size in bytes
        """
        self._ensure_state_attr("_gossip_compression_stats", {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "messages_compressed": 0,
        })
        stats = self._gossip_compression_stats
        stats["total_original_bytes"] += original_size
        stats["total_compressed_bytes"] += compressed_size
        stats["messages_compressed"] += 1

    def _get_gossip_metrics_summary(self) -> dict[str, Any]:
        """Get summary of gossip metrics for /status endpoint.

        Returns:
            Dict with message counts, latency, and compression stats
        """
        self._ensure_state_attr("_gossip_metrics", {})
        self._ensure_state_attr("_gossip_compression_stats", {})
        metrics = self._gossip_metrics
        delays = metrics.get("propagation_delay_ms", [])

        # Include compression stats
        compression = self._gossip_compression_stats
        original = compression.get("total_original_bytes", 0)
        compressed = compression.get("total_compressed_bytes", 0)
        compression_ratio = 1.0 - (compressed / original) if original > 0 else 0

        return {
            "message_sent": metrics.get("message_sent", 0),
            "message_received": metrics.get("message_received", 0),
            "state_updates": metrics.get("state_updates", 0),
            "anti_entropy_repairs": metrics.get("anti_entropy_repairs", 0),
            "stale_states_detected": metrics.get("stale_states_detected", 0),
            "avg_latency_ms": sum(delays) / max(1, len(delays)) if delays else 0,
            "compression_ratio": round(compression_ratio, 3),
            "bytes_saved_kb": round((original - compressed) / 1024, 2),
            "messages_compressed": compression.get("messages_compressed", 0),
        }

    def _get_gossip_health_status(self) -> dict[str, Any]:
        """Get gossip protocol health status.

        Returns health indicators for monitoring:
        - is_healthy: True if gossip is functioning well
        - warnings: List of any warning conditions
        """
        summary = self._get_gossip_metrics_summary()
        warnings = []

        # Check for high latency
        avg_latency = summary.get("avg_latency_ms", 0)
        if avg_latency > 1000:
            warnings.append(f"High gossip latency: {avg_latency:.0f}ms")

        # Check for low message rate (stale cluster)
        sent = summary.get("message_sent", 0)
        received = summary.get("message_received", 0)
        if sent + received < 10:
            warnings.append("Low gossip activity")

        # Check for high stale rate
        stale = summary.get("stale_states_detected", 0)
        updates = summary.get("state_updates", 0)
        if updates > 0 and stale / updates > 0.5:
            warnings.append(f"High stale rate: {stale}/{updates}")

        return {
            "is_healthy": len(warnings) == 0,
            "warnings": warnings,
            "metrics": summary,
        }


# Standalone utility function (from GossipMetricsMixin)
def calculate_compression_ratio(original: int, compressed: int) -> float:
    """Calculate compression ratio.

    Args:
        original: Original size in bytes
        compressed: Compressed size in bytes

    Returns:
        Ratio of bytes saved (0.0 to 1.0)
    """
    if original <= 0:
        return 0.0
    return 1.0 - (compressed / original)

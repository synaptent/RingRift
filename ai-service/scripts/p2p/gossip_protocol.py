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
import os
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


# Dec 28, 2025: Phase 6 - Gossip health tracking threshold
# Dec 30, 2025: Now uses centralized GossipDefaults for configurability
try:
    from app.config.coordination_defaults import GossipDefaults
    GOSSIP_FAILURE_SUSPECT_THRESHOLD = GossipDefaults.FAILURE_THRESHOLD
except ImportError:
    GOSSIP_FAILURE_SUSPECT_THRESHOLD = 5  # Fallback

# Jan 2026: Use centralized timeout from LoopTimeouts
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    _GOSSIP_LOCK_TIMEOUT = LoopTimeouts.GOSSIP_LOCK
except ImportError:
    _GOSSIP_LOCK_TIMEOUT = 5.0  # Fallback


class GossipHealthTracker:
    """Tracks gossip protocol health per peer.

    December 2025: Part of Phase 6 cluster availability improvements.
    Tracks gossip failures per peer and emits NODE_SUSPECT when a peer
    has too many consecutive gossip failures.

    Jan 3, 2026: Added exponential backoff for failed peers.
    When a peer fails gossip, we wait progressively longer before retrying:
    - 1st failure: 1s backoff
    - 2nd failure: 2s backoff
    - 3rd failure: 4s backoff
    - 4th failure: 8s backoff
    - 5th+ failure: 16s backoff (max)

    Features:
    - Per-peer failure counting
    - Automatic reset on success
    - Threshold-based suspect detection
    - Last success timestamp tracking for staleness
    - Exponential backoff for failed peers (Jan 3, 2026)
    """

    # Jan 3, 2026: Backoff configuration
    BACKOFF_BASE_SECONDS: float = 1.0
    BACKOFF_MULTIPLIER: float = 2.0
    BACKOFF_MAX_SECONDS: float = 16.0

    def __init__(self, failure_threshold: int = GOSSIP_FAILURE_SUSPECT_THRESHOLD):
        """Initialize gossip health tracker.

        Args:
            failure_threshold: Number of consecutive failures before emitting suspect
        """
        self._failure_counts: dict[str, int] = {}
        self._last_success: dict[str, float] = {}
        self._last_failure: dict[str, float] = {}  # Jan 3, 2026: Track last failure time
        self._failure_threshold = failure_threshold
        self._suspect_emitted: set[str] = set()  # Track which peers have been marked suspect

    def record_gossip_failure(self, peer_id: str) -> tuple[bool, int]:
        """Record a gossip failure for a peer.

        Args:
            peer_id: The peer that failed to respond to gossip

        Returns:
            Tuple of (should_emit_suspect, failure_count)
            - should_emit_suspect: True if this failure crosses the threshold
            - failure_count: Current consecutive failure count
        """
        self._failure_counts[peer_id] = self._failure_counts.get(peer_id, 0) + 1
        self._last_failure[peer_id] = time.time()  # Jan 3, 2026: Track failure time
        count = self._failure_counts[peer_id]

        # Check if we should emit suspect (only once per failure streak)
        should_emit = (
            count >= self._failure_threshold
            and peer_id not in self._suspect_emitted
        )

        if should_emit:
            self._suspect_emitted.add(peer_id)

        return should_emit, count

    def get_backoff_seconds(self, peer_id: str) -> float:
        """Calculate exponential backoff delay for a peer.

        Jan 3, 2026: Returns the number of seconds to wait before retrying
        gossip to this peer based on consecutive failure count.

        Args:
            peer_id: The peer to calculate backoff for

        Returns:
            Backoff delay in seconds (0 if no failures)
        """
        failure_count = self._failure_counts.get(peer_id, 0)
        if failure_count == 0:
            return 0.0

        # Exponential backoff: base * multiplier^(failures-1), capped at max
        backoff = self.BACKOFF_BASE_SECONDS * (
            self.BACKOFF_MULTIPLIER ** (failure_count - 1)
        )
        return min(backoff, self.BACKOFF_MAX_SECONDS)

    def should_skip_peer(self, peer_id: str) -> bool:
        """Check if a peer should be skipped due to backoff.

        Jan 3, 2026: Returns True if the peer is in backoff period.

        Args:
            peer_id: The peer to check

        Returns:
            True if peer should be skipped, False if OK to gossip
        """
        last_failure = self._last_failure.get(peer_id)
        if last_failure is None:
            return False  # No failures, OK to gossip

        backoff = self.get_backoff_seconds(peer_id)
        if backoff == 0:
            return False  # No backoff needed

        elapsed = time.time() - last_failure
        return elapsed < backoff

    def record_gossip_success(self, peer_id: str) -> bool:
        """Record a successful gossip exchange with a peer.

        Args:
            peer_id: The peer that responded successfully

        Returns:
            True if peer was previously suspected (recovered), False otherwise
        """
        was_suspected = peer_id in self._suspect_emitted

        # Reset failure count and remove from suspect set
        self._failure_counts[peer_id] = 0
        self._last_success[peer_id] = time.time()
        self._suspect_emitted.discard(peer_id)

        return was_suspected

    def get_failure_count(self, peer_id: str) -> int:
        """Get the current consecutive failure count for a peer."""
        return self._failure_counts.get(peer_id, 0)

    def get_last_success(self, peer_id: str) -> float | None:
        """Get the timestamp of the last successful gossip with a peer."""
        return self._last_success.get(peer_id)

    def is_suspected(self, peer_id: str) -> bool:
        """Check if a peer is currently marked as suspect due to gossip failures."""
        return peer_id in self._suspect_emitted

    def get_suspected_peers(self) -> set[str]:
        """Get the set of peers currently marked as suspect."""
        return self._suspect_emitted.copy()

    def get_stats(self) -> dict[str, int | float | list[str]]:
        """Get tracker statistics for monitoring."""
        now = time.time()
        stale_peers = [
            peer_id for peer_id, last_seen in self._last_success.items()
            if now - last_seen > 300  # 5 minutes
        ]
        return {
            "total_tracked_peers": len(self._failure_counts),
            "suspected_peers": len(self._suspect_emitted),
            "suspected_peer_ids": list(self._suspect_emitted),
            "stale_peers": len(stale_peers),
            "failure_threshold": self._failure_threshold,
        }

    def cleanup_stale_peers(self, max_age_seconds: float = 3600.0) -> int:
        """Remove stale peer tracking data.

        Args:
            max_age_seconds: Remove peers not seen in this many seconds

        Returns:
            Number of peers cleaned up
        """
        now = time.time()
        cutoff = now - max_age_seconds

        # Find stale peers
        stale_peers = [
            peer_id for peer_id, last_seen in self._last_success.items()
            if last_seen < cutoff
        ]

        # Also include peers with high failure counts but no success
        for peer_id in list(self._failure_counts.keys()):
            if peer_id not in self._last_success:
                # Peer has never succeeded, check if it's been tracked too long
                # We use a simple heuristic: high failure count = stale
                if self._failure_counts[peer_id] > self._failure_threshold * 10:
                    if peer_id not in stale_peers:
                        stale_peers.append(peer_id)

        # Clean up
        for peer_id in stale_peers:
            self._failure_counts.pop(peer_id, None)
            self._last_success.pop(peer_id, None)
            self._last_failure.pop(peer_id, None)  # Jan 3, 2026: Clean up backoff tracking
            self._suspect_emitted.discard(peer_id)

        return len(stale_peers)


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

    # Dec 30, 2025: Configurable gossip parameters via environment variables
    # These can be tuned per cluster without code changes
    GOSSIP_FANOUT_LEADER = int(os.environ.get("RINGRIFT_GOSSIP_FANOUT_LEADER", "8"))
    GOSSIP_FANOUT_FOLLOWER = int(os.environ.get("RINGRIFT_GOSSIP_FANOUT_FOLLOWER", "5"))
    GOSSIP_INTERVAL_SECONDS = float(os.environ.get("RINGRIFT_GOSSIP_INTERVAL", "30"))
    ANTI_ENTROPY_INTERVAL_SECONDS = float(
        os.environ.get("RINGRIFT_ANTI_ENTROPY_INTERVAL", "120")
    )

    # Dec 30, 2025: Message size limits for network stability
    # RINGRIFT_GOSSIP_MAX_SIZE is in bytes (default 1MB)
    GOSSIP_MAX_MESSAGE_SIZE_BYTES = int(
        os.environ.get("RINGRIFT_GOSSIP_MAX_SIZE", str(1 * 1024 * 1024))
    )
    GOSSIP_MESSAGE_SIZE_WARNING_BYTES = GOSSIP_MAX_MESSAGE_SIZE_BYTES // 2

    def _init_gossip_protocol(self) -> None:
        """Initialize gossip protocol state and metrics.

        Call this in __init__ to set up gossip storage.
        Uses P2PMixinBase._ensure_multiple_state_attrs() for cleaner initialization.

        Phase 4 consolidation: Now includes GossipMetricsMixin state (previously separate).
        Phase 6 (Dec 28, 2025): Added GossipHealthTracker for peer health integration.
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

        # Dec 28, 2025: Phase 6 - Gossip health tracker for peer status integration
        if not hasattr(self, "_gossip_health_tracker"):
            self._gossip_health_tracker = GossipHealthTracker()

        # Dec 30, 2025: Async lock for gossip state mutations to prevent race conditions
        # Uses TimeoutAsyncLockWrapper to prevent deadlocks on contention
        # Jan 2026: Use centralized _GOSSIP_LOCK_TIMEOUT from LoopTimeouts
        if not hasattr(self, "_gossip_state_lock"):
            from .network import TimeoutAsyncLockWrapper
            self._gossip_state_lock = TimeoutAsyncLockWrapper(timeout=_GOSSIP_LOCK_TIMEOUT)

        # Jan 3, 2026: Add threading lock for sync state modifications
        # The async lock above is for async cleanup, this sync lock protects
        # state modifications from _process_gossip_response and HTTP handlers
        if not hasattr(self, "_gossip_state_sync_lock"):
            import threading
            self._gossip_state_sync_lock = threading.RLock()

        # Dec 29, 2025: Restore persisted gossip state on startup
        # This allows faster cluster state recovery after P2P restarts
        self._restore_gossip_state_on_startup()

    async def _cleanup_gossip_state(self) -> None:
        """Dec 28, 2025: Clean up stale gossip state to prevent memory growth.

        Called periodically to:
        1. Remove entries older than TTL
        2. Enforce max size limits with LRU eviction

        Dec 30, 2025: Made async and added locking to prevent race conditions
        when multiple coroutines access _gossip_peer_states concurrently.
        """
        now = time.time()

        # Rate limit cleanup to every 5 minutes
        if now - getattr(self, "_last_gossip_cleanup", 0) < 300:
            return
        self._last_gossip_cleanup = now

        # Acquire locks with timeout to prevent deadlocks
        # Graceful fallback if lock unavailable (e.g., during initialization)
        # Jan 2026: Use centralized _GOSSIP_LOCK_TIMEOUT from LoopTimeouts
        async_lock = getattr(self, "_gossip_state_lock", None)
        sync_lock = getattr(self, "_gossip_state_sync_lock", None)

        # Acquire async lock first (for other async callers)
        async_acquired = False
        if async_lock is not None:
            async_acquired = await async_lock.acquire(timeout=_GOSSIP_LOCK_TIMEOUT)
            if not async_acquired:
                self._log_warning("Gossip state lock acquisition timed out during cleanup")
                return

        # Jan 3, 2026: Also acquire sync lock to prevent race with sync state modifiers
        sync_acquired = False
        if sync_lock is not None:
            sync_acquired = sync_lock.acquire(blocking=True, timeout=_GOSSIP_LOCK_TIMEOUT)
            if not sync_acquired:
                if async_lock is not None and async_acquired:
                    async_lock.release()
                self._log_warning("Gossip sync lock acquisition timed out during cleanup")
                return

        try:
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

            # 6. Dec 28, 2025 (Phase 6): Clean up stale gossip health tracking data
            cleaned_health = 0
            if hasattr(self, "_gossip_health_tracker"):
                cleaned_health = self._gossip_health_tracker.cleanup_stale_peers(
                    max_age_seconds=self.GOSSIP_STATE_TTL
                )

            # Log if significant cleanup occurred
            total_cleaned = cleaned_states + cleaned_manifests + cleaned_endpoints + cleaned_health
            if total_cleaned > 10:
                # Phase 4: Use base class logging helper
                self._log_info(
                    f"Cleanup: removed {cleaned_states} stale states, "
                    f"{cleaned_manifests} manifests, {cleaned_endpoints} endpoints, "
                    f"{cleaned_health} health tracking entries"
                )
        finally:
            # Jan 3, 2026: Release both locks in reverse order
            if sync_lock is not None and sync_acquired:
                sync_lock.release()
            if async_lock is not None and async_acquired:
                async_lock.release()

    # =========================================================================
    # Endpoint Validation (Dec 30, 2025)
    # Proactive validation of stale peer endpoints to prevent partition isolation
    # =========================================================================

    async def _probe_endpoint_with_retry(self, peer: "NodeInfo") -> str | None:
        """Probe peer endpoint with retry logic for transient failures.

        December 30, 2025: Added retry wrapper to handle transient network failures
        that could cause stale IP addresses to persist. Uses exponential backoff.

        Args:
            peer: The NodeInfo to probe

        Returns:
            The first working IP address, or None if all retries exhausted
        """
        try:
            from app.utils.retry import RetryConfig
        except ImportError:
            # Fallback if retry module unavailable
            return await self._probe_best_endpoint(peer)

        config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=5.0,
            exponential=True,
            jitter=0.2,
        )

        for attempt in config.attempts():
            try:
                result = await self._probe_best_endpoint(peer)
                if result:
                    return result
                # No IPs worked but no exception - don't retry
                if attempt.is_last:
                    return None
                # Log and retry
                if self.verbose:
                    self._log_debug(
                        f"Endpoint probe returned None for {peer.node_id}, "
                        f"attempt {attempt.number}/{attempt.max_attempts}"
                    )
                await attempt.wait_async()
            except (asyncio.TimeoutError, OSError) as e:
                # Transient network error - retry
                if attempt.is_last:
                    self._log_warning(
                        f"All endpoint probe attempts failed for {peer.node_id}: {e}"
                    )
                    return None
                if self.verbose:
                    self._log_debug(
                        f"Endpoint probe failed for {peer.node_id} (attempt "
                        f"{attempt.number}/{attempt.max_attempts}): {e}"
                    )
                await attempt.wait_async()
            except (aiohttp.ClientError, json.JSONDecodeError, ValueError, AttributeError, KeyError) as e:
                # Jan 3, 2026: Narrowed from broad Exception to specific types
                # Non-transient error - don't retry
                self._log_warning(f"Unexpected error probing {peer.node_id}: {type(e).__name__}: {e}")
                return None

        return None

    async def _validate_stale_endpoints(self) -> int:
        """Validate and refresh stale peer endpoints.

        December 30, 2025: Added for P2P partition recovery.

        When a peer's endpoint hasn't been validated for endpoint_ttl_seconds,
        we probe alternate IPs (Tailscale, public IP) to check if the primary
        IP is stale. If we find a working alternate, update the peer's host.

        This prevents network partitions caused by:
        - Private IPs that became unreachable after network changes
        - Container IPs that changed after restart
        - VPN IPs that changed after reconnection

        Returns:
            Number of endpoints that were refreshed with better IPs
        """
        try:
            from app.config.coordination_defaults import EndpointValidationDefaults
        except ImportError:
            return 0  # Defaults not available

        if not EndpointValidationDefaults.ENABLED:
            return 0

        now = time.time()

        # Rate limit: run every VALIDATION_INTERVAL seconds
        last_validation = getattr(self, "_last_endpoint_validation", 0)
        if now - last_validation < EndpointValidationDefaults.VALIDATION_INTERVAL:
            return 0
        self._last_endpoint_validation = now

        # Find peers with stale endpoints
        with self.peers_lock:
            stale_peers = [
                peer for peer in list(self.peers.values())
                if not getattr(peer, "retired", False)
                and peer.is_endpoint_stale()
            ]

        if not stale_peers:
            return 0

        # Limit validations per cycle to avoid thundering herd
        stale_peers = stale_peers[:EndpointValidationDefaults.MAX_VALIDATIONS_PER_CYCLE]

        refreshed = 0
        for peer in stale_peers:
            try:
                # Dec 30, 2025: Use retry wrapper for transient failures
                new_host = await self._probe_endpoint_with_retry(peer)
                if new_host:
                    with self.peers_lock:
                        if peer.node_id in self.peers:
                            old_host = self.peers[peer.node_id].host
                            if new_host != old_host:
                                self.peers[peer.node_id].host = new_host
                                refreshed += 1
                                self._log_info(
                                    f"Endpoint refresh: {peer.node_id} "
                                    f"{old_host} -> {new_host}"
                                )
                            self.peers[peer.node_id].mark_endpoint_validated()
            except (asyncio.TimeoutError, OSError, aiohttp.ClientError, ValueError, AttributeError, KeyError) as e:
                # Jan 3, 2026: Narrowed from broad Exception to specific types
                if self.verbose:
                    self._log_debug(
                        f"Failed to validate endpoint for {peer.node_id}: {type(e).__name__}: {e}"
                    )

        if refreshed > 0:
            self._log_info(f"Endpoint validation: refreshed {refreshed} stale endpoints")

        return refreshed

    async def _probe_best_endpoint(self, peer: "NodeInfo") -> str | None:
        """Probe multiple IPs for a peer and return the first working one.

        Tries in order:
        1. Current host (validate it still works)
        2. Tailscale IP (from learned endpoints or reported_host if 100.x.x.x)
        3. Alternate IPs (from alternate_ips set)

        Args:
            peer: The NodeInfo to probe

        Returns:
            The first working IP address, or None if all probes fail
        """
        if aiohttp is None:
            return None

        try:
            from app.config.coordination_defaults import EndpointValidationDefaults
            timeout_seconds = EndpointValidationDefaults.PROBE_TIMEOUT
        except ImportError:
            timeout_seconds = 5.0

        # Build list of IPs to probe, in priority order
        ips_to_probe: list[str] = []

        # 1. Current host
        if peer.host:
            ips_to_probe.append(peer.host)

        # 2. Tailscale IP (100.x.x.x range)
        if peer.reported_host and peer.reported_host.startswith("100."):
            if peer.reported_host not in ips_to_probe:
                ips_to_probe.append(peer.reported_host)

        # 3. Check learned endpoints for Tailscale IP
        learned = self._gossip_learned_endpoints.get(peer.node_id, {})
        tailscale_ip = learned.get("tailscale_ip")
        if tailscale_ip and tailscale_ip not in ips_to_probe:
            ips_to_probe.append(tailscale_ip)

        # 4. Alternate IPs
        for alt_ip in getattr(peer, "alternate_ips", set()) or set():
            if alt_ip and alt_ip not in ips_to_probe:
                ips_to_probe.append(alt_ip)

        if not ips_to_probe:
            return None

        # Probe each IP until one works
        timeout = ClientTimeout(total=timeout_seconds)
        for ip in ips_to_probe:
            try:
                url = f"http://{ip}:{peer.port}/status"
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Verify it's the right node
                            if data.get("node_id") == peer.node_id:
                                return ip
            except (asyncio.TimeoutError, OSError) as e:
                # Dec 30, 2025: Log timeout/network errors at debug level
                self._log_debug(f"[EndpointProbe] {peer.node_id} IP {ip} probe failed: {type(e).__name__}")
                continue
            except (aiohttp.ClientError, json.JSONDecodeError, ValueError, AttributeError, KeyError) as e:
                # Jan 3, 2026: Narrowed from broad Exception to specific types
                self._log_debug(f"[EndpointProbe] {peer.node_id} IP {ip} unexpected error: {type(e).__name__}: {e}")
                continue

        return None

    def _on_heartbeat_success(self, peer_id: str) -> None:
        """Called when heartbeat to a peer succeeds.

        Updates endpoint validation timestamp to prevent unnecessary probing.
        """
        with self.peers_lock:
            if peer_id in self.peers:
                self.peers[peer_id].mark_endpoint_validated()

    # =========================================================================
    # SQLite Persistence for Gossip State (Dec 29, 2025)
    # =========================================================================

    GOSSIP_PERSISTENCE_TABLE = "gossip_peer_states"
    GOSSIP_ENDPOINT_TABLE = "gossip_learned_endpoints"

    def _ensure_gossip_tables(self) -> bool:
        """Ensure gossip persistence tables exist in SQLite.

        Creates tables for storing:
        1. Peer states - Last known state from each peer (jobs, resources, health)
        2. Learned endpoints - Discovered peer connection info

        Returns:
            True if tables are ready, False on error

        December 2025: Added for SWIM/Raft stability improvements.
        Allows gossip state to survive P2P orchestrator restarts, reducing
        the time to recover cluster state after a coordinator failover.
        """
        # Table for peer states
        state_schema = f"""
            CREATE TABLE IF NOT EXISTS {self.GOSSIP_PERSISTENCE_TABLE} (
                node_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                timestamp REAL NOT NULL,
                cluster_epoch INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """
        state_index = f"""
            CREATE INDEX IF NOT EXISTS idx_gossip_timestamp
            ON {self.GOSSIP_PERSISTENCE_TABLE}(timestamp DESC)
        """

        # Table for learned endpoints
        endpoint_schema = f"""
            CREATE TABLE IF NOT EXISTS {self.GOSSIP_ENDPOINT_TABLE} (
                node_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                tailscale_ip TEXT,
                learned_at REAL NOT NULL,
                last_verified REAL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0
            )
        """
        endpoint_index = f"""
            CREATE INDEX IF NOT EXISTS idx_endpoint_learned
            ON {self.GOSSIP_ENDPOINT_TABLE}(learned_at DESC)
        """

        # Use base class helper to create tables
        state_ok = self._ensure_table(
            self.GOSSIP_PERSISTENCE_TABLE,
            state_schema,
            state_index,
        )
        endpoint_ok = self._ensure_table(
            self.GOSSIP_ENDPOINT_TABLE,
            endpoint_schema,
            endpoint_index,
        )

        return state_ok and endpoint_ok

    def _persist_gossip_state(self, node_id: str, state: dict) -> bool:
        """Persist a peer's gossip state to SQLite.

        Args:
            node_id: The peer's node ID
            state: The state dictionary to persist

        Returns:
            True on success, False on error

        Note: Uses upsert pattern (INSERT OR REPLACE) for simplicity.
        """
        now = time.time()
        timestamp = state.get("timestamp", now)
        epoch = state.get("cluster_epoch", getattr(self, "_cluster_epoch", 0))

        try:
            state_json = json.dumps(state)
        except (TypeError, ValueError) as e:
            self._log_warning(f"Failed to serialize state for {node_id}: {e}")
            return False

        result = self._execute_db_query(
            f"""
            INSERT OR REPLACE INTO {self.GOSSIP_PERSISTENCE_TABLE}
            (node_id, state_json, timestamp, cluster_epoch, created_at, updated_at)
            VALUES (?, ?, ?, ?,
                COALESCE((SELECT created_at FROM {self.GOSSIP_PERSISTENCE_TABLE} WHERE node_id = ?), ?),
                ?)
            """,
            (node_id, state_json, timestamp, epoch, node_id, now, now),
            fetch=False,
            commit=True,
        )
        return result is not None and result > 0

    def _persist_learned_endpoint(
        self,
        node_id: str,
        host: str,
        port: int,
        tailscale_ip: str | None = None,
    ) -> bool:
        """Persist a learned peer endpoint to SQLite.

        Args:
            node_id: The peer's node ID
            host: The peer's host/IP address
            port: The peer's port number
            tailscale_ip: Optional Tailscale IP

        Returns:
            True on success, False on error
        """
        now = time.time()

        result = self._execute_db_query(
            f"""
            INSERT INTO {self.GOSSIP_ENDPOINT_TABLE}
            (node_id, host, port, tailscale_ip, learned_at, success_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(node_id) DO UPDATE SET
                host = excluded.host,
                port = excluded.port,
                tailscale_ip = COALESCE(excluded.tailscale_ip, tailscale_ip),
                last_verified = excluded.learned_at,
                success_count = success_count + 1
            """,
            (node_id, host, port, tailscale_ip, now),
            fetch=False,
            commit=True,
        )
        return result is not None

    def _load_persisted_gossip_states(self, max_age_seconds: float = 3600.0) -> dict[str, dict]:
        """Load persisted gossip states from SQLite.

        Args:
            max_age_seconds: Only load states newer than this (default 1 hour)

        Returns:
            Dictionary of {node_id: state_dict}

        Called during P2P startup to recover cluster state quickly.
        """
        cutoff = time.time() - max_age_seconds

        rows = self._execute_db_query(
            f"""
            SELECT node_id, state_json, timestamp, cluster_epoch
            FROM {self.GOSSIP_PERSISTENCE_TABLE}
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
            fetch=True,
            commit=False,
        )

        if not rows:
            return {}

        states = {}
        for row in rows:
            try:
                node_id, state_json, timestamp, epoch = row
                state = json.loads(state_json)
                state["timestamp"] = timestamp  # Ensure timestamp is set
                state["cluster_epoch"] = epoch
                state["_persisted"] = True  # Mark as loaded from disk
                states[node_id] = state
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                self._log_warning(f"Failed to load persisted state: {e}")
                continue

        self._log_info(f"Loaded {len(states)} persisted gossip states")
        return states

    def _load_persisted_endpoints(self, max_age_seconds: float = 1800.0) -> dict[str, dict]:
        """Load persisted learned endpoints from SQLite.

        Args:
            max_age_seconds: Only load endpoints newer than this (default 30 min)

        Returns:
            Dictionary of {node_id: endpoint_info}
        """
        cutoff = time.time() - max_age_seconds

        rows = self._execute_db_query(
            f"""
            SELECT node_id, host, port, tailscale_ip, learned_at, success_count, failure_count
            FROM {self.GOSSIP_ENDPOINT_TABLE}
            WHERE learned_at > ?
            ORDER BY success_count DESC
            """,
            (cutoff,),
            fetch=True,
            commit=False,
        )

        if not rows:
            return {}

        endpoints = {}
        for row in rows:
            try:
                node_id, host, port, tailscale_ip, learned_at, successes, failures = row
                endpoints[node_id] = {
                    "host": host,
                    "port": port,
                    "tailscale_ip": tailscale_ip,
                    "learned_at": learned_at,
                    "success_count": successes,
                    "failure_count": failures,
                    "_persisted": True,
                }
            except (ValueError, IndexError) as e:
                self._log_warning(f"Failed to load persisted endpoint: {e}")
                continue

        self._log_info(f"Loaded {len(endpoints)} persisted endpoints")
        return endpoints

    def _cleanup_persisted_gossip_state(self, max_age_seconds: float = 7200.0) -> int:
        """Clean up old persisted gossip state from SQLite.

        Args:
            max_age_seconds: Delete states older than this (default 2 hours)

        Returns:
            Number of rows deleted
        """
        cutoff = time.time() - max_age_seconds

        # Clean peer states
        state_deleted = self._execute_db_query(
            f"DELETE FROM {self.GOSSIP_PERSISTENCE_TABLE} WHERE timestamp < ?",
            (cutoff,),
            fetch=False,
            commit=True,
        ) or 0

        # Clean learned endpoints
        endpoint_deleted = self._execute_db_query(
            f"DELETE FROM {self.GOSSIP_ENDPOINT_TABLE} WHERE learned_at < ?",
            (cutoff,),
            fetch=False,
            commit=True,
        ) or 0

        total = state_deleted + endpoint_deleted
        if total > 0:
            self._log_debug(f"Cleaned {state_deleted} states, {endpoint_deleted} endpoints from persistence")

        return total

    def _save_gossip_state_periodic(self) -> None:
        """Periodically save current gossip state to SQLite.

        Called from the gossip loop to persist state. This ensures that
        if the P2P orchestrator restarts, it can quickly recover the
        cluster state from SQLite rather than waiting for fresh gossip.

        Rate limited to once per minute to avoid excessive disk I/O.
        """
        now = time.time()

        # Rate limit: persist every 60 seconds
        last_persist = getattr(self, "_last_gossip_persist", 0)
        if now - last_persist < 60:
            return
        self._last_gossip_persist = now

        # Ensure tables exist (lazy initialization)
        if not self._ensure_gossip_tables():
            return

        # Persist current peer states
        persisted_count = 0
        for node_id, state in list(self._gossip_peer_states.items()):
            if self._persist_gossip_state(node_id, state):
                persisted_count += 1

        # Persist learned endpoints
        endpoint_count = 0
        for node_id, endpoint in list(self._gossip_learned_endpoints.items()):
            host = endpoint.get("host")
            port = endpoint.get("port")
            if host and port:
                if self._persist_learned_endpoint(
                    node_id,
                    host,
                    port,
                    endpoint.get("tailscale_ip"),
                ):
                    endpoint_count += 1

        # Clean up old persisted data
        self._cleanup_persisted_gossip_state()

        if persisted_count > 0 or endpoint_count > 0:
            self._log_debug(
                f"Persisted {persisted_count} peer states, {endpoint_count} endpoints"
            )

    def _restore_gossip_state_on_startup(self) -> None:
        """Restore gossip state from SQLite on startup.

        Called during P2P initialization to recover cluster state quickly
        after a restart. This reduces the time needed to rebuild the
        cluster view from O(minutes) to O(seconds).

        December 2025: Part of SWIM/Raft stability improvements.
        """
        # Ensure tables exist
        if not self._ensure_gossip_tables():
            self._log_warning("Gossip tables not available, starting with empty state")
            return

        # Load persisted peer states
        persisted_states = self._load_persisted_gossip_states()
        if persisted_states:
            # Merge with any existing states (prefer fresher data)
            for node_id, state in persisted_states.items():
                existing = self._gossip_peer_states.get(node_id)
                if existing is None or state.get("timestamp", 0) > existing.get("timestamp", 0):
                    self._gossip_peer_states[node_id] = state

        # Load persisted endpoints
        persisted_endpoints = self._load_persisted_endpoints()
        if persisted_endpoints:
            for node_id, endpoint in persisted_endpoints.items():
                existing = self._gossip_learned_endpoints.get(node_id)
                if existing is None or endpoint.get("learned_at", 0) > existing.get("learned_at", 0):
                    self._gossip_learned_endpoints[node_id] = endpoint

        total_restored = len(persisted_states) + len(persisted_endpoints)
        if total_restored > 0:
            self._log_info(
                f"Restored gossip state: {len(persisted_states)} peer states, "
                f"{len(persisted_endpoints)} endpoints"
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
        # Dec 30, 2025: Now async with locking for thread safety
        await self._cleanup_gossip_state()

        # Dec 29, 2025: Periodically persist gossip state to SQLite
        # Enables fast cluster state recovery after P2P restart
        self._save_gossip_state_periodic()

        # Dec 30, 2025: Validate stale endpoints to prevent partition isolation
        # Probes alternate IPs when peers haven't responded for endpoint_ttl_seconds
        await self._validate_stale_endpoints()

        now = time.time()

        # Rate limit: gossip interval configurable via RINGRIFT_GOSSIP_INTERVAL
        last_gossip = getattr(self, "_last_gossip_time", 0)
        if now - last_gossip < self.GOSSIP_INTERVAL_SECONDS:
            return
        self._last_gossip_time = now

        # Prepare our state to share
        # Dec 30, 2025: Use async version to avoid blocking event loop with subprocess calls
        if hasattr(self, '_update_self_info_async'):
            await self._update_self_info_async()
        else:
            self._update_self_info()  # Fallback for legacy orchestrators
        local_state = self._build_local_gossip_state(now)

        # Select K random peers to gossip with
        # Dec 29, 2025: Increased fanout (3 → 5) for faster state propagation
        # and improved partition recovery. With 30+ peers, 5-peer fanout gives
        # O(log30/log5) ≈ 2.1 rounds for full propagation vs ~3.1 with fanout=3.
        # Dec 30, 2025: Coordinators get higher fanout (8) for more reliable state
        # propagation, since they're critical for cluster-wide visibility.
        # Fanout is now configurable via RINGRIFT_GOSSIP_FANOUT_LEADER/_FOLLOWER env vars.
        is_coordinator = getattr(self, "role", None) in ("coordinator", "leader") or \
                         getattr(self, "is_coordinator", False)
        GOSSIP_FANOUT = self.GOSSIP_FANOUT_LEADER if is_coordinator else self.GOSSIP_FANOUT_FOLLOWER

        # Dec 2025: Copy peer IDs under lock to avoid stale references.
        # Previously, we copied NodeInfo objects which could become stale
        # between lock release and gossip sending, causing race conditions.
        import random
        with self.peers_lock:
            alive_peer_ids = [
                p.node_id for p in self.peers.values()
                if p.is_alive() and not getattr(p, "retired", False)
            ]

        if not alive_peer_ids:
            return

        # Dec 30, 2025: Prioritize voter nodes in gossip fanout to prevent partitioning
        # This ensures voter nodes (critical for quorum) are always included
        voter_ids = getattr(self, "voter_node_ids", []) or []
        voter_peer_ids = [p for p in alive_peer_ids if p in voter_ids]
        non_voter_peer_ids = [p for p in alive_peer_ids if p not in voter_ids]

        selected_ids = []
        # Always include at least 1 voter if available
        if voter_peer_ids:
            selected_ids.append(random.choice(voter_peer_ids))

        # Fill remaining fanout with other peers
        remaining_fanout = GOSSIP_FANOUT - len(selected_ids)
        if remaining_fanout > 0 and non_voter_peer_ids:
            sample_size = min(remaining_fanout, len(non_voter_peer_ids))
            selected_ids.extend(random.sample(non_voter_peer_ids, sample_size))

        # If we still have room and more voters, add them
        remaining = GOSSIP_FANOUT - len(selected_ids)
        if remaining > 0 and len(voter_peer_ids) > 1:
            unused_voters = [v for v in voter_peer_ids if v not in selected_ids]
            if unused_voters:
                selected_ids.extend(random.sample(unused_voters, min(remaining, len(unused_voters))))

        # Import session helper
        try:
            from .network import get_client_session
        except ImportError:
            # Fallback for legacy imports
            get_client_session = None

        # Send gossip to selected peers, fetching fresh peer info under lock
        # Jan 2, 2026 (Sprint 3.5): Use per-peer timeout based on NAT status
        # NAT-blocked nodes need longer timeouts (120s) due to relay latency
        try:
            from app.p2p.constants import get_peer_timeout_for_node, PEER_TIMEOUT_NAT_BLOCKED
        except ImportError:
            get_peer_timeout_for_node = None
            PEER_TIMEOUT_NAT_BLOCKED = 120

        try:
            from app.config.coordination_defaults import GossipDefaults
            default_timeout = GossipDefaults.PROBE_HTTP_TIMEOUT
        except ImportError:
            default_timeout = 5

        for peer_id in selected_ids:
            # Jan 3, 2026: Skip peers in backoff period (exponential backoff for failures)
            if self.health_tracker.should_skip_peer(peer_id):
                backoff_remaining = self.health_tracker.get_backoff_seconds(peer_id)
                self._log_debug(
                    f"[Gossip] Skipping {peer_id} - in backoff ({backoff_remaining:.1f}s remaining)"
                )
                continue

            with self.peers_lock:
                peer = self.peers.get(peer_id)
            if peer and peer.is_alive():
                # Jan 2, 2026: Per-peer timeout based on NAT status
                is_nat_blocked = getattr(peer, "nat_blocked", False)
                if get_peer_timeout_for_node and is_nat_blocked:
                    peer_timeout = get_peer_timeout_for_node(
                        is_coordinator=False,
                        nat_blocked=True,
                    )
                elif is_nat_blocked:
                    # Fallback if function unavailable
                    peer_timeout = PEER_TIMEOUT_NAT_BLOCKED
                else:
                    peer_timeout = default_timeout
                timeout = ClientTimeout(total=peer_timeout)
                await self._send_gossip_to_peer(peer, local_state, timeout, get_client_session)

    def _get_circuit_breaker_gossip_state(self) -> dict[str, Any] | None:
        """Get circuit breaker states for gossip replication.

        Jan 3, 2026: Enables cluster-wide circuit breaker awareness.
        When a node discovers a failing target (e.g., a host with network issues),
        it shares this information via gossip. Other nodes can then preemptively
        avoid that target, reducing duplicated failure discovery.

        Only shares OPEN and HALF_OPEN circuits (not CLOSED) to minimize
        gossip payload size. Circuit breaker state is keyed by operation type
        (ssh, http, p2p, etc.) and target (host/endpoint).

        Returns:
            Dict with open circuit states, or None if no open circuits or
            circuit breaker module unavailable.
        """
        try:
            from app.distributed.circuit_breaker import (
                get_circuit_registry,
                CircuitState,
            )
        except ImportError:
            return None

        try:
            registry = get_circuit_registry()
            open_circuits = registry.get_all_open_circuits()

            if not open_circuits:
                return None

            # Convert to serializable format
            # Structure: {operation_type: {target: {state, failure_count, opened_at}}}
            result = {}
            now = time.time()

            for op_type, targets in open_circuits.items():
                result[op_type] = {}
                for target, status in targets.items():
                    # Only include minimal info needed for replication
                    result[op_type][target] = {
                        "state": status.state.value,
                        "failure_count": status.failure_count,
                        "opened_at": status.opened_at,
                        "escalation_tier": status.escalation_tier,
                        # Age of the open circuit for freshness decisions
                        "age_seconds": now - status.opened_at if status.opened_at else 0,
                    }

            return result if result else None

        except (AttributeError, TypeError, ValueError) as e:
            self._log_debug(f"Circuit breaker state not available for gossip: {type(e).__name__}: {e}")
            return None

    def _build_local_gossip_state(self, now: float) -> dict[str, Any]:
        """Build local state dict to share via gossip."""
        local_state = {
            "node_id": self.node_id,
            "timestamp": now,
            "version": int(now * 1000),  # Millisecond version for conflict resolution
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "leader_id": self.leader_id,
            "leader_lease_expires": getattr(self, "leader_lease_expires", 0),
            # Jan 2026: ULSM epoch for stale leader claim rejection
            "leadership_epoch": getattr(self, "_leadership_sm", None) and self._leadership_sm.epoch or 0,
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

        # December 2025 Phase 3D: Include model locations for distribution tracking
        if hasattr(self, "_get_local_model_locations"):
            model_locations = self._get_local_model_locations()
            if model_locations:
                local_state["model_locations"] = model_locations

        # December 2025: Include config version for distributed config sync
        try:
            from app.config.cluster_config import get_config_version

            config_version = get_config_version()
            local_state["config"] = config_version.to_dict()
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            # Jan 2026: Narrowed exception types for better debugging
            self._log_debug(f"Config version not available for gossip: {type(e).__name__}: {e}")

        # Jan 3, 2026: Include circuit breaker states for cluster-wide failure awareness
        # This allows nodes to preemptively avoid targets that other nodes have found to be failing
        cb_state = self._get_circuit_breaker_gossip_state()
        if cb_state:
            local_state["circuit_breakers"] = cb_state

        return local_state

    async def _send_gossip_to_peer(
        self,
        peer: Any,
        local_state: dict[str, Any],
        timeout: Any,
        get_client_session: Any,
    ) -> None:
        """Send gossip message to a single peer.

        Dec 28, 2025 (Phase 6): Integrated with GossipHealthTracker for peer
        health status. Records success/failure and emits NODE_SUSPECT when
        a peer has too many consecutive gossip failures.
        """
        success = False
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
            # Dec 30, 2025: Use asyncio.to_thread() to avoid blocking event loop
            json_bytes = json.dumps(gossip_payload).encode("utf-8")
            original_size = len(json_bytes)
            compressed_bytes = await asyncio.to_thread(
                gzip.compress, json_bytes, 6  # compresslevel=6
            )
            compressed_size = len(compressed_bytes)

            # Track compression metrics (method now in this class)
            self._record_gossip_compression(original_size, compressed_size)

            # Dec 30, 2025: Validate message size before sending
            if compressed_size > self.GOSSIP_MAX_MESSAGE_SIZE_BYTES:
                self._log_error(
                    f"Gossip message too large: {compressed_size / 1024:.1f}KB "
                    f"(max {self.GOSSIP_MAX_MESSAGE_SIZE_BYTES / 1024:.0f}KB). "
                    f"Peer: {peer.node_id}. Skipping gossip to prevent network issues."
                )
                return  # Skip this gossip exchange
            elif compressed_size > self.GOSSIP_MESSAGE_SIZE_WARNING_BYTES:
                self._log_warning(
                    f"Large gossip message: {compressed_size / 1024:.1f}KB "
                    f"(warning threshold: {self.GOSSIP_MESSAGE_SIZE_WARNING_BYTES / 1024:.0f}KB). "
                    f"Peer: {peer.node_id}. Consider reducing state size."
                )

            start_time = time.time()

            # Try each URL for the peer
            if get_client_session:
                async with get_client_session(timeout) as session:
                    success = await self._try_gossip_urls(
                        session, peer, compressed_bytes, start_time
                    )
            else:
                # Fallback without session helper
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    success = await self._try_gossip_urls(
                        session, peer, compressed_bytes, start_time
                    )

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, AttributeError):
            success = False

        # Dec 28, 2025 (Phase 6): Track gossip health and emit events
        await self._update_gossip_health(peer.node_id, success)

    async def _update_gossip_health(self, peer_id: str, success: bool) -> None:
        """Update gossip health tracker and emit events as needed.

        Dec 28, 2025 (Phase 6): Central point for gossip health tracking.
        Emits NODE_SUSPECT when failures cross threshold, NODE_RECOVERED
        when a previously suspect peer recovers.

        Args:
            peer_id: The peer ID
            success: Whether the gossip exchange succeeded
        """
        # Ensure tracker is initialized
        if not hasattr(self, "_gossip_health_tracker"):
            self._gossip_health_tracker = GossipHealthTracker()

        tracker = self._gossip_health_tracker

        if success:
            # Record success - check if peer was previously suspected
            was_suspected = tracker.record_gossip_success(peer_id)

            # Dec 30, 2025: Mark endpoint as validated to prevent stale IP probing
            self._on_heartbeat_success(peer_id)

            if was_suspected:
                # Peer recovered from suspect state
                self._log_info(f"Peer {peer_id} recovered from gossip failures")
                # Emit NODE_RECOVERED event if we have the method
                if hasattr(self, "_emit_node_recovered"):
                    try:
                        await self._emit_node_recovered(
                            node_id=peer_id,
                            recovery_type="gossip_success",
                            offline_duration_seconds=0.0,
                        )
                    except (AttributeError, RuntimeError, TypeError):
                        pass
        else:
            # Record failure - check if we should emit suspect
            should_emit, failure_count = tracker.record_gossip_failure(peer_id)
            if should_emit:
                # Log and emit NODE_SUSPECT
                self._log_warning(
                    f"Peer {peer_id} has {failure_count} consecutive gossip failures, "
                    f"marking as suspect"
                )
                if hasattr(self, "_emit_node_suspect"):
                    try:
                        await self._emit_node_suspect(
                            node_id=peer_id,
                            last_seen=tracker.get_last_success(peer_id),
                            seconds_since_heartbeat=0.0,
                        )
                    except (AttributeError, RuntimeError, TypeError):
                        pass

    # Jan 3, 2026: Per-URL timeout for gossip requests
    # Prevents single slow URL from blocking entire gossip round
    GOSSIP_PER_URL_TIMEOUT: float = 5.0

    async def _try_gossip_urls(
        self,
        session: Any,
        peer: Any,
        compressed_bytes: bytes,
        start_time: float,
    ) -> bool:
        """Try gossip to peer via multiple URLs.

        Dec 28, 2025 (Phase 6): Now returns bool indicating success/failure
        for integration with GossipHealthTracker.

        Jan 3, 2026: Added per-URL timeout protection to prevent slow URLs
        from blocking the entire gossip round. Each URL attempt is wrapped
        with asyncio.wait_for() using GOSSIP_PER_URL_TIMEOUT.

        Returns:
            True if gossip succeeded on any URL, False if all URLs failed.
        """
        for url in self._urls_for_peer(peer, "/gossip"):
            try:
                # Jan 3, 2026: Wrap each URL attempt with timeout
                success = await asyncio.wait_for(
                    self._try_single_gossip_url(
                        session, peer, url, compressed_bytes, start_time
                    ),
                    timeout=self.GOSSIP_PER_URL_TIMEOUT,
                )
                if success:
                    return True

            except asyncio.TimeoutError:
                # Per-URL timeout - try next URL
                self._log_debug(
                    f"[Gossip] Timeout ({self.GOSSIP_PER_URL_TIMEOUT}s) for {peer.node_id} at {url}"
                )
                continue
            except (aiohttp.ClientError, json.JSONDecodeError, AttributeError):
                continue

        return False  # All URLs failed

    async def _try_single_gossip_url(
        self,
        session: Any,
        peer: Any,
        url: str,
        compressed_bytes: bytes,
        start_time: float,
    ) -> bool:
        """Try gossip to a single URL.

        Jan 3, 2026: Extracted from _try_gossip_urls for per-URL timeout wrapping.

        Returns:
            True if gossip succeeded, False otherwise.
        """
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
                return True

        return False

    async def _read_gossip_response(self, resp: Any) -> dict:
        """Read and decompress gossip response.

        Dec 30, 2025: Uses asyncio.to_thread() for decompression to avoid
        blocking the event loop on large payloads.

        Jan 3, 2026: Added robust error handling for corrupted gzip payloads.
        Returns empty dict on corruption to allow gossip to continue with other peers.
        """
        import hashlib
        import zlib

        content_encoding = resp.headers.get("Content-Encoding", "")
        if content_encoding == "gzip":
            response_bytes = await resp.read()
            try:
                decompressed = await asyncio.to_thread(gzip.decompress, response_bytes)
                return json.loads(decompressed.decode("utf-8"))
            except (gzip.BadGzipFile, zlib.error, OSError) as e:
                # Log corruption with payload hash for debugging
                payload_hash = hashlib.sha256(response_bytes).hexdigest()[:16]
                self._log_warning(
                    f"Gossip payload corruption detected: {type(e).__name__}: {e}, "
                    f"payload_hash={payload_hash}, size={len(response_bytes)} bytes"
                )
                # Emit event for monitoring (safe emit won't crash if unavailable)
                self._safe_emit_event(
                    "GOSSIP_CORRUPTION_DETECTED",
                    {
                        "error_type": type(e).__name__,
                        "payload_hash": payload_hash,
                        "payload_size": len(response_bytes),
                        "timestamp": time.time(),
                    },
                )
                return {}  # Return empty dict to allow gossip to continue
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Decompressed but invalid JSON/encoding
                self._log_warning(f"Gossip payload decode error: {type(e).__name__}: {e}")
                return {}
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
                # Dec 2025: Use dynamic is_alive() instead of hardcoded True
                "is_alive": peer.is_alive(),
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

        # December 2025 Phase 3D: Process model locations for distribution tracking
        sender_state = response.get("sender_state", {})
        if sender_state.get("model_locations"):
            self._process_model_locations(sender_state["model_locations"])
        # Also check known_states for model locations
        for node_id, state in known_states.items():
            if node_id != self.node_id and state.get("model_locations"):
                self._process_model_locations(state["model_locations"])

        # December 2025: Check config freshness for distributed config sync
        sender_id = sender_state.get("node_id") if sender_state else None
        if sender_id and sender_state.get("config"):
            asyncio.create_task(
                self._check_config_freshness(sender_id, sender_state["config"])
            )

        # Jan 3, 2026: Process circuit breaker states for cluster-wide failure awareness
        if sender_state.get("circuit_breakers"):
            self._process_circuit_breaker_states(
                sender_id or "unknown",
                sender_state["circuit_breakers"]
            )
        # Also check known_states for circuit breaker info
        for node_id, state in known_states.items():
            if node_id != self.node_id and state.get("circuit_breakers"):
                self._process_circuit_breaker_states(node_id, state["circuit_breakers"])

    async def _check_config_freshness(
        self, peer_id: str, peer_config: dict[str, Any]
    ) -> None:
        """Compare peer's config version to ours, request sync if stale.

        Called when we receive gossip containing a config version newer
        than ours. Uses rsync/SSH to pull the updated config.

        Args:
            peer_id: Node ID of the peer with newer config.
            peer_config: Config version dict with hash, timestamp, source_node.
        """
        try:
            from app.config.cluster_config import get_config_version, get_config_cache

            local_version = get_config_version()
            peer_timestamp = peer_config.get("timestamp", 0)
            local_timestamp = local_version.timestamp

            # 60 second tolerance to avoid sync storms
            if peer_timestamp <= local_timestamp + 60:
                return

            self._log_info(
                f"[ConfigSync] Peer {peer_id} has newer config "
                f"({peer_timestamp:.0f} vs {local_timestamp:.0f}), requesting sync"
            )

            # Request config sync from peer
            await self._request_config_sync(peer_id)

        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as e:
            # Jan 2026: Narrowed exception types for better debugging
            self._log_debug(f"[ConfigSync] Error checking config freshness: {e}")

    async def _request_config_sync(self, source_node: str) -> None:
        """Request fresh config from node with newer version.

        Pulls distributed_hosts.yaml from the source node via SSH/rsync,
        then triggers a config reload and emits CONFIG_UPDATED event.

        Args:
            source_node: Node ID to pull config from.
        """
        try:
            from app.config.cluster_config import get_config_cache, get_cluster_nodes

            # Find source node connection info
            nodes = get_cluster_nodes()
            source = nodes.get(source_node)
            if not source or not source.best_ip:
                self._log_debug(f"[ConfigSync] Cannot find connection info for {source_node}")
                return

            # Build rsync command
            import subprocess

            local_config_path = "config/distributed_hosts.yaml"
            remote_config_path = f"{source.ringrift_path}/config/distributed_hosts.yaml"
            ssh_args = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
            if source.ssh_key:
                ssh_args.extend(["-i", source.ssh_key])
            if source.ssh_port != 22:
                ssh_args.extend(["-p", str(source.ssh_port)])

            remote_spec = f"{source.ssh_user}@{source.best_ip}:{remote_config_path}"
            cmd = ["rsync", "-az", "-e", f"ssh {' '.join(ssh_args)}", remote_spec, local_config_path]

            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, timeout=30
            )

            if result.returncode != 0:
                self._log_debug(f"[ConfigSync] rsync failed: {result.stderr.decode()[:200]}")
                return

            # Force config reload
            cache = get_config_cache()
            cache.get_config(force_reload=True)

            self._log_info(f"[ConfigSync] Config synced from {source_node}")

            # Emit CONFIG_UPDATED event for daemons to reload
            try:
                from app.coordination.data_events import DataEventType
                from scripts.p2p.handlers.handlers_base import get_event_bridge

                bridge = get_event_bridge()
                await bridge.emit(
                    DataEventType.CONFIG_UPDATED.value if hasattr(DataEventType, "CONFIG_UPDATED") else "config_updated",
                    {
                        "source_node": source_node,
                        "timestamp": time.time(),
                    },
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                # Jan 2026: Narrowed exception types for better debugging
                self._log_debug(f"[ConfigSync] CONFIG_UPDATED event emission failed: {type(e).__name__}: {e}")

        except (ImportError, OSError, subprocess.SubprocessError, AttributeError, KeyError) as e:
            # Jan 2026: Narrowed exception types for better debugging
            self._log_debug(f"[ConfigSync] Sync from {source_node} failed: {e}")

    def _process_sender_state(self, sender_state: dict) -> None:
        """Process the sender's state from a gossip response."""
        sender_id = sender_state.get("node_id")
        if not sender_id or sender_id == self.node_id:
            return

        # Jan 3, 2026: Use sync lock to prevent race with cleanup
        lock = getattr(self, "_gossip_state_sync_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            existing = self._gossip_peer_states.get(sender_id, {})
            # Last-write-wins conflict resolution
            if sender_state.get("version", 0) > existing.get("version", 0):
                self._gossip_peer_states[sender_id] = sender_state
        finally:
            if lock is not None:
                lock.release()

        # Update leader info if sender claims to know a leader
        # Jan 2026: Use ULSM epoch validation to reject stale claims
        if sender_state.get("leader_id") and not self.leader_id:
            claimed_leader = sender_state.get("leader_id")
            claimed_epoch = sender_state.get("leadership_epoch", 0)
            lease_expires = sender_state.get("leader_lease_expires", 0)

            # Check if we have ULSM state machine for validation
            if hasattr(self, "_leadership_sm") and self._leadership_sm:
                if self._leadership_sm.validate_leader_claim(claimed_leader, claimed_epoch, lease_expires):
                    self.leader_id = claimed_leader
                    self.last_leader_seen = time.time()
                    logger.debug(f"Gossip: Accepted leader claim {claimed_leader} epoch={claimed_epoch}")
                else:
                    logger.debug(f"Gossip: Rejected leader claim {claimed_leader} epoch={claimed_epoch}")
            else:
                # Fallback: simple lease expiry check (pre-ULSM behavior)
                if lease_expires > time.time():
                    self.leader_id = claimed_leader
                    self.last_leader_seen = time.time()

    def _process_known_states(self, known_states: dict[str, dict]) -> None:
        """Process known states from gossip propagation."""
        # Jan 3, 2026: Use sync lock to prevent race with cleanup
        lock = getattr(self, "_gossip_state_sync_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            for node_id, state in known_states.items():
                if node_id == self.node_id:
                    continue
                existing = self._gossip_peer_states.get(node_id, {})
                if state.get("version", 0) > existing.get("version", 0):
                    self._gossip_peer_states[node_id] = state
        finally:
            if lock is not None:
                lock.release()

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

    def _process_model_locations(self, model_locations: list[dict]) -> None:
        """December 2025 Phase 3D: Process model locations from gossip for distribution tracking.

        Syncs model location data from peers into the local cluster_manifest.db,
        enabling any node to query model availability across the cluster.

        Args:
            model_locations: List of model location dicts from peer gossip
        """
        if not model_locations:
            return

        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            count = manifest.sync_model_locations_from_peers(model_locations)

            if count > 0:
                self._log_debug(f"Synced {count} model locations from gossip")

        except ImportError:
            # Cluster manifest not available (e.g., minimal node)
            pass
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            # Jan 3, 2026: Narrowed from broad Exception to specific types
            # Log but don't fail on sync errors
            self._log_debug(f"Model location sync failed: {type(e).__name__}: {e}")

    def _process_circuit_breaker_states(
        self, source_node: str, cb_states: dict[str, dict]
    ) -> None:
        """Process circuit breaker states from gossip for preemptive failure avoidance.

        Jan 3, 2026: When a peer reports OPEN circuits for targets, we can preemptively
        record failures on our local circuit breakers. This prevents all nodes in the
        cluster from independently discovering the same failure, reducing wasted
        connection attempts and speeding up cluster-wide failure adaptation.

        Strategy:
        - Only process circuits that are OPEN or HALF_OPEN (not CLOSED)
        - Only apply if the remote circuit is "fresh" (opened within last 5 minutes)
        - Record a single failure locally (doesn't immediately open, just increments)
        - This way, if local attempts also fail, circuit opens faster

        Args:
            source_node: Node ID that reported the circuit breaker states
            cb_states: Dict of {operation_type: {target: {state, failure_count, ...}}}
        """
        if not cb_states:
            return

        try:
            from app.distributed.circuit_breaker import (
                get_circuit_registry,
                CircuitState,
            )
        except ImportError:
            return

        try:
            registry = get_circuit_registry()
            now = time.time()

            # Max age for considering a remote circuit "fresh"
            # If circuit opened >5 min ago, target may have recovered
            MAX_CIRCUIT_AGE_SECONDS = 300.0

            processed_count = 0

            for op_type, targets in cb_states.items():
                breaker = registry.get_breaker(op_type)

                for target, state_info in targets.items():
                    remote_state = state_info.get("state", "")
                    age_seconds = state_info.get("age_seconds", 0)

                    # Skip if not OPEN/HALF_OPEN or too old
                    if remote_state not in ("open", "half_open"):
                        continue
                    if age_seconds > MAX_CIRCUIT_AGE_SECONDS:
                        continue

                    # Check if we already have this circuit open
                    local_status = breaker.get_status(target)
                    if local_status and local_status.state != CircuitState.CLOSED:
                        # Already tracking failure for this target
                        continue

                    # Record a preemptive failure to bias our local circuit
                    # This doesn't immediately open the circuit, just increments failure count
                    # If local attempts also fail, we'll reach threshold faster
                    breaker.record_failure(target, preemptive=True)
                    processed_count += 1

            if processed_count > 0:
                self._log_debug(
                    f"[CB-Gossip] Applied {processed_count} preemptive failures "
                    f"from {source_node} circuit breaker state"
                )

        except (AttributeError, TypeError, ValueError, KeyError) as e:
            self._log_debug(f"Circuit breaker gossip processing failed: {type(e).__name__}: {e}")

    def _get_local_model_locations(self) -> list[dict]:
        """December 2025 Phase 3D: Get local model locations for gossip sharing.

        Queries the local cluster_manifest.db for model locations on this node,
        returning them in a format suitable for gossip propagation.

        Returns:
            List of model location dicts with this node's models
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()

            # Only share locations for this node to reduce gossip payload
            with manifest._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT model_path, node_id, board_type, num_players,
                           model_version, file_size, registered_at, last_seen
                    FROM model_locations
                    WHERE node_id = ?
                    ORDER BY last_seen DESC
                    LIMIT 50
                """, (self.node_id,))

                return [
                    {
                        "model_path": row[0],
                        "node_id": row[1],
                        "board_type": row[2],
                        "num_players": row[3],
                        "model_version": row[4],
                        "file_size": row[5],
                        "registered_at": row[6],
                        "last_seen": row[7],
                    }
                    for row in cursor.fetchall()
                ]

        except ImportError:
            return []
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            # Database query errors - log and return empty
            logger.debug(f"Database error in get_active_checkpoint_peers: {e}")
            return []
        except (TypeError, KeyError) as e:
            # Malformed row data - log and return empty
            logger.debug(f"Data format error in get_active_checkpoint_peers: {e}")
            return []

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
        except (OSError, ConnectionError, asyncio.TimeoutError, AttributeError) as e:
            # Jan 2026: Narrowed exception types for better debugging
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

        Frequency: Every 2 minutes with a random healthy peer
        """
        if aiohttp is None:
            return

        now = time.time()

        # Rate limit: anti-entropy interval is now configurable via RINGRIFT_ANTI_ENTROPY_INTERVAL
        # Dec 29, 2025: Reduced interval (300s → 120s) for faster partition recovery.
        # This catches missed updates more quickly, especially for the local-mac node
        # which has intermittent visibility due to stricter home network NAT.
        last_repair = getattr(self, "_last_anti_entropy_repair", 0)
        if now - last_repair < self.ANTI_ENTROPY_INTERVAL_SECONDS:
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
        full_state = await self._build_anti_entropy_state_async(now)

        # Send anti-entropy request
        await self._send_anti_entropy_request(peer, full_state, now)

    async def _build_anti_entropy_state_async(self, now: float) -> dict[str, Any]:
        """Build full state for anti-entropy repair (async version).

        Dec 30, 2025: Converted to async to avoid blocking event loop.
        """
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

        # Include our own state (use async version if available)
        if hasattr(self, '_update_self_info_async'):
            await self._update_self_info_async()
        else:
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
        # Dec 30, 2025: Use configurable timeout instead of hardcoded value
        try:
            from app.config.coordination_defaults import GossipDefaults
            timeout = ClientTimeout(total=GossipDefaults.ANTI_ENTROPY_TIMEOUT)
        except ImportError:
            timeout = ClientTimeout(total=10)  # Fallback

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

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            # Track anti-entropy failures for observability (Dec 30, 2025)
            self._record_gossip_metrics("anti_entropy_error")
            self._log_debug(f"Anti-entropy repair failed for {peer.node_id}: {type(e).__name__}: {e}")

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
    # Gossip Health Tracking (Phase 6 - Dec 28, 2025)
    # =========================================================================

    def get_gossip_health_tracker(self) -> GossipHealthTracker:
        """Get the gossip health tracker instance.

        Public API for accessing the gossip health tracker.
        Returns the tracker, creating one if it doesn't exist.

        December 2025 (Phase 6): Added for cluster availability improvements.
        """
        if not hasattr(self, "_gossip_health_tracker"):
            self._gossip_health_tracker = GossipHealthTracker()
        return self._gossip_health_tracker

    def get_gossip_suspected_peers(self) -> set[str]:
        """Get the set of peers currently suspected due to gossip failures.

        Public API for querying which peers have failed consecutive gossip
        attempts and have been marked as suspect.

        December 2025 (Phase 6): Added for cluster availability improvements.

        Returns:
            Set of peer node IDs that are currently suspected
        """
        if hasattr(self, "_gossip_health_tracker"):
            return self._gossip_health_tracker.get_suspected_peers()
        return set()

    def get_gossip_failure_count(self, peer_id: str) -> int:
        """Get the consecutive gossip failure count for a specific peer.

        December 2025 (Phase 6): Added for cluster availability improvements.

        Args:
            peer_id: The peer to check

        Returns:
            Number of consecutive gossip failures for this peer
        """
        if hasattr(self, "_gossip_health_tracker"):
            return self._gossip_health_tracker.get_failure_count(peer_id)
        return 0

    # =========================================================================
    # Partition Detection (Phase 2.3 - Dec 29, 2025)
    # =========================================================================

    def detect_partition_status(self) -> tuple[str, float]:
        """Detect if we're in a network partition.

        December 2025 (Phase 2.3): Active partition detection for cluster stability.
        December 30, 2025: Fixed to exclude long-dead and retired peers from denominator.

        The health ratio now only considers "relevant" peers:
        - Alive peers (heartbeat within PEER_TIMEOUT)
        - Recently dead peers (dead < 5 minutes) - may recover
        - Excludes: retired peers, long-dead peers (>5 min without heartbeat)

        This prevents false "minority" partition detection when the peers dict
        contains stale entries from nodes that were spun down or never connected.

        Partition detection enables the cluster to:
        - Pause writes when in minority partition (prevent split-brain divergence)
        - Alert operators about network issues
        - Automatically resume when connectivity restores

        Returns:
            Tuple of (status, health_ratio) where:
            - status: 'healthy' (>50% peers alive), 'minority' (20-50%), 'isolated' (<20%)
            - health_ratio: Float between 0.0 and 1.0 representing peer connectivity
        """
        import time

        # Long-dead threshold: peers dead for >5 minutes are excluded from calculation
        LONG_DEAD_THRESHOLD = 300  # 5 minutes

        with self.peers_lock:
            if len(self.peers) == 0:
                # No peers known at all - we're isolated
                return ("isolated", 0.0)

            now = time.time()
            alive_peers = 0
            relevant_peers = 0

            for p in self.peers.values():
                # Skip retired peers entirely - they're intentionally offline
                if getattr(p, 'retired', False):
                    continue

                time_since_heartbeat = now - p.last_heartbeat

                if p.is_alive():
                    alive_peers += 1
                    relevant_peers += 1
                elif time_since_heartbeat < LONG_DEAD_THRESHOLD:
                    # Recently dead - still count in denominator (may recover)
                    relevant_peers += 1
                # else: long-dead peer, exclude from both counts

            if relevant_peers == 0:
                # All peers are long-dead or retired - we're isolated
                return ("isolated", 0.0)

        health_ratio = alive_peers / relevant_peers

        if health_ratio > 0.5:
            return ("healthy", health_ratio)
        elif health_ratio > 0.2:
            return ("minority", health_ratio)
        else:
            return ("isolated", health_ratio)

    def get_partition_details(self) -> dict[str, Any]:
        """Get detailed partition status information.

        December 2025 (Phase 2.3): Extended partition info for monitoring.

        Returns:
            Dict with partition status, counts, and peer details.
        """
        status, ratio = self.detect_partition_status()

        with self.peers_lock:
            total_peers = len(self.peers)
            alive_peers = [p.node_id for p in self.peers.values() if p.is_alive()]
            dead_peers = [p.node_id for p in self.peers.values() if not p.is_alive()]
            suspected_peers = list(self.get_gossip_suspected_peers())

        return {
            "status": status,
            "health_ratio": round(ratio, 3),
            "total_peers": total_peers,
            "alive_count": len(alive_peers),
            "dead_count": len(dead_peers),
            "suspected_count": len(suspected_peers),
            "alive_peers": alive_peers[:10],  # Limit for response size
            "dead_peers": dead_peers[:10],
            "suspected_peers": suspected_peers[:10],
        }

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

        # Use .get() with defaults to prevent KeyError in case of race conditions
        # with _reset_gossip_metrics_hourly() (Dec 2025)
        if event == "sent":
            metrics["message_sent"] = metrics.get("message_sent", 0) + 1
        elif event == "received":
            metrics["message_received"] = metrics.get("message_received", 0) + 1
        elif event == "update":
            metrics["state_updates"] = metrics.get("state_updates", 0) + 1
        elif event == "anti_entropy":
            metrics["anti_entropy_repairs"] = metrics.get("anti_entropy_repairs", 0) + 1
        elif event == "stale":
            metrics["stale_states_detected"] = metrics.get("stale_states_detected", 0) + 1
        elif event == "latency":
            # Keep last 100 latency measurements
            delays = metrics.setdefault("propagation_delay_ms", [])
            delays.append(latency_ms)
            if len(delays) > 100:
                metrics["propagation_delay_ms"] = delays[-100:]

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
        self._ensure_state_attr("_gossip_compression_stats", {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "messages_compressed": 0,
        })
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

        Dec 28, 2025 (Phase 6): Now includes peer health tracking stats.
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

        # Dec 28, 2025 (Phase 6): Check for suspected peers via gossip failures
        health_tracker_stats: dict[str, Any] = {}
        if hasattr(self, "_gossip_health_tracker"):
            tracker = self._gossip_health_tracker
            health_tracker_stats = tracker.get_stats()
            suspected_count = health_tracker_stats.get("suspected_peers", 0)
            if suspected_count > 0:
                suspected_ids = health_tracker_stats.get("suspected_peer_ids", [])
                if suspected_count <= 3:
                    warnings.append(f"Gossip failures: {', '.join(suspected_ids)}")
                else:
                    warnings.append(f"Gossip failures: {suspected_count} peers unresponsive")

        return {
            "is_healthy": len(warnings) == 0,
            "warnings": warnings,
            "metrics": summary,
            "peer_health": health_tracker_stats,  # Phase 6
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for gossip protocol mixin (DaemonManager integration).

        December 2025: Added for unified health check interface.
        Uses base class helper for standardized response format.

        Returns:
            dict with healthy status, message, and details
        """
        status = self._get_gossip_health_status()
        is_healthy = status.get("is_healthy", False)
        warnings = status.get("warnings", [])
        message = "Gossip healthy" if is_healthy else f"Gossip issues: {', '.join(warnings)}"
        return self._build_health_response(is_healthy, message, status)


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

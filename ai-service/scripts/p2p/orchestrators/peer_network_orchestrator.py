"""Peer Network Orchestrator - Handles peer discovery and network management.

January 2026: Created as part of P2POrchestrator decomposition.

Responsibilities:
- Peer discovery and address resolution
- SWIM gossip integration (member alive/failed callbacks)
- Dynamic voter management
- Network partition detection
- Peer reputation and trust scoring
- Local IP caching
- Multi-transport address collection (SWIM/HTTP/SSH)
- Peer snapshot caching with TTL
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class PeerNetworkOrchestrator(BaseOrchestrator):
    """Orchestrator for peer network management.

    This orchestrator handles all aspects of peer networking in the P2P cluster:
    - Discovering and tracking peer nodes
    - SWIM gossip protocol integration
    - Voter management for quorum
    - Network partition detection
    - Peer health and reputation tracking

    Usage:
        # In P2POrchestrator.__init__:
        self.network = PeerNetworkOrchestrator(self)

        # Get cached peer snapshot:
        peers = self.network.get_cached_peer_snapshot()

        # Handle SWIM events:
        self.network.on_swim_member_alive(member_id)
    """

    # Cache TTL for peer snapshots
    DEFAULT_PEER_CACHE_TTL = 1.0  # seconds

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the peer network orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Peer snapshot cache
        self._peer_snapshot_cache: list | None = None
        self._peer_snapshot_cache_time: float = 0.0

        # Local IP cache (populated on first call)
        self._local_ips_cache: set[str] | None = None

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "peer_network"

    def health_check(self) -> HealthCheckResult:
        """Check the health of peer network orchestrator.

        Returns:
            HealthCheckResult with peer network status details.
        """
        try:
            issues = []

            # Get peer counts
            peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
            total_peers = 0
            alive_peers = 0

            if peer_snapshot is not None:
                snapshot = peer_snapshot.get_snapshot()
                total_peers = len(snapshot)
                alive_peers = sum(1 for p in snapshot.values() if p.is_alive())

            # Check for network issues
            if total_peers == 0:
                issues.append("No peers discovered")
            elif alive_peers == 0:
                issues.append("All peers unreachable")
            elif alive_peers < total_peers / 2:
                issues.append(f"Low peer availability: {alive_peers}/{total_peers}")

            # Check for partition
            if hasattr(self._p2p, "_detect_network_partition"):
                if self._p2p._detect_network_partition():
                    issues.append("Network partition detected")

            healthy = len(issues) == 0
            message = "Peer network healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "total_peers": total_peers,
                    "alive_peers": alive_peers,
                    "cache_age": time.time() - self._peer_snapshot_cache_time,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Peer Snapshot Management
    # =========================================================================

    def get_cached_peer_snapshot(self, max_age_seconds: float = 1.0) -> list:
        """Get cached peer snapshot to reduce lock acquisitions.

        Jan 29, 2026: Implementation moved from P2POrchestrator.

        Returns cached copy if < max_age_seconds old, otherwise takes new snapshot.

        Args:
            max_age_seconds: Maximum age of cached snapshot before refreshing (default 1.0s)

        Returns:
            List of peer NodeInfo objects (may be up to max_age_seconds stale)
        """
        now = time.time()

        if self._peer_snapshot_cache is not None and (now - self._peer_snapshot_cache_time) < max_age_seconds:
            return self._peer_snapshot_cache

        # Get fresh snapshot from P2P
        peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return []

        snapshot = list(peer_snapshot.get_snapshot().values())
        self._peer_snapshot_cache = snapshot
        self._peer_snapshot_cache_time = now
        return snapshot

    def get_peer_count(self) -> tuple[int, int]:
        """Get total and alive peer counts.

        Returns:
            Tuple of (total_peers, alive_peers)
        """
        peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return 0, 0

        snapshot = peer_snapshot.get_snapshot()
        total = len(snapshot)
        alive = sum(1 for p in snapshot.values() if p.is_alive())
        return total, alive

    # =========================================================================
    # SWIM Integration
    # =========================================================================

    def on_swim_member_alive(self, member_id: str) -> None:
        """Handle SWIM member becoming alive - sync to gossip layer.

        Jan 29, 2026: Implementation moved from P2POrchestrator.

        When SWIM detects a member is alive (via ping/ack or indirect probe),
        this callback syncs the state to the HTTP gossip layer by:
        1. Updating the peer's last_heartbeat timestamp
        2. Clearing any retired status

        Args:
            member_id: SWIM member identifier (usually "IP:7947")
        """
        self._log_info(f"[SWIM->Gossip] Member {member_id} ALIVE")

        # Extract host from member_id (format: "IP:7947")
        host = member_id.rsplit(":", 1)[0] if ":" in member_id else member_id

        # Get peers lock and peers dict from P2P
        peers_lock = getattr(self._p2p, "peers_lock", None)
        peers = getattr(self._p2p, "peers", {})

        if peers_lock is None:
            return

        with peers_lock:
            for node_id, peer in peers.items():
                peer_host = getattr(peer, "host", "")
                peer_tailscale = getattr(peer, "tailscale_ip", "")
                if peer_host == host or peer_tailscale == host:
                    peer.last_heartbeat = time.time()
                    if getattr(peer, "retired", False):
                        peer.retired = False
                        self._log_info(f"[SWIM->Gossip] Unretired peer {node_id} via SWIM alive")
                    break

    def on_swim_member_failed(self, member_id: str) -> None:
        """Handle SWIM member failure - mark as suspect in gossip layer.

        Jan 29, 2026: Implementation moved from P2POrchestrator.

        When SWIM detects a member has failed (via suspicion timeout),
        this callback records the failure in the health tracker.

        Note: This does NOT immediately retire the peer - the gossip layer
        uses its own timeout for final retirement decisions.

        Args:
            member_id: SWIM member identifier (usually "IP:7947")
        """
        self._log_warning(f"[SWIM->Gossip] Member {member_id} FAILED")

        host = member_id.rsplit(":", 1)[0] if ":" in member_id else member_id

        # Record failure in gossip health tracker if available
        gossip_health_tracker = getattr(self._p2p, "_gossip_health_tracker", None)
        if gossip_health_tracker is None:
            return

        peers_lock = getattr(self._p2p, "peers_lock", None)
        peers = getattr(self._p2p, "peers", {})

        if peers_lock is None:
            return

        with peers_lock:
            for node_id, peer in peers.items():
                peer_host = getattr(peer, "host", "")
                if peer_host == host:
                    gossip_health_tracker.record_gossip_failure(node_id)
                    self._log_info(f"[SWIM->Gossip] Recorded failure for {node_id}")
                    break

    # =========================================================================
    # Local IP Management
    # =========================================================================

    def get_local_ips(self) -> set[str]:
        """Get cached local IP addresses.

        Jan 29, 2026: Wrapper for _cache_local_ips() with caching.

        Returns:
            Set of local IP addresses.
        """
        if self._local_ips_cache is not None:
            return self._local_ips_cache

        # Delegate to P2P's implementation if available
        if hasattr(self._p2p, "_cache_local_ips"):
            self._local_ips_cache = self._p2p._cache_local_ips()
            return self._local_ips_cache

        # Fallback: minimal implementation
        return self._discover_local_ips()

    def _discover_local_ips(self) -> set[str]:
        """Discover local IP addresses.

        Returns:
            Set of local IP addresses.
        """
        import socket
        import subprocess

        local_ips: set[str] = set()

        # Method 1: Hostname resolution
        try:
            hostname = socket.gethostname()
            local_ips.add(socket.gethostbyname(hostname))
        except (socket.error, OSError):
            pass

        # Method 2: getaddrinfo for all local addresses
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                local_ips.add(info[4][0])
        except (socket.error, OSError):
            pass

        # Method 3: Connect to external address to find default route IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ips.add(s.getsockname()[0])
        except (socket.error, OSError):
            pass

        # Method 4: ifconfig/ip command parsing (Unix only)
        try:
            result = subprocess.run(
                ["ip", "addr"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "inet " in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            ip = parts[1].split("/")[0]
                            local_ips.add(ip)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        self._local_ips_cache = local_ips
        return local_ips

    def is_local_ip(self, ip: str) -> bool:
        """Check if an IP address is local to this node.

        Args:
            ip: IP address to check.

        Returns:
            True if the IP is local.
        """
        return ip in self.get_local_ips()

    # =========================================================================
    # Network Partition Detection
    # =========================================================================

    def detect_network_partition(self) -> bool:
        """Detect if we're in a network partition (>50% peers unreachable via primary IP).

        Jan 29, 2026: Implementation moved from P2POrchestrator._detect_network_partition().

        Used to trigger Tailscale-first connectivity mode when the public network
        is fragmented but mesh connectivity remains intact.

        Returns:
            True if partition detected (majority of peers unreachable)
        """
        # Get peer snapshot from P2P
        peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return False

        node_id = getattr(self._p2p, "node_id", "")
        peers_snapshot = [
            p for p in peer_snapshot.get_snapshot().values()
            if p.node_id != node_id
        ]

        if len(peers_snapshot) < 2:
            return False

        # Count peers with recent heartbeat failures
        # Use jittered timeout if available
        now = time.time()
        jittered_timeout = 90.0  # Default timeout
        if hasattr(self._p2p, "_get_cached_jittered_timeout"):
            jittered_timeout = self._p2p._get_cached_jittered_timeout()

        unreachable = 0
        for peer in peers_snapshot:
            if peer.consecutive_failures >= 3 or (now - peer.last_heartbeat > jittered_timeout):
                unreachable += 1

        partition_ratio = unreachable / len(peers_snapshot)
        if partition_ratio > 0.5:
            self._log_info(
                f"Network partition detected: {unreachable}/{len(peers_snapshot)} "
                f"peers unreachable ({partition_ratio:.0%})"
            )
            return True
        return False

    # =========================================================================
    # Voter Management
    # =========================================================================

    def get_eligible_voters(self) -> list[str]:
        """Get list of nodes eligible to be voters (GPU nodes with good health).

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_eligible_voters().

        Returns:
            List of node IDs that are eligible to vote.
        """
        import time

        # Import constants - do it here to avoid circular imports at module level
        try:
            from scripts.p2p.constants import (
                VOTER_PROMOTION_UPTIME,
                VOTER_DEMOTION_FAILURES,
            )
        except ImportError:
            VOTER_PROMOTION_UPTIME = 300  # 5 minutes default
            VOTER_DEMOTION_FAILURES = 5

        # Get peer snapshot from P2P
        peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return list(getattr(self._p2p, "voter_node_ids", []))

        peers = peer_snapshot.get_snapshot()
        eligible = []
        now = time.time()
        node_id = getattr(self._p2p, "node_id", "")

        for peer_id, peer in peers.items():
            # Skip retired or NAT-blocked without relay
            if getattr(peer, "retired", False):
                continue

            # Must be alive
            if not peer.is_alive():
                continue

            # Must have GPU (CUDA or MPS)
            has_gpu = getattr(peer, "has_gpu", False)
            gpu_name = str(getattr(peer, "gpu_name", "") or "")
            if not has_gpu and "GH200" not in gpu_name and "H100" not in gpu_name and "A10" not in gpu_name and "aws" not in peer_id.lower():
                continue

            # Must have been up for minimum time
            first_seen = getattr(peer, "first_seen", 0) or peer.last_heartbeat
            if now - first_seen < VOTER_PROMOTION_UPTIME:
                continue

            # Check health score (response rate)
            failures = getattr(peer, "consecutive_failures", 0)
            if failures >= VOTER_DEMOTION_FAILURES:
                continue

            eligible.append(peer_id)

        # Always include self if we have GPU
        if node_id and node_id not in eligible:
            self_gpu = getattr(self._p2p, "has_gpu", False)
            if self_gpu or "aws" in node_id.lower() or "lambda" in node_id.lower():
                eligible.append(node_id)

        return sorted(eligible)

    def get_voter_count(self) -> tuple[int, int]:
        """Get total and alive voter counts.

        Returns:
            Tuple of (total_voters, alive_voters)
        """
        voter_ids = getattr(self._p2p, "voter_node_ids", [])
        if not voter_ids:
            return 0, 0

        total = len(voter_ids)
        alive = 0

        if hasattr(self._p2p, "_count_alive_voters"):
            alive = self._p2p._count_alive_voters()
        else:
            # Fallback: count manually
            peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
            if peer_snapshot is not None:
                snapshot = peer_snapshot.get_snapshot()
                for voter_id in voter_ids:
                    peer = snapshot.get(voter_id)
                    if peer is not None and peer.is_alive():
                        alive += 1

        return total, alive

    def manage_dynamic_voters(self) -> bool:
        """Manage dynamic voter pool - promote/demote voters as needed.

        Jan 29, 2026: Implementation moved from P2POrchestrator._manage_dynamic_voters().

        Returns:
            True if voter set was changed.
        """
        import os

        # Import constants - do it here to avoid circular imports at module level
        try:
            from scripts.p2p.constants import (
                DYNAMIC_VOTER_ENABLED,
                DYNAMIC_VOTER_TARGET,
                VOTER_DEMOTION_FAILURES,
                VOTER_MIN_QUORUM,
            )
        except ImportError:
            self._log_warning("Cannot manage dynamic voters: constants not available")
            return False

        if not DYNAMIC_VOTER_ENABLED:
            return False

        # Don't override env-configured voters
        if (os.environ.get("RINGRIFT_P2P_VOTERS") or "").strip():
            return False

        current_voters = list(getattr(self._p2p, "voter_node_ids", []) or [])
        eligible = self.get_eligible_voters()

        # Get peer snapshot for health checks
        peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return False

        peers = peer_snapshot.get_snapshot()
        node_id = getattr(self._p2p, "node_id", "")

        healthy_voters = []
        unhealthy_voters = []

        for voter_id in current_voters:
            if voter_id == node_id:
                healthy_voters.append(voter_id)
                continue
            peer = peers.get(voter_id)
            if peer and peer.is_alive():
                failures = getattr(peer, "consecutive_failures", 0)
                if failures < VOTER_DEMOTION_FAILURES:
                    healthy_voters.append(voter_id)
                else:
                    unhealthy_voters.append(voter_id)
            else:
                unhealthy_voters.append(voter_id)

        changed = False
        new_voters = healthy_voters.copy()

        # Demote unhealthy voters
        if unhealthy_voters:
            self._log_info(f"Dynamic voters: demoting unhealthy voters: {unhealthy_voters}")
            changed = True

        # Promote new voters if below target
        if len(new_voters) < DYNAMIC_VOTER_TARGET:
            candidates = [n for n in eligible if n not in new_voters]
            # Sort by reliability (fewer failures = better)
            candidates.sort(
                key=lambda n: getattr(peers.get(n), "consecutive_failures", 0) if peers.get(n) else 999
            )

            needed = DYNAMIC_VOTER_TARGET - len(new_voters)
            for candidate in candidates[:needed]:
                new_voters.append(candidate)
                self._log_info(f"Dynamic voters: promoting {candidate} to voter")
                changed = True

        if changed and new_voters:
            new_voters = sorted(set(new_voters))
            self._p2p.voter_node_ids = new_voters
            # SIMPLIFIED QUORUM: Fixed at 3 voters (or less if fewer voters exist)
            self._p2p.voter_quorum_size = min(VOTER_MIN_QUORUM, len(new_voters))
            self._p2p.voter_config_source = "dynamic"
            print(
                f"[P2P] Dynamic voter set updated: {len(new_voters)} voters, "
                f"quorum={self._p2p.voter_quorum_size} ({', '.join(new_voters)})"
            )
            return True

        return False

    # =========================================================================
    # Address Collection
    # =========================================================================

    def collect_all_addresses(
        self,
        include_swim: bool = True,
        include_http: bool = True,
        include_ssh: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Collect all known addresses for peers across transports.

        Jan 29, 2026: Wrapper for P2POrchestrator's address collection.

        Args:
            include_swim: Include SWIM protocol addresses
            include_http: Include HTTP service addresses
            include_ssh: Include SSH transport addresses

        Returns:
            Dict mapping node_id to address info dict.
        """
        if hasattr(self._p2p, "_collect_all_addresses"):
            return self._p2p._collect_all_addresses(
                include_swim=include_swim,
                include_http=include_http,
                include_ssh=include_ssh,
            )
        return {}

    # =========================================================================
    # Host Online/Offline Events
    # =========================================================================

    def emit_host_online_sync(self, node_id: str, capabilities: list[str] | None = None) -> None:
        """Sync version of emit_host_online for non-async contexts.

        Jan 29, 2026: Implementation moved from P2POrchestrator._emit_host_online_sync().

        Args:
            node_id: The node ID that came online.
            capabilities: Optional list of capabilities for the node.
        """
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            # Use lock-free snapshot to prevent race conditions
            peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
            peer_info = peer_snapshot.get_snapshot().get(node_id) if peer_snapshot else None

            emit_event(DataEventType.HOST_ONLINE.value, {
                "node_id": node_id,
                "host": getattr(peer_info, "host", "") if peer_info else "",
                "port": getattr(peer_info, "port", 0) if peer_info else 0,
                "has_gpu": getattr(peer_info, "has_gpu", False) if peer_info else False,
                "gpu_name": getattr(peer_info, "gpu_name", "") if peer_info else "",
                "capabilities": capabilities or [],
                "source": "peer_recovery_sync",
            })
            self._log_debug(f"Emitted HOST_ONLINE (sync) for peer: {node_id}")
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            self._log_debug(f"Failed to emit HOST_ONLINE (sync) for {node_id}: {e}")

    # =========================================================================
    # SWIM/Raft Protocol Status
    # =========================================================================

    def get_swim_raft_status(self) -> dict[str, Any]:
        """Get SWIM/Raft protocol status summary.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_swim_raft_status().

        Returns status of the new P2P protocols (Phase 5, Dec 26, 2025):
        - SWIM: Gossip-based membership with 5s failure detection
        - Raft: Replicated work queue with sub-second failover

        These protocols are enabled via feature flags and provide
        gradual migration from HTTP/Bully to SWIM/Raft.

        Returns:
            Dict with protocol status details.
        """
        try:
            from scripts.p2p.constants import (
                SWIM_ENABLED, RAFT_ENABLED, MEMBERSHIP_MODE, CONSENSUS_MODE
            )
        except ImportError:
            SWIM_ENABLED = False
            RAFT_ENABLED = False
            MEMBERSHIP_MODE = "http"
            CONSENSUS_MODE = "bully"

        # Check SWIM status
        swim_status: dict[str, Any] = {
            "enabled": SWIM_ENABLED,
            "available": False,
            "started": getattr(self._p2p, "_swim_started", False),
            "alive_count": 0,
        }
        try:
            from app.p2p.swim_adapter import SWIM_AVAILABLE
            swim_status["available"] = SWIM_AVAILABLE

            # Get membership summary if SWIM is active
            if hasattr(self._p2p, "get_swim_membership_summary"):
                summary = self._p2p.get_swim_membership_summary()
                swim_status.update({
                    "started": summary.get("swim_started", False),
                    "alive_count": summary.get("swim", {}).get("alive", 0),
                    "suspected_count": summary.get("swim", {}).get("suspected", 0),
                    "failed_count": summary.get("swim", {}).get("failed", 0),
                })
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            swim_status["error"] = str(e)

        # Check Raft status
        raft_status: dict[str, Any] = {
            "enabled": RAFT_ENABLED,
            "available": False,
            "initialized": getattr(self._p2p, "_raft_initialized", False),
            "is_leader": False,
        }
        try:
            from scripts.p2p.consensus_mixin import PYSYNCOBJ_AVAILABLE
            raft_status["available"] = PYSYNCOBJ_AVAILABLE

            # Get Raft consensus status if available
            if hasattr(self._p2p, "get_raft_status"):
                raft_info = self._p2p.get_raft_status()
                raft_status.update({
                    "initialized": raft_info.get("raft_initialized", False),
                    "is_leader": raft_info.get("is_raft_leader", False),
                    "leader_address": raft_info.get("raft_leader", ""),
                    "should_use_raft": raft_info.get("should_use_raft", False),
                })
                if "work_queue_status" in raft_info:
                    wq = raft_info["work_queue_status"]
                    raft_status["work_queue"] = {
                        "pending": wq.get("by_status", {}).get("pending", 0),
                        "claimed": wq.get("by_status", {}).get("claimed", 0),
                        "completed": wq.get("by_status", {}).get("completed", 0),
                    }
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            raft_status["error"] = str(e)

        return {
            "membership_mode": MEMBERSHIP_MODE,
            "consensus_mode": CONSENSUS_MODE,
            "swim": swim_status,
            "raft": raft_status,
            "hybrid_status": {
                "swim_fallback_active": not swim_status.get("started", False) and MEMBERSHIP_MODE != "http",
                "raft_fallback_active": not raft_status.get("initialized", False) and CONSENSUS_MODE != "bully",
            },
        }

    # =========================================================================
    # Self Registration
    # =========================================================================

    def register_self_in_peers(self) -> None:
        """Register this node in the peers dict.

        Jan 29, 2026: Implementation moved from P2POrchestrator._register_self_in_peers().

        Jan 5, 2026: Ensures the leader (and any node) is visible in self.peers
        for components that iterate over peers directly. This fixes an issue
        where the leader was not in its own peers dict after becoming leader.

        This is idempotent - calling multiple times is safe.
        """
        import asyncio

        # Update self_info first
        if hasattr(self._p2p, "_update_self_info"):
            self._p2p._update_self_info()

        node_id = getattr(self._p2p, "node_id", "")
        self_info = getattr(self._p2p, "self_info", None)
        peers_lock = getattr(self._p2p, "peers_lock", None)
        peers = getattr(self._p2p, "peers", {})

        was_present = False
        if peers_lock is not None and self_info is not None:
            with peers_lock:
                was_present = node_id in peers
                peers[node_id] = self_info
                if not was_present:
                    self._log_info(f"Registered self in peers: {node_id}")

        # Sync to lock-free snapshot
        if hasattr(self._p2p, "_sync_peer_snapshot"):
            self._p2p._sync_peer_snapshot()

        # Emit HOST_ONLINE if this is first registration
        if not was_present:
            try:
                asyncio.create_task(self._emit_host_online_for_self())
            except RuntimeError:
                # No event loop running
                pass

    async def _emit_host_online_for_self(self) -> None:
        """Emit HOST_ONLINE event for self-registration."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            node_id = getattr(self._p2p, "node_id", "")
            self_info = getattr(self._p2p, "self_info", None)
            if self_info is None:
                return

            emit_event(DataEventType.HOST_ONLINE.value, {
                "node_id": node_id,
                "host": getattr(self_info, "host", ""),
                "port": getattr(self_info, "port", 0),
                "has_gpu": getattr(self_info, "has_gpu", False),
                "gpu_name": getattr(self_info, "gpu_name", ""),
                "capabilities": list(self_info.capabilities) if getattr(self_info, "capabilities", None) else [],
                "source": "leader_self_registration",
            })
            self._log_debug(f"Emitted HOST_ONLINE for self: {node_id}")
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            self._log_debug(f"Failed to emit HOST_ONLINE: {e}")

    # =========================================================================
    # Peer Deduplication
    # =========================================================================

    def deduplicate_peers(self) -> int:
        """Periodic deduplication of peers dict.

        Jan 29, 2026: Implementation moved from P2POrchestrator._deduplicate_peers().

        Dec 29, 2025: Scans for any duplicate entries that may have been
        created before deduplication was implemented, and merges them.

        Jan 13, 2026: Now actually removes duplicates instead of just logging.
        Keeps the most recently active node per IP (by last_heartbeat).

        Returns:
            Number of duplicates removed
        """
        removed = 0
        node_id = getattr(self._p2p, "node_id", "")
        peers_lock = getattr(self._p2p, "peers_lock", None)
        peers = getattr(self._p2p, "peers", {})

        if peers_lock is None:
            return 0

        with peers_lock:
            # Build IP -> node_id mapping to detect duplicates
            ip_to_nodes: dict[str, list[str]] = {}
            for peer_id, peer in peers.items():
                # Skip self
                if peer_id == node_id:
                    continue
                # Prefer reported_host (Tailscale IPv6) for dedup - most stable identifier
                dedup_key = getattr(peer, "reported_host", None) or getattr(peer, "effective_host", None) or getattr(peer, "host", None)
                if dedup_key and dedup_key not in ("127.0.0.1", ""):
                    ip_to_nodes.setdefault(dedup_key, []).append(peer_id)

            # Find IPs shared by multiple node_ids (duplicates)
            nodes_to_remove: set[str] = set()
            for ip, node_ids in ip_to_nodes.items():
                if len(node_ids) <= 1:
                    continue

                # Multiple nodes claim the same IP - keep the freshest one
                node_freshness = []
                for nid in node_ids:
                    peer = peers.get(nid)
                    if peer:
                        freshness = getattr(peer, "last_heartbeat", 0) or getattr(peer, "last_seen", 0) or 0
                        node_freshness.append((nid, freshness))

                if not node_freshness:
                    continue

                # Sort by freshness (most recent first)
                node_freshness.sort(key=lambda x: x[1], reverse=True)
                keeper = node_freshness[0][0]
                stale = [nid for nid, _ in node_freshness[1:]]

                if stale:
                    self._log_info(
                        f"IP {ip}: keeping {keeper} (freshest), "
                        f"removing {len(stale)} stale entries: {stale}"
                    )
                    nodes_to_remove.update(stale)

            # Actually remove the duplicate nodes
            for peer_id in nodes_to_remove:
                if peer_id in peers:
                    del peers[peer_id]
                    removed += 1
                    self._log_info(f"Removed duplicate peer: {peer_id}")

        if removed > 0:
            self._log_info(f"Removed {removed} duplicate peer entries")

        return removed

    # =========================================================================
    # Local IP Caching
    # =========================================================================

    def cache_local_ips(self) -> set[str]:
        """Cache all local IPs at startup to avoid DNS blocking in health endpoints.

        Jan 29, 2026: Implementation moved from P2POrchestrator._cache_local_ips().

        Jan 26, 2026: Called once at initialization and cached.
        This eliminates per-request DNS lookups that were blocking Lambda health endpoints.

        Returns:
            Set of local IP addresses.
        """
        import os
        import socket
        import subprocess

        local_ips: set[str] = set()

        # Method 1: Hostname resolution
        try:
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                local_ips.add(addr[4][0])
        except (socket.gaierror, socket.herror, OSError, UnicodeError):
            pass

        # Method 2: Localhost variations
        try:
            for addr in socket.getaddrinfo("localhost", None):
                local_ips.add(addr[4][0])
        except (socket.gaierror, socket.herror, OSError):
            pass

        # Method 3: Subprocess fallback for containers
        try:
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for ip in result.stdout.strip().split():
                    local_ips.add(ip)
        except Exception:  # noqa: BLE001
            pass

        # Method 4: Add advertise_host if configured
        advertise_host = os.environ.get("RINGRIFT_ADVERTISE_HOST", "").strip()
        if advertise_host:
            local_ips.add(advertise_host)

        # Method 5: Add Tailscale IP from environment
        ts_ip = os.environ.get("TAILSCALE_IP", "").strip()
        if ts_ip:
            local_ips.add(ts_ip)

        return local_ips

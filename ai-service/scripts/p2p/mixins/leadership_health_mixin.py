"""Leadership Health Mixin - Voter health and quorum monitoring.

January 2026: Extracted from p2p_orchestrator.py to reduce file size.

This mixin provides voter/quorum health tracking functionality:
- _build_voter_ip_mapping(): Map voter node IDs to their known IPs
- _count_alive_voters(): Count alive voters for quorum calculation
- _is_peer_alive(): Check if a peer dict is alive
- _is_swim_peer_id(): Check if peer_id is SWIM protocol format
- _is_self_voter(): Comprehensive self-recognition for voter matching
- _check_voter_health(): Full voter health status with alerts
- _check_leader_health(): Leader health based on peer response rates
- update_fence_token_from_leader(): Fence token synchronization
- _log_cluster_health_snapshot(): Detailed cluster health logging

Usage:
    class P2POrchestrator(LeadershipHealthMixin, ...):
        pass

Dependencies on parent class attributes:
    - voter_node_ids: list[str]
    - voter_quorum_size: int
    - peers: dict[str, NodeInfo]
    - peers_lock: threading.RLock
    - node_id: str
    - leader_id: str | None
    - role: NodeRole
    - ringrift_path: str
    - _peer_snapshot: PeerSnapshot
    - _startup_time: float
    - advertise_host: str | None
    - host: str | None
    - _cached_local_ips: set[str]
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from scripts.p2p.models import NodeRole

logger = logging.getLogger(__name__)

# Constants - imported or defined locally
try:
    from scripts.p2p.constants import (
        LEADER_MIN_RESPONSE_RATE,
        LEADER_DEGRADED_STEPDOWN_DELAY,
        STARTUP_GRACE_PERIOD,
    )
except ImportError:
    LEADER_MIN_RESPONSE_RATE = 0.5
    LEADER_DEGRADED_STEPDOWN_DELAY = 60.0
    STARTUP_GRACE_PERIOD = 30.0


class LeadershipHealthMixin(P2PMixinBase):
    """Mixin providing voter health and quorum monitoring for P2P orchestrator.

    This mixin centralizes all voter/quorum health logic including:
    - Voter IP mapping and matching
    - Alive voter counting with multiple detection methods
    - Voter health status reporting
    - Leader health monitoring
    - Fence token synchronization

    Inherits from P2PMixinBase for shared helper methods.
    """

    MIXIN_TYPE = "leadership_health"

    # Type hints for parent class attributes
    voter_node_ids: list[str]
    voter_quorum_size: int
    node_id: str
    leader_id: str | None
    role: Any  # NodeRole
    ringrift_path: str
    advertise_host: str | None
    host: str | None
    _startup_time: float
    _cached_local_ips: set[str]
    _last_seen_epoch: int
    _leader_degraded_since: float

    def _build_voter_ip_mapping(self) -> dict[str, set[str]]:
        """Build a mapping from voter node_ids to their known IPs.

        Jan 2, 2026: Added to support voter matching when peers are discovered
        via SWIM as IP:port format instead of proper node_ids.

        Returns:
            Dict mapping voter node_id -> set of known IPs (tailscale_ip, ssh_host)
        """
        voter_ids = getattr(self, "voter_node_ids", []) or []
        if not voter_ids:
            return {}

        # Load config to get IP mappings
        # Jan 2, 2026: Handle ringrift_path that already includes ai-service suffix
        rp = Path(self.ringrift_path)
        if rp.name == "ai-service":
            cfg_path = rp / "config" / "distributed_hosts.yaml"
        else:
            cfg_path = rp / "ai-service" / "config" / "distributed_hosts.yaml"
        if not cfg_path.exists():
            return {}

        try:
            import yaml
            data = yaml.safe_load(cfg_path.read_text()) or {}
            hosts = data.get("hosts", {}) or {}
        except Exception:
            return {}

        voter_ip_map: dict[str, set[str]] = {}
        for voter_id in voter_ids:
            host_cfg = hosts.get(voter_id, {})
            ips: set[str] = set()

            # Jan 13, 2026: Add BOTH Tailscale IP and ssh_host to voter IP set.
            if host_cfg.get("tailscale_ip"):
                ips.add(host_cfg["tailscale_ip"])
            # Also add ssh_host if it's a valid IP (not a hostname)
            if host_cfg.get("ssh_host"):
                ssh_host = host_cfg["ssh_host"]
                if ssh_host and not any(c.isalpha() for c in ssh_host.replace(".", "")):
                    ips.add(ssh_host)

            if ips:
                voter_ip_map[voter_id] = ips

        return voter_ip_map

    def _is_peer_alive(self, peer_info: dict) -> bool:
        """Check if a peer (as dict) is alive based on its status field."""
        if isinstance(peer_info, dict):
            status = peer_info.get("status", "unknown")
            return status in ("alive", "healthy", "connected")
        return False

    def _is_swim_peer_id(self, peer_id: str) -> bool:
        """Check if peer_id is a SWIM protocol entry (IP:7947 format).

        SWIM entries use port 7947 and should not be in the HTTP peer list.
        These leak from the SWIM membership layer and cause peer pollution.

        Args:
            peer_id: Node identifier to check.

        Returns:
            True if this is a SWIM-format peer ID (should be skipped).
        """
        if not peer_id or ":" not in peer_id:
            return False
        parts = peer_id.rsplit(":", 1)
        if len(parts) == 2 and parts[1] == "7947":
            return True
        return False

    def _is_self_voter(self, voter_id: str, voter_ips: set[str]) -> bool:
        """Check if we are this voter using multiple identification methods.

        Jan 22, 2026: Comprehensive self-recognition for Lambda nodes and other
        environments where node_id differs from configured voter names.

        Jan 26, 2026: Now uses cached local IPs (self._cached_local_ips) to avoid
        DNS blocking in health endpoints. DNS lookups are done once at startup.

        Args:
            voter_id: The voter ID to check against ourselves.
            voter_ips: Set of IPs associated with this voter.

        Returns:
            True if we are this voter.
        """
        # Method 1: Direct node_id match
        if voter_id == self.node_id:
            return True

        # Method 2: Host IP match
        self_host = getattr(self, "advertise_host", None) or getattr(self, "host", None)
        if self_host and self_host in voter_ips:
            logger.debug(f"[VoterSelfRecognition] Matched via host IP: {self_host}")
            return True

        # Method 3: Use cached local IPs (Jan 26, 2026: avoids DNS blocking)
        local_ips = getattr(self, "_cached_local_ips", set())
        if voter_ips & local_ips:
            matching = voter_ips & local_ips
            logger.debug(f"[VoterSelfRecognition] Matched via cached local IP: {matching}")
            return True

        # Method 4: Tailscale IP from environment
        ts_ip = os.environ.get("TAILSCALE_IP")
        if ts_ip and ts_ip in voter_ips:
            logger.debug(f"[VoterSelfRecognition] Matched via TAILSCALE_IP env: {ts_ip}")
            return True

        # Method 5: Check Tailscale CGNAT range IPs from cached local IPs
        try:
            from app.p2p.config import TAILSCALE_CGNAT_NETWORK
            import ipaddress

            for ip in local_ips:
                try:
                    if ipaddress.ip_address(ip) in TAILSCALE_CGNAT_NETWORK:
                        if ip in voter_ips:
                            logger.debug(f"[VoterSelfRecognition] Matched via Tailscale CGNAT: {ip}")
                            return True
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            logger.debug(f"[VoterSelfRecognition] Tailscale CGNAT check failed: {e}")

        return False

    def _count_alive_voters(self) -> int:
        """Count alive voters by checking both node_id and IP:port matches.

        Jan 2, 2026: Added because SWIM discovers peers as IP:port format
        (e.g., "135.181.39.239:7947") but voter_node_ids are proper names
        (e.g., "hetzner-cpu1"). This method checks both.

        Returns:
            Number of alive voters (including self if we are a voter)
        """
        voter_ids = getattr(self, "voter_node_ids", []) or []
        if not voter_ids:
            return 0

        alive_count = 0

        # Build voter IP mapping
        voter_ip_map = self._build_voter_ip_mapping()

        # Build reverse map: IP -> voter_id for quick lookup
        ip_to_voter: dict[str, str] = {}
        for voter_id, ips in voter_ip_map.items():
            for ip in ips:
                ip_to_voter[ip] = voter_id

        # Track which voters we've counted to avoid double-counting
        counted_voters: set[str] = set()

        # Check each voter
        for voter_id in voter_ids:
            if voter_id in counted_voters:
                continue

            # Check 1: Is this voter us?
            voter_ips = voter_ip_map.get(voter_id, set())
            if self._is_self_voter(voter_id, voter_ips):
                alive_count += 1
                counted_voters.add(voter_id)
                logger.info(f"[VoterCount] Self recognized as voter: {voter_id}")
                continue

            # Check 2: Direct node_id match in peers
            peer = self.peers.get(voter_id)
            if peer and peer.is_alive():
                alive_count += 1
                counted_voters.add(voter_id)
                continue

            # Check 3: Peer info 'host' field match
            voter_ips = voter_ip_map.get(voter_id, set())
            for peer_id, peer in self.peers.items():
                if voter_id in counted_voters:
                    break
                if self._is_swim_peer_id(peer_id):
                    continue
                if isinstance(peer, dict):
                    peer_host = peer.get("host", "")
                    if peer_host in voter_ips and self._is_peer_alive(peer):
                        alive_count += 1
                        counted_voters.add(voter_id)
                        break
                elif hasattr(peer, "host") and peer.host in voter_ips:
                    if peer.is_alive():
                        alive_count += 1
                        counted_voters.add(voter_id)
                        break

            if voter_id in counted_voters:
                continue

            # Check 4: Multi-address matching
            for peer_id, peer in self.peers.items():
                if voter_id in counted_voters:
                    break
                if self._is_swim_peer_id(peer_id):
                    continue

                peer_addresses: set[str] = set()
                if isinstance(peer, dict):
                    peer_addresses.update(peer.get("addresses", []))
                    if peer.get("tailscale_ip"):
                        peer_addresses.add(peer["tailscale_ip"])
                    if peer.get("host"):
                        peer_addresses.add(peer["host"])
                elif hasattr(peer, "addresses"):
                    peer_addresses.update(getattr(peer, "addresses", []) or [])
                    if getattr(peer, "tailscale_ip", None):
                        peer_addresses.add(peer.tailscale_ip)
                    if getattr(peer, "host", None):
                        peer_addresses.add(peer.host)

                if peer_addresses & voter_ips:
                    is_alive = (
                        peer.is_alive()
                        if hasattr(peer, "is_alive")
                        else self._is_peer_alive(peer)
                    )
                    if is_alive:
                        alive_count += 1
                        counted_voters.add(voter_id)
                        break

            if voter_id in counted_voters:
                continue

            # Check 5: IP:port extraction from peer_id
            for peer_id, peer in self.peers.items():
                if voter_id in counted_voters:
                    break
                if self._is_swim_peer_id(peer_id):
                    continue
                if ":" in peer_id:
                    peer_ip = peer_id.rsplit(":", 1)[0]
                    if peer_ip in voter_ips:
                        is_alive = (
                            peer.is_alive()
                            if hasattr(peer, "is_alive")
                            else self._is_peer_alive(peer)
                        )
                        if is_alive:
                            alive_count += 1
                            counted_voters.add(voter_id)
                            break

            if voter_id in counted_voters:
                continue

            # Check 6: SWIM-discovered voters
            for peer_id, peer in self.peers.items():
                if voter_id in counted_voters:
                    break
                if not self._is_swim_peer_id(peer_id):
                    continue
                swim_ip = peer_id.rsplit(":", 1)[0]
                if swim_ip in voter_ips:
                    is_alive = (
                        peer.is_alive()
                        if hasattr(peer, "is_alive")
                        else self._is_peer_alive(peer)
                    )
                    if is_alive:
                        alive_count += 1
                        counted_voters.add(voter_id)
                        logger.info(
                            f"[VoterSwim] Counted voter {voter_id} via SWIM IP {swim_ip}"
                        )
                        break

        return alive_count

    def _check_voter_health(self) -> dict[str, Any]:
        """Check health status of all configured voters and emit alerts.

        Returns a dict with voter health status including:
        - voters_total, voters_alive, voters_offline
        - quorum_size, quorum_ok, quorum_threatened
        - voter_status: per-voter status dict

        Returns:
            Dict with full voter health status.
        """
        voter_ids = getattr(self, "voter_node_ids", []) or []
        if not voter_ids:
            return {
                "voters_total": 0,
                "voters_alive": 0,
                "voters_offline": [],
                "quorum_size": 0,
                "quorum_ok": True,
                "quorum_threatened": False,
                "voter_status": {},
            }

        quorum_size = getattr(self, "voter_quorum_size", 0) or 0
        voter_ip_map = self._build_voter_ip_mapping()

        voter_status: dict[str, dict] = {}
        alive_voters: list[str] = []
        offline_voters: list[str] = []

        prev_offline = set(getattr(self, "_last_offline_voters", []))
        self_host = getattr(self, "advertise_host", None) or getattr(self, "host", None)

        for voter_id in voter_ids:
            is_alive = False
            status_detail = "unknown"

            # Check 1a: Is this voter us? (by node_id)
            if voter_id == self.node_id:
                is_alive = True
                status_detail = "self"
            # Check 1b: Is this voter us? (by IP)
            elif self_host and self_host in voter_ip_map.get(voter_id, set()):
                is_alive = True
                status_detail = "self_via_ip"
            else:
                # Check 2: Direct node_id match in peers
                peer = self.peers.get(voter_id)
                if peer and peer.is_alive():
                    is_alive = True
                    status_detail = "direct"
                else:
                    # Check 3: Peer host or IP:port match
                    voter_ips = voter_ip_map.get(voter_id, set())
                    for peer_id, p in self.peers.items():
                        if self._is_swim_peer_id(peer_id):
                            continue
                        peer_host = getattr(p, "host", None) or (p.get("host") if isinstance(p, dict) else None)
                        if peer_host and peer_host in voter_ips and p.is_alive():
                            is_alive = True
                            status_detail = f"host_match:{peer_host}"
                            break
                        if ":" in peer_id:
                            peer_ip = peer_id.split(":")[0]
                            if peer_ip in voter_ips and p.is_alive():
                                is_alive = True
                                status_detail = f"ip_match:{peer_ip}"
                                break
                    if not is_alive:
                        status_detail = "unreachable"

            voter_status[voter_id] = {
                "alive": is_alive,
                "detail": status_detail,
            }
            if is_alive:
                alive_voters.append(voter_id)
            else:
                offline_voters.append(voter_id)

        self._last_offline_voters = offline_voters

        voters_alive = len(alive_voters)
        voters_total = len(voter_ids)
        quorum_ok = voters_alive >= quorum_size
        quorum_threatened = voters_alive <= quorum_size

        # Emit events for status changes
        newly_offline = set(offline_voters) - prev_offline
        newly_online = prev_offline - set(offline_voters)

        in_grace_period = (time.time() - getattr(self, "_startup_time", 0)) < STARTUP_GRACE_PERIOD

        for voter_id in newly_offline:
            if in_grace_period:
                logger.debug(
                    f"[VoterHealth] Voter {voter_id} appears offline during startup grace period"
                )
            else:
                logger.warning(
                    f"[VoterHealth] Voter {voter_id} went OFFLINE "
                    f"({voters_alive}/{voters_total} alive, quorum={quorum_size})"
                )

        for voter_id in newly_online:
            logger.info(
                f"[VoterHealth] Voter {voter_id} came ONLINE "
                f"({voters_alive}/{voters_total} alive, quorum={quorum_size})"
            )

        if offline_voters and not in_grace_period:
            logger.warning(
                f"[VoterHealth] Status: {voters_alive}/{voters_total} voters alive, "
                f"quorum={'OK' if quorum_ok else 'LOST'}, "
                f"offline: {offline_voters}"
            )

        return {
            "voters_total": voters_total,
            "voters_alive": voters_alive,
            "voters_offline": offline_voters,
            "quorum_size": quorum_size,
            "quorum_ok": quorum_ok,
            "quorum_threatened": quorum_threatened,
            "voter_status": voter_status,
        }

    def _check_leader_health(self) -> bool:
        """Check leader health based on peer response rates.

        Returns True if health is good, False if degraded.
        """
        # Import NodeRole here to avoid circular imports
        try:
            from scripts.p2p.models import NodeRole
        except ImportError:
            # Fallback if models not available
            class NodeRole:
                LEADER = "leader"

        if self.role != NodeRole.LEADER:
            return True

        # Use lock-free PeerSnapshot for read-only access
        peers = list(self._peer_snapshot.get_snapshot().values())

        if not peers:
            return True

        now = time.time()
        alive_count = sum(1 for p in peers if p.is_alive() and not getattr(p, "retired", False))
        total_count = sum(1 for p in peers if not getattr(p, "retired", False))

        if total_count == 0:
            return True

        response_rate = alive_count / total_count

        # Track degraded state
        if not hasattr(self, "_leader_degraded_since"):
            self._leader_degraded_since = 0.0

        if response_rate < LEADER_MIN_RESPONSE_RATE:
            if self._leader_degraded_since == 0.0:
                self._leader_degraded_since = now
                logger.info(
                    f"Leader health degraded: {response_rate:.1%} response rate "
                    f"(min: {LEADER_MIN_RESPONSE_RATE:.0%})"
                )
            elif now - self._leader_degraded_since > LEADER_DEGRADED_STEPDOWN_DELAY:
                logger.info(
                    f"Leader health critically degraded for {LEADER_DEGRADED_STEPDOWN_DELAY}s, "
                    "stepping down"
                )
                self._leader_degraded_since = 0.0
                return False
        else:
            if self._leader_degraded_since > 0:
                logger.info(f"Leader health recovered: {response_rate:.1%} response rate")
            self._leader_degraded_since = 0.0

        return True

    def update_fence_token_from_leader(self, token: str, leader_id: str) -> bool:
        """Update internal fence token state when receiving leader announcement.

        Args:
            token: Fence token from leader
            leader_id: Leader node ID

        Returns:
            True if update was successful
        """
        if not token:
            return False

        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False

            token_epoch = int(parts[1])

            # Update last seen epoch (but never decrease)
            if hasattr(self, "_last_seen_epoch"):
                if token_epoch >= self._last_seen_epoch:
                    self._last_seen_epoch = token_epoch
                    return True
                else:
                    logger.warning(
                        f"Rejecting fence token with stale epoch {token_epoch} "
                        f"(current: {self._last_seen_epoch}) from {leader_id}"
                    )
                    return False
            else:
                self._last_seen_epoch = token_epoch
                return True

        except (ValueError, IndexError):
            return False

    def _log_cluster_health_snapshot(self) -> None:
        """Log detailed cluster health for debugging.

        Provides real-time visibility into peer discovery and voter health.
        Called periodically from a background loop.
        """
        try:
            # Count alive peers (excluding SWIM entries)
            alive_peers = 0
            total_peers = 0
            swim_peers = 0
            for peer_id, peer in self.peers.items():
                total_peers += 1
                if self._is_swim_peer_id(peer_id):
                    swim_peers += 1
                    continue
                if hasattr(peer, "is_alive") and peer.is_alive():
                    alive_peers += 1
                elif isinstance(peer, dict) and self._is_peer_alive(peer):
                    alive_peers += 1

            # Count voters
            voters_alive = self._count_alive_voters()
            voters_total = len(getattr(self, "voter_node_ids", []) or [])

            # Get uptime
            uptime = time.time() - getattr(self, "_startup_time", time.time())

            snapshot = {
                "timestamp": time.time(),
                "node_id": self.node_id,
                "leader_id": self.leader_id,
                "is_leader": self.leader_id == self.node_id,
                "alive_peers": alive_peers,
                "total_peers": total_peers,
                "swim_peers": swim_peers,
                "voters_alive": voters_alive,
                "voters_total": voters_total,
                "quorum_ok": voters_alive >= getattr(self, "voter_quorum_size", 0),
                "election_in_progress": getattr(self, "election_in_progress", False),
                "uptime_seconds": int(uptime),
            }
            logger.info(f"[ClusterHealth] {json.dumps(snapshot)}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ClusterHealth] Snapshot failed: {e}")

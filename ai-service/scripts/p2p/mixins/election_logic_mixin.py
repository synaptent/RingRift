"""Election Logic Mixin - leadership election and lease coordination.

April 2026: Extracted from p2p_orchestrator.py (Target 2 of P2P decomposition).

This mixin contains the election and leadership methods that were still embedded
in the monolith after startup/bootstrap and training pipeline extraction:
- _endpoint_key(), _endpoint_conflict_keys(), _is_leader_eligible()
- _acquire_voter_lease_quorum(), _determine_leased_leader_from_voters()
- _query_arbiter_for_leader()
- _start_election(), _become_leader()
- _check_probabilistic_leadership(), _claim_provisional_leadership()
- _check_provisional_promotion(), _promote_provisional_to_leader()
- _step_down_from_provisional(), _request_election_from_voters()
- _check_emergency_coordinator_fallback()
- _renew_leader_lease()

This is a medium-high risk extraction because these methods coordinate the
leadership state machine, `leader_state_lock`, and `_election_lock`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp

from app.core.async_context import fire_and_forget
from scripts.p2p.constants import (
    ARBITER_URL,
    DEFAULT_PORT,
    ELECTION_PARTICIPATION_DELAY,
    ELECTION_TIMEOUT,
    LEADER_LEASE_DURATION,
    LEADER_LEASE_RENEW_INTERVAL,
    MAX_CONSECUTIVE_FAILURES,
    PROVISIONAL_LEADER_CHECK_INTERVAL,
    PROVISIONAL_LEADER_INITIAL_PROBABILITY,
    PROVISIONAL_LEADER_MAX_PROBABILITY,
    PROVISIONAL_LEADER_MIN_LEADERLESS_TIME,
    PROVISIONAL_LEADER_PROBABILITY_GROWTH_RATE,
    PROVISIONAL_LEADER_QUORUM_TIMEOUT,
    VOTER_MIN_QUORUM,
)
from scripts.p2p.leadership_state_machine import TransitionReason
from scripts.p2p.models import NodeInfo
from scripts.p2p.network import ClientTimeout, get_client_session
from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.types import NodeRole
from scripts.p2p.utils import safe_json_response

if TYPE_CHECKING:
    import threading

logger = logging.getLogger(__name__)


class ElectionLogicMixin(P2PMixinBase):
    """Mixin providing election, provisional leadership, and lease logic."""

    MIXIN_TYPE = "election_logic"

    role: Any  # NodeRole
    node_id: str
    leader_id: str | None
    leader_lease_id: str
    leader_lease_expires: float
    last_lease_renewal: float
    last_leader_seen: float
    voter_node_ids: list[str]
    voter_quorum_size: int
    voter_grant_leader_id: str
    voter_grant_lease_id: str
    voter_grant_expires: float
    known_peers: list[str]
    peers: dict[str, Any]
    self_info: Any
    recovery_manager: Any
    leadership: Any
    quorum_manager: Any
    _peer_snapshot: Any
    _leadership_sm: Any
    leader_state_lock: Any  # threading.RLock
    _election_lock: Any  # asyncio.Lock
    election_in_progress: bool
    _lease_epoch: int
    _fence_token: str
    _forced_leader_override: bool
    _preferred_leader_id: str | None
    _provisional_claim_probability: float
    _last_provisional_check: float
    _provisional_leader_claimed_at: float
    _provisional_leader_acks: set[str]
    _provisional_leader_challengers: dict[str, float]
    _fallback_leader_since: float
    _fallback_leader_reason: str
    _last_become_leader_time: float
    _last_election_completed: float
    _last_emergency_coord_check: float
    _quorum_missing_since: float
    _startup_time: float

    def _endpoint_key(self, info: NodeInfo) -> tuple[str, str, int] | None:
        """Return the normalized reachable endpoint key for a peer."""
        host = str(getattr(info, "host", "") or "").strip()
        if not host:
            return None
        scheme = str(getattr(info, "scheme", "http") or "http").lower()
        try:
            port = int(getattr(info, "port", DEFAULT_PORT) or DEFAULT_PORT)
        except ValueError:
            port = DEFAULT_PORT
        reported_host = str(getattr(info, "reported_host", "") or "").strip()
        try:
            reported_port = int(getattr(info, "reported_port", 0) or 0)
        except ValueError:
            reported_port = 0

        if reported_host and reported_port > 0:
            if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"} or self._is_tailscale_host(reported_host):
                host, port = reported_host, reported_port
        return (scheme, host, port)

    def _endpoint_conflict_keys(self, peers: list[NodeInfo]) -> set[tuple[str, str, int]]:
        """Compute endpoint keys shared by more than one live node."""
        counts: dict[tuple[str, str, int], int] = {}
        for peer in peers:
            if not peer.is_alive():
                continue
            key = self._endpoint_key(peer)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        return {key for key, count in counts.items() if count > 1}

    def _is_leader_eligible(
        self,
        peer: NodeInfo,
        conflict_keys: set[tuple[str, str, int]],
        *,
        require_alive: bool = True,
    ) -> bool:
        """Heuristic: leaders must be directly reachable and uniquely addressable."""
        if require_alive and not peer.is_alive():
            return False
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if voters and peer.node_id not in voters:
            return False
        if int(getattr(peer, "consecutive_failures", 0) or 0) >= MAX_CONSECUTIVE_FAILURES:
            return False
        if getattr(peer, "nat_blocked", False):
            return False
        if getattr(peer, "force_relay_mode", False):
            return False
        node_status = getattr(peer, "status", "")
        if node_status == "proxy_only" or self._is_node_proxy_only(peer.node_id):
            return False
        if self.recovery_manager.compute_connectivity_score(peer) < 0.3:
            return False
        key = self._endpoint_key(peer)
        return not (key and key in conflict_keys)

    async def _acquire_voter_lease_quorum(self, lease_id: str, duration: int) -> float | None:
        """Acquire or renew an exclusive leader lease from a quorum of voters."""
        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_ids:
            return time.time() + float(duration)

        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            quorum = min(VOTER_MIN_QUORUM, len(voter_ids))

        duration = max(10, min(int(duration), int(LEADER_LEASE_DURATION * 2)))
        max_retries = 3
        retry_delays = [0, 2, 5]

        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep(retry_delays[attempt])
                logger.info("Voter lease acquisition retry %s/%s", attempt + 1, max_retries)

            now = time.time()
            acks = 0
            lease_ttls: list[float] = []

            if self.node_id in voter_ids:
                self.voter_grant_leader_id = self.node_id
                self.voter_grant_lease_id = lease_id
                self.voter_grant_expires = now + float(duration)
                lease_ttls.append(float(duration))
                acks += 1

            peers_by_id = self._peer_snapshot.get_snapshot()
            timeout = ClientTimeout(total=15)

            async def _request_lease_from_voter(
                session: aiohttp.ClientSession,
                voter_id: str,
                voter: NodeInfo,
            ) -> tuple[bool, float | None]:
                payload = {
                    "leader_id": self.node_id,
                    "lease_id": lease_id,
                    "lease_duration": duration,
                    "lease_epoch": self._lease_epoch + 1,
                }
                for url in self._tailscale_urls_for_voter(voter, "/election/lease"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data, json_error = await safe_json_response(resp, default={}, log_errors=False)
                            if json_error or not data.get("granted"):
                                return False, None
                            ttl_raw = data.get("lease_ttl_seconds") or data.get("ttl_seconds")
                            if ttl_raw is not None:
                                try:
                                    return True, float(ttl_raw)
                                except (ValueError, TypeError):
                                    pass
                            return True, float(duration)
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, AttributeError, OSError):
                        continue
                return False, None

            async with get_client_session(timeout) as session:
                voter_tasks = []
                for voter_id in voter_ids:
                    if voter_id == self.node_id:
                        continue
                    voter = peers_by_id.get(voter_id)
                    if not voter or not voter.is_alive():
                        continue
                    voter_tasks.append(_request_lease_from_voter(session, voter_id, voter))

                if voter_tasks:
                    results = await asyncio.gather(*voter_tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            continue
                        success, ttl = result
                        if success:
                            acks += 1
                            lease_ttls.append(ttl if ttl is not None and ttl > 0 else float(duration))

            if acks >= quorum:
                effective_ttl = min(lease_ttls) if lease_ttls else float(duration)
                effective_ttl = max(10.0, min(float(duration), float(effective_ttl)))
                if attempt > 0:
                    logger.info("Voter lease acquired on retry %s", attempt + 1)
                return now + float(effective_ttl)

            if attempt < max_retries - 1:
                logger.warning(
                    "Voter lease quorum not reached: %s/%s acks, retrying in %ss...",
                    acks,
                    quorum,
                    retry_delays[attempt + 1],
                )

        logger.error("Failed to acquire voter lease quorum after %s attempts", max_retries)
        return None

    async def _determine_leased_leader_from_voters(self) -> str | None:
        """Return the current lease-holder as reported by a quorum of voters."""
        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_ids:
            return None

        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            quorum = min(VOTER_MIN_QUORUM, len(voter_ids))

        now = time.time()
        counts: dict[str, int] = {}

        if self.node_id in voter_ids:
            leader_id = str(getattr(self, "voter_grant_leader_id", "") or "")
            expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
            if leader_id and expires > now:
                counts[leader_id] = counts.get(leader_id, 0) + 1

        peers_by_id = self._peer_snapshot.get_snapshot()
        timeout = ClientTimeout(total=15)
        async with get_client_session(timeout) as session:
            for voter_id in voter_ids:
                if voter_id == self.node_id:
                    continue
                voter = peers_by_id.get(voter_id)
                if not voter or not voter.is_alive():
                    continue
                for url in self._tailscale_urls_for_voter(voter, "/election/grant"):
                    try:
                        async with session.get(url, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                        leader_id = str((data or {}).get("leader_id") or "")
                        if not leader_id:
                            break
                        ttl_raw = (data or {}).get("lease_ttl_seconds")
                        if ttl_raw is None:
                            ttl_raw = (data or {}).get("ttl_seconds")
                        ttl_val: float | None = None
                        if ttl_raw is not None:
                            try:
                                ttl_val = float(ttl_raw)
                            except ValueError:
                                ttl_val = None

                        if ttl_val is not None:
                            if ttl_val <= 0:
                                break
                        else:
                            expires = float((data or {}).get("lease_expires") or 0.0)
                            if expires <= 0:
                                break
                            if expires + float(LEADER_LEASE_DURATION) < now:
                                break
                        counts[leader_id] = counts.get(leader_id, 0) + 1
                        break
                    except (ValueError, AttributeError):
                        continue

        winners = [leader_id for leader_id, count in counts.items() if count >= quorum]
        if not winners:
            return None
        return sorted(winners)[-1]

    async def _query_arbiter_for_leader(self) -> str | None:
        """Query the arbiter for the authoritative leader when voter quorum fails."""
        arbiter_url = ARBITER_URL
        if not arbiter_url:
            return None

        urls_to_try = [arbiter_url]
        for peer_addr in (self.known_peers or []):
            if peer_addr not in urls_to_try:
                urls_to_try.append(peer_addr)

        timeout = ClientTimeout(total=5)
        try:
            async with get_client_session(timeout) as session:
                for url in urls_to_try:
                    try:
                        base_url = url.rstrip("/")
                        async with session.get(f"{base_url}/election/grant", headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                leader_id = str((data or {}).get("leader_id") or "")
                                if leader_id:
                                    logger.info("Arbiter %s reports leader: %s", base_url, leader_id)
                                    return leader_id
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
            pass
        return None

    async def _start_election(self):
        """Start leader election using Bully algorithm."""
        _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
        if not _is_coordinator:
            logger.info("[Election] Skipping: non-coordinator node, will adopt leader via gossip")
            return

        grace_until = getattr(self, "_election_grace_until", 0) or 0
        if time.time() < grace_until:
            logger.info(
                "[Election] Skipping: election grace period active (%ss remaining)",
                f"{grace_until - time.time():.0f}",
            )
            return

        preferred = getattr(self, "_preferred_leader_id", None)
        if preferred and preferred != self.node_id:
            preferred_info = self.peers.get(preferred)
            if preferred_info:
                last_seen = getattr(preferred_info, "last_seen", 0) or 0
                age = time.time() - last_seen
                if age < 120:
                    logger.info(
                        "[Election] Suppressing: preferred leader '%s' alive (%ss ago, threshold 120s)",
                        preferred,
                        f"{age:.0f}",
                    )
                    return

        self._start_election_timing()

        elapsed = time.time() - getattr(self, "_startup_time", 0)
        if elapsed < ELECTION_PARTICIPATION_DELAY:
            logger.info(
                "[Election] Skipping: still in startup grace (%ss < %ss)",
                f"{elapsed:.0f}",
                ELECTION_PARTICIPATION_DELAY,
            )
            return

        election_global_cooldown = 30.0
        now = time.time()
        last_election = getattr(self, "_last_election_completed", 0.0)
        if now - last_election < election_global_cooldown:
            logger.debug(
                "[Election] Skipping: global cooldown (%.1fs < %.1fs)",
                now - last_election,
                election_global_cooldown,
            )
            return

        import random

        await asyncio.sleep(random.uniform(0.5, 3.0))
        if self.leader_id and self.leader_id != self.node_id:
            logger.debug("[Election] Skipping after jitter: leader %s emerged", self.leader_id)
            return

        await asyncio.to_thread(self._update_self_info)
        if getattr(self.self_info, "nat_blocked", False):
            return

        voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
        if voter_node_ids:
            if self.node_id not in voter_node_ids:
                await self._request_election_from_voters("non_voter_detected_leaderless")
                return
            try:
                from scripts.p2p.leader_election import should_block_election

                snapshot = self._peer_snapshot.get_snapshot()
                should_block, reason = should_block_election(voter_node_ids, snapshot, self.node_id)
                if should_block:
                    logger.warning("[Election] Blocked: %s", reason)
                    self._safe_emit_event(
                        "ELECTION_BLOCKED",
                        {
                            "node_id": self.node_id,
                            "reason": reason,
                            "voter_count": len(voter_node_ids),
                            "timestamp": time.time(),
                        },
                    )
                    return
            except ImportError:
                if not self._has_voter_quorum():
                    return

        snapshot = self._peer_snapshot.get_snapshot()
        peers_snapshot = [peer for peer in snapshot.values() if peer.node_id != self.node_id]
        conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])

        if self.leader_id and self.leader_id != self.node_id:
            leader = snapshot.get(self.leader_id)
            leader_ok = (
                leader is not None
                and leader.is_alive()
                and leader.role == NodeRole.LEADER
                and self._is_leader_eligible(leader, conflict_keys)
                and self._is_leader_lease_valid()
            )
            if leader_ok:
                return
            self._set_leader(None, reason="stale_ineligible_leader", save_state=False)
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
        if self._maybe_adopt_leader_from_peers():
            return

        async with self._election_lock:
            if self.election_in_progress:
                logger.debug("[Election] Already in progress, skipping")
                return
            self.election_in_progress = True

        if self.leadership.was_recently_leader() and self.leadership.in_incumbent_grace_period():
            logger.info(
                "Incumbent advantage: attempting immediate leadership reclaim (stepped down %.1fs ago)",
                time.time() - self._last_step_down_time,
            )
            with self.leader_state_lock:
                self.role = NodeRole.CANDIDATE
            try:
                conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
                if self._is_leader_eligible(self.self_info, conflict_keys):
                    await self._become_leader()
                    if self.role == NodeRole.LEADER:
                        logger.info("Incumbent reclaimed leadership successfully")
                        return
            finally:
                with self.leader_state_lock:
                    if self.role == NodeRole.CANDIDATE:
                        self.role = NodeRole.FOLLOWER
            logger.info("Incumbent reclaim failed, falling back to normal election")

        with self.leader_state_lock:
            self.role = NodeRole.CANDIDATE
        logger.info("Starting election, my ID: %s", self.node_id)

        try:
            election_snapshot = self._peer_snapshot.get_snapshot()
            higher_nodes = [
                peer
                for peer in election_snapshot.values()
                if peer.node_id > self.node_id and self._is_leader_eligible(peer, conflict_keys)
            ]
            if voter_node_ids:
                higher_nodes = [peer for peer in higher_nodes if peer.node_id in voter_node_ids]

            got_response = False
            timeout = ClientTimeout(total=ELECTION_TIMEOUT)
            async with get_client_session(timeout) as session:
                for peer in higher_nodes:
                    try:
                        url = self._url_for_peer(peer, "/election")
                        async with session.post(url, json={"candidate_id": self.node_id}, headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("response") == "ALIVE":
                                    got_response = True
                                    logger.info("Higher node %s responded", peer.node_id)
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        pass

            if not got_response:
                if self._is_leader_eligible(self.self_info, conflict_keys):
                    await self._become_leader()
            else:
                await asyncio.sleep(ELECTION_TIMEOUT * 2)
                self._maybe_adopt_leader_from_peers()
        finally:
            self.election_in_progress = False
            with self.leader_state_lock:
                if self.role == NodeRole.CANDIDATE:
                    if getattr(self, "_election_started_at", 0) > 0:
                        self._record_election_latency("timeout")
                    self.role = NodeRole.FOLLOWER

    async def _become_leader(self):
        """Become the cluster leader with lease-based leadership."""
        await asyncio.to_thread(self._update_self_info)
        if getattr(self.self_info, "nat_blocked", False):
            logger.info("Refusing leadership while NAT-blocked: %s", self.node_id)
            return
        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            logger.info("Refusing leadership without voter quorum: %s", self.node_id)
            return

        lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        lease_expires = await self._acquire_voter_lease_quorum(lease_id, int(LEADER_LEASE_DURATION))
        if getattr(self, "voter_node_ids", []) and not lease_expires:
            logger.error("Failed to obtain voter lease quorum; refusing leadership: %s", self.node_id)
            self._record_election_latency("lost")
            self._set_leader(None, reason="election_failed_no_quorum", save_state=False)
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()
            self._last_election_completed = time.time()
            self._save_state()
            return

        logger.info("I am now the leader: %s", self.node_id)
        self._set_leader(self.node_id, reason="become_leader", save_state=False)
        self.last_leader_seen = time.time()
        self._register_self_in_peers()
        self._record_election_latency("won")
        self._last_become_leader_time = time.time()
        if hasattr(self, "_leadership_sm") and self._leadership_sm:
            self._leadership_sm.quorum_health.reset()

        self._increment_cluster_epoch()
        self._lease_epoch += 1
        self._fence_token = f"{self.node_id}:{self._lease_epoch}:{time.time()}"
        logger.info("Leader lease fencing: epoch=%s, token=%s", self._lease_epoch, self._fence_token)

        fire_and_forget(
            self._emit_leader_elected(self.node_id, getattr(self, "cluster_epoch", 0)),
            name=f"emit_leader_elected:{self.node_id}:{self._lease_epoch}",
        )

        self.leader_lease_id = lease_id
        self.leader_lease_expires = float(lease_expires or (time.time() + LEADER_LEASE_DURATION))
        self.last_lease_renewal = time.time()

        peers = self.get_peers_list_ro()
        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(
                            url,
                            json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                                "lease_epoch": self._lease_epoch,
                                "fence_token": self._fence_token,
                            },
                            headers=self._auth_headers(),
                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, AttributeError):
                        pass

        self._last_election_completed = time.time()
        self._save_state()
        await self._start_monitoring_if_leader()
        await self._start_p2p_auto_deployer()

    async def _check_probabilistic_leadership(self, now: float) -> None:
        """Claim provisional leadership after an extended leaderless period."""
        import random

        if self.leader_id or self.role in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER, NodeRole.CANDIDATE):
            return
        if now - self._last_provisional_check < PROVISIONAL_LEADER_CHECK_INTERVAL:
            return
        self._last_provisional_check = now

        leaderless_duration = now - self.last_leader_seen
        if leaderless_duration < PROVISIONAL_LEADER_MIN_LEADERLESS_TIME:
            return

        await asyncio.to_thread(self._update_self_info)
        if getattr(self.self_info, "nat_blocked", False):
            logger.debug("Skipping probabilistic leadership: NAT-blocked")
            return

        minutes_beyond_minimum = (leaderless_duration - PROVISIONAL_LEADER_MIN_LEADERLESS_TIME) / 60.0
        current_prob = min(
            PROVISIONAL_LEADER_MAX_PROBABILITY,
            PROVISIONAL_LEADER_INITIAL_PROBABILITY
            * (PROVISIONAL_LEADER_PROBABILITY_GROWTH_RATE ** minutes_beyond_minimum),
        )
        self._provisional_claim_probability = current_prob

        roll = random.random()
        logger.debug(
            "Probabilistic leadership check: roll=%.3f, threshold=%.3f, leaderless=%ss",
            roll,
            current_prob,
            int(leaderless_duration),
        )
        if roll >= current_prob:
            return

        logger.info(
            "Claiming provisional leadership after %ss leaderless (prob=%.2f%%, roll=%.3f)",
            int(leaderless_duration),
            current_prob * 100.0,
            roll,
        )
        await self._claim_provisional_leadership()

    async def _claim_provisional_leadership(self) -> None:
        """Claim provisional leadership and announce to peers."""
        import uuid as _uuid

        now = time.time()
        with self.leader_state_lock:
            self.role = NodeRole.PROVISIONAL_LEADER
            self._provisional_leader_claimed_at = now
            self._provisional_leader_acks = {self.node_id}
            self._provisional_leader_challengers = {}

            provisional_lease_id = f"PROVISIONAL_{self.node_id}_{_uuid.uuid4().hex[:8]}"
            self.leader_lease_id = provisional_lease_id
            self.leader_lease_expires = now + PROVISIONAL_LEADER_QUORUM_TIMEOUT
            self.last_lease_renewal = now
            self.leader_id = self.node_id

        logger.info("Provisional leadership claimed: lease=%s", provisional_lease_id)
        peers = [peer for peer in self.get_peers_list_ro() if peer.node_id != self.node_id and peer.is_alive()]

        if not peers:
            logger.info("No alive peers to acknowledge, promoting immediately to full leader")
            await self._promote_provisional_to_leader("no_peers")
            return

        timeout = aiohttp.ClientTimeout(total=5)
        acks_received = 0
        challengers: list[str] = []

        async with get_client_session(timeout) as session:
            for peer in peers:
                try:
                    url = self._url_for_peer(peer, "/provisional-leader/claim")
                    async with session.post(
                        url,
                        json={"claimant_id": self.node_id, "lease_id": provisional_lease_id, "claimed_at": now},
                        headers=self._auth_headers(),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ack"):
                                self._provisional_leader_acks.add(peer.node_id)
                                acks_received += 1
                                logger.debug("Provisional ack from %s", peer.node_id)
                            elif data.get("challenge"):
                                challenger_id = data.get("challenger_id", peer.node_id)
                                self._provisional_leader_challengers[challenger_id] = now
                                challengers.append(challenger_id)
                                logger.info("Provisional challenge from %s", challenger_id)
                except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                    pass

        logger.info("Provisional claim results: %s acks, %s challengers", acks_received, len(challengers))

        if challengers:
            all_claimants = [self.node_id] + challengers
            all_claimants.sort(reverse=True)
            winner = all_claimants[0]
            if winner != self.node_id:
                logger.info("Stepping down from provisional: %s > %s", winner, self.node_id)
                self._step_down_from_provisional()
                return
            logger.info("Won provisional tiebreaker against %s", challengers)

        total_peers = len(peers) + 1
        quorum_size = (total_peers // 2) + 1
        current_acks = len(self._provisional_leader_acks)

        if current_acks >= quorum_size:
            logger.info("Quorum achieved (%s/%s), promoting to full leader", current_acks, quorum_size)
            await self._promote_provisional_to_leader("quorum_achieved")
        else:
            logger.info("Quorum not yet achieved (%s/%s), waiting for timeout", current_acks, quorum_size)
            asyncio.get_running_loop().call_later(
                PROVISIONAL_LEADER_QUORUM_TIMEOUT,
                lambda: fire_and_forget(
                    self._check_provisional_promotion(),
                    name=f"check_provisional_promotion:{self.node_id}",
                ),
            )

    async def _check_provisional_promotion(self) -> None:
        """Promote or step down after the provisional leadership timeout."""
        if self.role != NodeRole.PROVISIONAL_LEADER:
            return

        now = time.time()
        claim_duration = now - self._provisional_leader_claimed_at
        if claim_duration < PROVISIONAL_LEADER_QUORUM_TIMEOUT:
            return

        if self._provisional_leader_challengers:
            all_claimants = [self.node_id] + list(self._provisional_leader_challengers.keys())
            all_claimants.sort(reverse=True)
            winner = all_claimants[0]
            if winner != self.node_id:
                logger.info("Challenger %s won during timeout period", winner)
                self._step_down_from_provisional()
                return

        logger.info("Provisional timeout elapsed with no successful challengers, promoting to full leader")
        await self._promote_provisional_to_leader("timeout_no_challengers")

    async def _promote_provisional_to_leader(self, reason: str) -> None:
        """Promote from provisional to full leader."""
        if self.role == NodeRole.LEADER:
            return

        logger.info("Promoting from provisional to full leader: %s", reason)
        self._provisional_leader_claimed_at = 0.0
        self._provisional_leader_acks.clear()
        self._provisional_leader_challengers.clear()

        now = time.time()
        self._set_leader(self.node_id, reason=f"promote_provisional_{reason}", save_state=False)
        self.last_leader_seen = now
        self._register_self_in_peers()

        lease_id = f"FALLBACK_{self.node_id}_{int(now)}_{uuid.uuid4().hex[:8]}"
        self.leader_lease_id = lease_id
        self.leader_lease_expires = now + LEADER_LEASE_DURATION
        self.last_lease_renewal = now
        self._fallback_leader_since = now
        self._fallback_leader_reason = reason

        self._increment_cluster_epoch()
        self._lease_epoch += 1
        self._fence_token = f"{self.node_id}:{self._lease_epoch}:{now}"

        fire_and_forget(
            self._emit_leader_elected(self.node_id, getattr(self, "cluster_epoch", 0)),
            name=f"emit_leader_elected:{self.node_id}:{self._lease_epoch}",
        )

        peers = self.get_peers_list_ro()
        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(
                            url,
                            json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "fallback_leadership": True,
                                "lease_epoch": self._lease_epoch,
                                "fence_token": self._fence_token,
                            },
                            headers=self._auth_headers(),
                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        pass

        self._save_state()
        await self._start_monitoring_if_leader()
        await self._start_p2p_auto_deployer()
        logger.info("Full fallback leadership established: lease=%s", lease_id)

    def _step_down_from_provisional(self) -> None:
        """Step down from provisional leadership."""
        return self.leadership.step_down_from_provisional()

    async def _request_election_from_voters(self, reason: str = "non_voter_request") -> bool:
        """Allow non-voters to request that voters start an election."""
        return await self.leadership.request_election_from_voters(reason)

    async def _check_emergency_coordinator_fallback(self):
        """Allow emergency coordinator mode when voter quorum is unreachable for too long."""
        _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
        if not _is_coordinator:
            return

        now = time.time()
        last_check = getattr(self, "_last_emergency_coord_check", 0)
        if now - last_check < 60:
            return
        self._last_emergency_coord_check = now

        if self.role == NodeRole.LEADER:
            return
        if self.leader_id:
            self._emergency_coordinator_since = 0
            return
        if self._has_voter_quorum():
            self._emergency_coordinator_since = 0
            return

        quorum_missing_since = getattr(self, "_quorum_missing_since", 0)
        if quorum_missing_since == 0:
            self._quorum_missing_since = now
            return

        emergency_threshold = 300
        quorum_missing_duration = now - quorum_missing_since
        if quorum_missing_duration < emergency_threshold:
            return

        await asyncio.to_thread(self._update_self_info)
        if not getattr(self.self_info, "has_gpu", False):
            return
        if getattr(self.self_info, "nat_blocked", False):
            return

        candidates = [self.node_id]
        for peer in self.get_peers_list_ro():
            if not peer.is_alive():
                continue
            if not getattr(peer, "has_gpu", False):
                continue
            if getattr(peer, "nat_blocked", False):
                continue
            candidates.append(peer.node_id)

        if not candidates:
            return
        candidates.sort(reverse=True)
        designated_coordinator = candidates[0]
        if designated_coordinator != self.node_id:
            return

        logger.info(
            "EMERGENCY COORDINATOR: Taking leadership without voter quorum (quorum missing for %ss, %s candidates)",
            int(quorum_missing_duration),
            len(candidates),
        )
        self._set_leader(self.node_id, reason="emergency_coordinator", save_state=False)
        self.last_leader_seen = now
        self._emergency_coordinator_since = now

        self.leader_lease_id = f"EMERGENCY_{self.node_id}_{uuid.uuid4().hex[:8]}"
        self.leader_lease_expires = now + 120
        self.last_lease_renewal = now

        peers = self.get_peers_list_ro()
        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(
                            url,
                            json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "emergency": True,
                            },
                            headers=self._auth_headers(),
                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        pass

        self._save_state()
        logger.info("EMERGENCY COORDINATOR: %s is now emergency leader", self.node_id)

    async def _renew_leader_lease(self):
        """Renew our leadership lease and broadcast to peers."""
        if self.role != NodeRole.LEADER:
            return

        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            voters_alive = self._count_alive_voters()
            quorum_size = getattr(self, "voter_quorum_size", 0)
            if getattr(self, "_forced_leader_override", False):
                logger.debug(
                    "[LeaseRenewal] Voter quorum check failed (voters_alive=%s, quorum_size=%s) but forced leader override active; continuing",
                    voters_alive,
                    quorum_size,
                )
            else:
                threshold_exceeded = self._leadership_sm.quorum_health.record_failure(voters_alive)
                fail_count = self._leadership_sm.quorum_health.consecutive_failures
                threshold = self._leadership_sm.quorum_health.failure_threshold
                logger.warning(
                    "[LeaseRenewal] Voter quorum check failed (%s/%s): voters_alive=%s, quorum_size=%s",
                    fail_count,
                    threshold,
                    voters_alive,
                    quorum_size,
                )
                if threshold_exceeded:
                    logger.info(
                        "Lost voter quorum (%s consecutive failures via ULSM); stepping down: %s",
                        threshold,
                        self.node_id,
                    )
                    self._schedule_step_down_sync(TransitionReason.QUORUM_LOST)
                    self._release_voter_grant_if_self()
                return
        else:
            voters_alive = self._count_alive_voters()
            self._leadership_sm.quorum_health.record_success(voters_alive)

        now = time.time()
        if now - self.last_lease_renewal < LEADER_LEASE_RENEW_INTERVAL:
            return

        lease_id = str(self.leader_lease_id or "")
        if not lease_id:
            lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        lease_expires = await self._acquire_voter_lease_quorum(lease_id, int(LEADER_LEASE_DURATION))
        if getattr(self, "voter_node_ids", []) and not lease_expires:
            if getattr(self, "_forced_leader_override", False):
                logger.info("Voter lease quorum failed but forced leader override active; self-renewing lease")
                lease_expires = now + LEADER_LEASE_DURATION
            else:
                logger.info("Voter lease quorum failed; checking arbiter...")
                arbiter_leader = await self._query_arbiter_for_leader()
                if arbiter_leader == self.node_id:
                    logger.info("Arbiter confirms us as leader despite quorum failure; continuing with provisional lease")
                    lease_expires = now + LEADER_LEASE_DURATION / 2
                elif arbiter_leader:
                    if getattr(self, "_forced_leader_override", False):
                        logger.warning("Arbiter reports %s but forced leader override active; ignoring", arbiter_leader)
                    else:
                        logger.info("Arbiter reports different leader (%s); stepping down", arbiter_leader)
                        self._set_leader(arbiter_leader, reason="arbiter_override", save_state=False)
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self.last_lease_renewal = 0.0
                    self._release_voter_grant_if_self()
                    self._save_state()
                    return
                else:
                    if getattr(self, "_forced_leader_override", False):
                        logger.warning("Arbiter unreachable but forced leader override active; maintaining leadership: %s", self.node_id)
                        return
                    logger.error("Failed to renew voter lease quorum and arbiter unreachable; stepping down: %s", self.node_id)
                    self._set_leader(None, reason="arbiter_unreachable", save_state=False)
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self.last_lease_renewal = 0.0
                    self._release_voter_grant_if_self()
                    self._save_state()
                    return

        self.leader_lease_id = lease_id
        self.leader_lease_expires = float(lease_expires or (now + LEADER_LEASE_DURATION))
        self.last_lease_renewal = now

        if hasattr(self, "_leadership_sm") and self._leadership_sm:
            try:
                self._leadership_sm.renew_self_leadership()
            except Exception as exc:
                logger.debug("[LeaseRenewal] Failed to renew self-leadership: %s", exc)

        peers = self.get_peers_list_ro()
        timeout = ClientTimeout(total=3)
        try:
            async with get_client_session(timeout) as session:
                for peer in peers:
                    if peer.node_id != self.node_id and peer.is_alive():
                        try:
                            url = self._url_for_peer(peer, "/coordinator")
                            await session.post(
                                url,
                                json={
                                    "leader_id": self.node_id,
                                    "lease_id": self.leader_lease_id,
                                    "lease_expires": self.leader_lease_expires,
                                    "lease_renewal": True,
                                    "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                                },
                                headers=self._auth_headers(),
                            )
                        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, AttributeError):
                            pass
        except Exception as exc:  # noqa: BLE001
            logger.info("Lease renewal error: %s", exc)

"""Heartbeat Loop Mixin - Heartbeat loop and bootstrap methods.

April 2026: Extracted from p2p_orchestrator.py (Target 4 of P2P decomposition).

This mixin provides the core heartbeat loop and peer bootstrap functionality:
- _heartbeat_loop(): Main heartbeat loop (~428 LOC), the single largest method
- _send_heartbeat_to_peer(): Send heartbeat to a peer (delegates to HeartbeatManager)
- _send_heartbeat_via_ssh_fallback(): SSH fallback for heartbeats
- _bootstrap_from_known_peers(): Import cluster membership from seed peers
- _bootstrap_from_multiple_seeds(): Try multiple seeds to join cluster
- _load_bootstrap_seeds_from_config(): Load seeds from distributed_hosts.yaml
- _send_initial_relay_heartbeats(): NAT-blocked node relay registration at startup
- _send_startup_peer_announcements(): Immediate announcements to peers on startup
- _send_voter_heartbeat(): Heartbeat to voter peer (delegates to HeartbeatManager)
- _try_voter_alternative_endpoints(): Try Tailscale/reported endpoints for voter
- _discover_voter_peer(): Discover voter peer from known peers
- _refresh_voter_mesh(): Ensure all voters have knowledge of each other

Usage:
    class P2POrchestrator(HeartbeatLoopMixin, ...):
        pass

Dependencies on parent class attributes:
    - running: bool
    - node_id: str
    - known_peers: list[str]
    - relay_peers: list[str]
    - peers: dict[str, NodeInfo]
    - peers_lock: threading.RLock
    - role: NodeRole
    - leader_id: str | None
    - leader_lease_id: str
    - leader_lease_expires: float
    - self_info: NodeInfo
    - heartbeat_manager: HeartbeatManager
    - network: NetworkOrchestrator
    - leadership: LeadershipOrchestrator
    - ip_discovery_manager: IpDiscoveryManager
    - quorum_manager: QuorumManager
    - voter_node_ids: list[str]
    - voter_config_source: str
    - voter_quorum_size: int
    - _peer_snapshot: PeerSnapshot
    - _force_relay_mode: bool
    - _cluster_epoch: int
    - last_relay_heartbeat: float
    - relay_command_attempts: dict
    - verbose: bool

Dependencies on parent class methods:
    - _parse_peer_address()
    - _get_bootstrap_peers_by_reputation()
    - _update_peer_reputation()
    - _save_peer_to_cache()
    - _publish_peers_snapshot()
    - _sync_peer_snapshot()
    - _set_leader()
    - _save_cluster_epoch()
    - _save_state()
    - _send_relay_heartbeat()
    - _emit_host_online()
    - _endpoint_conflict_keys()
    - _endpoint_key()
    - _is_leader_eligible()
    - _is_leader_lease_valid()
    - _get_leader_peer()
    - _get_tailscale_ip_for_peer()
    - _check_dead_peers_async()
    - _check_leader_health()
    - _has_voter_quorum()
    - _release_voter_grant_if_self()
    - _renew_leader_lease()
    - _stop_monitoring_if_not_leader()
    - _start_monitoring_if_leader()
    - _stop_p2p_auto_deployer()
    - _start_p2p_auto_deployer()
    - _update_self_info_async()
    - get_peers_list_ro()
    - _auth_headers()
    - _peer_query
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import aiohttp

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)


class HeartbeatLoopMixin(P2PMixinBase):
    """Mixin providing heartbeat loop and bootstrap logic for P2P orchestrator.

    This mixin contains the main heartbeat loop (the single largest method in the
    orchestrator at ~428 LOC) and all bootstrap/peer-announcement methods.

    Inherits from P2PMixinBase for shared helper methods.
    """

    MIXIN_TYPE = "heartbeat_loop"

    # Type hints for parent class attributes accessed by these methods
    running: bool
    node_id: str
    known_peers: list[str]
    relay_peers: list[str]
    peers: dict[str, Any]
    peers_lock: Any  # threading.RLock
    role: Any  # NodeRole
    leader_id: str | None
    leader_lease_id: str
    leader_lease_expires: float
    self_info: Any  # NodeInfo
    heartbeat_manager: Any  # HeartbeatManager
    network: Any  # NetworkOrchestrator
    leadership: Any  # LeadershipOrchestrator
    ip_discovery_manager: Any  # IpDiscoveryManager
    quorum_manager: Any  # QuorumManager
    voter_node_ids: list[str]
    voter_config_source: str
    voter_quorum_size: int
    _peer_snapshot: Any  # PeerSnapshot
    _force_relay_mode: bool
    _cluster_epoch: int
    last_relay_heartbeat: float
    relay_command_attempts: dict
    verbose: bool
    _auto_deployer_task: Any

    async def _send_heartbeat_to_peer(self, peer_host: str, peer_port: int, scheme: str = "http", timeout: int = 15) -> NodeInfo | None:
        """Send heartbeat to a peer and return their info.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.send_heartbeat_to_peer(peer_host, peer_port, scheme, timeout)

    async def _send_heartbeat_via_ssh_fallback(
        self, peer_host: str, peer_port: int, payload: dict[str, Any]
    ) -> NodeInfo | None:
        """Send heartbeat via SSH when HTTP fails.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager._send_heartbeat_via_ssh_fallback(peer_host, peer_port, payload)

    async def _bootstrap_from_known_peers(self) -> bool:
        """Import cluster membership from seed peers via `/relay/peers`.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.bootstrap_from_known_peers()

    async def _bootstrap_from_multiple_seeds(self) -> bool:
        """Phase 26.3: Try multiple seeds until we join the cluster.

        Priority order:
        1. Cached peers with high reputation (from peer_cache table)
        2. CLI --peers (self.known_peers)
        3. Hardcoded BOOTSTRAP_SEEDS

        Returns True if we successfully connected to any peer.
        """
        # Imports from startup_infrastructure (available via wildcard import in orchestrator)
        from scripts.p2p.constants import DEFAULT_PORT, MIN_BOOTSTRAP_ATTEMPTS, VOTER_MIN_QUORUM
        from scripts.p2p.models import NodeInfo, NodeRole
        from scripts.p2p.network import NonBlockingAsyncLockWrapper, get_client_session

        # Build seed list with priority ordering
        all_seeds: list[str] = []
        seen: set[str] = set()

        # 1. First, try cached peers by reputation (if available)
        cached_peers = self._get_bootstrap_peers_by_reputation(limit=3)
        for seed in cached_peers:
            if seed and seed not in seen:
                seen.add(seed)
                all_seeds.append(seed)

        # 2. Then, CLI peers and hardcoded seeds (already merged in self.known_peers)
        for seed in self.known_peers:
            if seed and seed not in seen:
                seen.add(seed)
                all_seeds.append(seed)

        if not all_seeds:
            logger.warning("No bootstrap seeds available")
            return False

        # Limit attempts per cycle
        max_attempts = min(MIN_BOOTSTRAP_ATTEMPTS * 2, len(all_seeds))
        timeout = aiohttp.ClientTimeout(total=10)
        success = False

        async with get_client_session(timeout) as session:
            for idx, seed_addr in enumerate(all_seeds[:max_attempts]):
                try:
                    scheme, host, port = self._parse_peer_address(seed_addr)
                    scheme = (scheme or "http").lower()
                    url = f"{scheme}://{host}:{port}/relay/peers"

                    async with session.get(url, headers=self._auth_headers()) as resp:
                        if resp.status != 200:
                            self._update_peer_reputation(seed_addr, success=False)
                            continue

                        data = await resp.json()
                        if not isinstance(data, dict) or not data.get("success"):
                            self._update_peer_reputation(seed_addr, success=False)
                            continue

                    # Successfully got peer list
                    self._update_peer_reputation(seed_addr, success=True)
                    success = True

                    # Import peers
                    peers_data = data.get("peers") or {}
                    if isinstance(peers_data, dict):
                        with self.peers_lock:
                            for node_id, peer_dict in peers_data.items():
                                if node_id and node_id != self.node_id:
                                    try:
                                        info = NodeInfo.from_dict(peer_dict)
                                        self.peers[info.node_id] = info
                                        # Cache the peer for future restarts
                                        self._save_peer_to_cache(
                                            info.node_id,
                                            str(getattr(info, "host", "") or ""),
                                            int(getattr(info, "port", DEFAULT_PORT) or DEFAULT_PORT),
                                            str(getattr(info, "tailscale_ip", "") or "")
                                        )
                                    except (ValueError, KeyError, IndexError, AttributeError):
                                        continue
                            self._publish_peers_snapshot()

                        # Jan 12, 2026: Sync to lock-free snapshot after relay peer import
                        self._sync_peer_snapshot()

                    # Adopt leader if provided
                    leader_id = str(data.get("leader_id") or "").strip()
                    if leader_id and leader_id != self.node_id:
                        if self.role == NodeRole.LEADER:
                            logger.info(f"Stepping down for discovered leader: {leader_id}")
                        # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                        self._set_leader(leader_id, reason="continuous_bootstrap_discover_leader", save_state=False)

                    # Handle cluster epoch (Phase 29)
                    incoming_epoch = data.get("cluster_epoch")
                    if incoming_epoch is not None:
                        try:
                            epoch = int(incoming_epoch)
                            if epoch > self._cluster_epoch:
                                logger.info(f"Adopting higher cluster epoch: {epoch} (was {self._cluster_epoch})")
                                self._cluster_epoch = epoch
                                self._save_cluster_epoch()
                        except (ValueError, TypeError):
                            pass

                    # Import voter config if provided
                    incoming_voters = data.get("voter_node_ids") or data.get("voters")
                    if incoming_voters:
                        voters_list = []
                        if isinstance(incoming_voters, list):
                            voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                        elif isinstance(incoming_voters, str):
                            voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                        if voters_list:
                            if self.quorum_manager.maybe_adopt_voter_node_ids(voters_list, source="learned"):
                                # Sync adopted state back to orchestrator attributes
                                self.voter_node_ids = self.quorum_manager.voter_node_ids
                                self.voter_config_source = self.quorum_manager.voter_config_source
                                self.voter_quorum_size = min(VOTER_MIN_QUORUM, len(self.voter_node_ids)) if self.voter_node_ids else 0

                    self._save_state()
                    logger.info(f"Bootstrap from {host}:{port}: imported {len(peers_data)} peers")
                    break  # Success, no need to try more seeds

                except asyncio.TimeoutError:
                    self._update_peer_reputation(seed_addr, success=False)
                    continue
                except Exception as e:  # noqa: BLE001
                    self._update_peer_reputation(seed_addr, success=False)
                    if self.verbose:
                        logger.debug(f"Bootstrap seed {seed_addr} failed: {e}")
                    continue

        return success

    def _load_bootstrap_seeds_from_config(self) -> list[str]:
        """Load bootstrap seed peers from distributed_hosts.yaml.

        Selects stable coordinator and voter nodes as default seeds when no --peers provided.
        This enables automatic peer discovery via Tailscale even when CLI args are missing.

        Returns:
            List of seed peer URLs (e.g., ["http://100.x.x.x:8770", ...])

        December 30, 2025: Added for automatic P2P peer discovery.
        """
        from scripts.p2p.constants import DEFAULT_PORT

        try:
            from app.config.cluster_config import get_cluster_nodes, get_coordinator_node

            seeds: list[str] = []
            seen_ips: set[str] = set()

            # Primary: coordinator node (most stable)
            coord = get_coordinator_node()
            if coord and getattr(coord, "tailscale_ip", None):
                ip = str(coord.tailscale_ip)
                if ip and ip not in seen_ips:
                    seeds.append(f"http://{ip}:{DEFAULT_PORT}")
                    seen_ips.add(ip)

            # Secondary: voter nodes (stable, always online)
            try:
                nodes = get_cluster_nodes()
                for node in nodes.values():
                    if getattr(node, "role", "") == "voter" and getattr(node, "tailscale_ip", None):
                        ip = str(node.tailscale_ip)
                        if ip and ip not in seen_ips:
                            seeds.append(f"http://{ip}:{DEFAULT_PORT}")
                            seen_ips.add(ip)
                            if len(seeds) >= 5:
                                break
            except Exception:  # noqa: BLE001
                pass

            # Fallback: any active nodes with Tailscale IPs
            if len(seeds) < 3:
                try:
                    nodes = get_cluster_nodes()
                    for node in nodes.values():
                        if getattr(node, "tailscale_ip", None) and getattr(node, "is_active", True):
                            ip = str(node.tailscale_ip)
                            if ip and ip not in seen_ips:
                                seeds.append(f"http://{ip}:{DEFAULT_PORT}")
                                seen_ips.add(ip)
                                if len(seeds) >= 5:
                                    break
                except Exception:  # noqa: BLE001
                    pass

            if seeds:
                logger.debug(f"Loaded {len(seeds)} bootstrap seeds from config: {seeds[:3]}...")

            return seeds

        except ImportError:
            logger.debug("cluster_config not available for bootstrap seeds")
            return []
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not load bootstrap seeds from config: {e}")
            return []

    async def _send_initial_relay_heartbeats(self) -> None:
        """Send immediate relay heartbeats on startup for NAT-blocked nodes.

        January 5, 2026: NAT-blocked nodes can't receive inbound connections,
        so they need to proactively register with relay-capable nodes to be
        discoverable by the cluster. This method sends relay heartbeats to
        all configured relay-capable nodes immediately at startup.

        Called after HTTP server starts but before regular heartbeat loop.
        """
        from scripts.p2p.constants import DEFAULT_PORT

        # Load relay-capable nodes from distributed_hosts.yaml
        relay_nodes: list[tuple[str, str, int]] = []  # (node_id, ip, port)
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}

            for node_id, node_cfg in nodes.items():
                if node_id == self.node_id:
                    continue  # Skip self
                if not node_cfg.get("relay_capable", False):
                    continue
                if not node_cfg.get("p2p_enabled", True):
                    continue

                # Get the best IP to reach this node (prefer Tailscale)
                ip = node_cfg.get("tailscale_ip") or node_cfg.get("ssh_host", "")
                port = node_cfg.get("p2p_port", DEFAULT_PORT)
                if ip:
                    relay_nodes.append((node_id, ip, port))

        except ImportError:
            logger.warning("[P2P] cluster_config not available for initial relay heartbeats")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P2P] Failed to load relay-capable nodes: {e}")
            return

        if not relay_nodes:
            logger.info("[P2P] No relay-capable nodes configured for initial heartbeat")
            return

        logger.info(f"[P2P] Sending initial relay heartbeats to {len(relay_nodes)} relay-capable nodes")

        # Send relay heartbeats to all relay-capable nodes
        success_count = 0
        for node_id, ip, port in relay_nodes:
            relay_url = f"http://{ip}:{port}"
            try:
                result = await self._send_relay_heartbeat(relay_url)
                if result.get("success"):
                    success_count += 1
                    logger.info(f"[P2P] Initial relay heartbeat to {node_id} ({ip}:{port}) succeeded")
                else:
                    error = result.get("error", "unknown")
                    logger.debug(f"[P2P] Initial relay heartbeat to {node_id} failed: {error}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Initial relay heartbeat to {node_id} error: {e}")

        if success_count > 0:
            logger.info(f"[P2P] NAT-blocked node registered with {success_count}/{len(relay_nodes)} relay nodes")
        else:
            logger.warning(f"[P2P] Failed to register with any relay nodes - cluster discovery may be delayed")

    async def _send_startup_peer_announcements(self) -> None:
        """Send immediate announcements to all known peers on startup.

        January 7, 2026: Instead of waiting for the first heartbeat interval,
        immediately announce to all known peers. This reduces discovery latency
        from 15-30s down to 2-5s after startup.

        Feb 22, 2026: Made concurrent with 10s per-peer timeout to prevent
        blocking startup for 3+ minutes when peers are unreachable.
        """
        from scripts.p2p.network import NonBlockingAsyncLockWrapper

        peers_to_announce = []
        for peer_addr in self.known_peers:
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
                peers_to_announce.append((scheme, host, port))
            except (AttributeError, ValueError):
                continue

        if not peers_to_announce:
            return

        success_count = 0

        async def _announce_one(scheme, host, port):
            nonlocal success_count
            try:
                info = await asyncio.wait_for(
                    self._send_heartbeat_to_peer(host, port, scheme=scheme),
                    timeout=10.0,
                )
                if info and info.node_id != self.node_id:
                    async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                        is_first_contact = info.node_id not in self.peers
                        info.last_heartbeat = time.time()
                        self.peers[info.node_id] = info
                        self._publish_peers_snapshot()
                    if is_first_contact:
                        logger.info(f"[P2P] Startup announcement discovered peer: {info.node_id}")
                    success_count += 1
            except asyncio.TimeoutError:
                logger.debug(f"[P2P] Startup announcement to {host}:{port} timed out (10s)")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Startup announcement to {host}:{port} failed: {e}")

        # Run all announcements concurrently with an overall 30s timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*[_announce_one(s, h, p) for s, h, p in peers_to_announce],
                               return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[P2P] Startup announcements overall timeout (30s)")

        total = len(peers_to_announce)
        if success_count > 0:
            logger.info(f"[P2P] Startup announcements: {success_count}/{total} peers reachable")
        elif total > 0:
            logger.warning(f"[P2P] Startup announcements: no peers reachable (tried {total})")

    async def _heartbeat_loop(self):
        """Send heartbeats to all known peers."""
        from app.core.async_context import fire_and_forget
        from scripts.p2p.constants import (
            DEFAULT_PORT,
            HEARTBEAT_INTERVAL,
            LEADER_LEASE_DURATION,
            LEADER_LEASE_RENEW_INTERVAL,
            PEER_RECOVERY_RETRY_INTERVAL,
            RELAY_HEARTBEAT_INTERVAL,
        )
        from scripts.p2p.models import NodeInfo, NodeRole
        from scripts.p2p.network import NonBlockingAsyncLockWrapper
        from scripts.p2p.utils import systemd_notify_watchdog

        # Lazy import for optional coordination module
        try:
            from app.coordination.resource_optimizer import NodeResources, get_resource_optimizer
            _has_new_coordination = True
        except ImportError:
            _has_new_coordination = False
            get_resource_optimizer = None  # type: ignore[assignment]

        # Jan 11, 2026: Phase 5 - Initial heartbeat burst to prevent startup race
        # Send immediate heartbeats to all known peers so they discover us quickly
        # This fixes the issue where peers think we're dead before our first heartbeat
        logger.info("Sending initial heartbeat burst to known peers")
        for peer_addr in self.known_peers:
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
                await self._send_heartbeat_to_peer(host, port, scheme=scheme, timeout=10)
            except Exception:
                pass  # Best effort, regular loop will retry

        while self.running:
            try:
                # Feb 23, 2026: SAFETY NET — force non-coordinator nodes to follower.
                # Many code paths (gossip, anti-entropy, Raft, etc.) bypass _set_leader()
                # and directly set self.leader_id = self.node_id. This catch-all check
                # clears self-leadership every heartbeat cycle (~10s) for non-coordinators.
                _is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
                if not _is_coordinator and self.leader_id == self.node_id:
                    logger.info(
                        "[HeartbeatLoop] Non-coordinator has self-leadership, clearing"
                    )
                    self.leader_id = None
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0
                    # NodeRole already imported at module level from scripts.p2p.types
                    # Do NOT use local import here - it shadows the global, causing
                    # UnboundLocalError at lines 9020+ when this branch doesn't execute.
                    self.role = NodeRole.FOLLOWER

                # Jan 20, 2026: Check for and fix leadership state desync every heartbeat
                # This recovers from gossip race conditions where leader_id/role diverge
                try:
                    if self.leadership.recover_leadership_desync():
                        logger.info("[HeartbeatLoop] Recovered from leadership desync")
                except Exception as e:
                    logger.debug(f"[HeartbeatLoop] Desync check failed: {e}")

                # Jan 23, 2026: Phase 2 - Reconcile ULSM with gossip consensus every 30s
                # This fixes the issue where nodes are consensus leader but don't claim leadership
                now = time.time()
                last_reconcile = getattr(self, "_last_leadership_reconcile", 0)
                if now - last_reconcile >= 30.0:
                    self._last_leadership_reconcile = now
                    try:
                        if self.leadership.reconcile_leadership_state():
                            logger.info("[HeartbeatLoop] Reconciled leadership state with gossip consensus")
                    except Exception as e:
                        logger.debug(f"[HeartbeatLoop] Leadership reconciliation failed: {e}")

                # Send to known peers from config
                for peer_addr in self.known_peers:
                    try:
                        scheme, host, port = self._parse_peer_address(peer_addr)
                    except (AttributeError):
                        continue

                    # Use relay heartbeat for HTTPS endpoints (they're proxies/relays),
                    # explicitly configured relay peers (--relay-peers flag),
                    # or if this node is NAT-blocked and needs to relay ALL outbound heartbeats
                    use_relay = scheme == "https" or peer_addr in self.relay_peers or self._force_relay_mode
                    if use_relay:
                        # Relay/proxy endpoint, use relay heartbeat
                        relay_url = f"{scheme}://{host}" if port in (80, 443) else f"{scheme}://{host}:{port}"
                        result = await self._send_relay_heartbeat(relay_url)
                        if result.get("success"):
                            # Relay heartbeat already updates peers and leader
                            continue

                    info = await self._send_heartbeat_to_peer(host, port, scheme=scheme)
                    if info:
                        if info.node_id == self.node_id:
                            continue
                        # Dec 2025: Track first-contact for HOST_ONLINE emission
                        async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                            is_first_contact = info.node_id not in self.peers
                            info.last_heartbeat = time.time()
                            self.peers[info.node_id] = info
                            self._publish_peers_snapshot()
                        # Dec 2025: Emit HOST_ONLINE for newly discovered peers
                        if is_first_contact:
                            capabilities = []
                            if getattr(info, "has_gpu", False):
                                gpu_type = getattr(info, "gpu_type", "") or "gpu"
                                capabilities.append(gpu_type)
                            else:
                                capabilities.append("cpu")
                            await self._emit_host_online(info.node_id, capabilities)
                            logger.info(f"First-contact peer via heartbeat loop: {info.node_id}")
                        if info.role == NodeRole.LEADER and info.node_id != self.node_id:
                            peers_snapshot = self.get_peers_list_ro()
                            conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
                            if not self._is_leader_eligible(info, conflict_keys, require_alive=False):
                                continue
                            if self.role == NodeRole.LEADER and info.node_id <= self.node_id:
                                continue
                            if (
                                self.leader_id
                                and self.leader_id != info.node_id
                                and self._is_leader_lease_valid()
                                and info.node_id <= self.leader_id
                            ):
                                continue
                            # Feb 2026: Skip leader adoption if we have forced leader override
                            if getattr(self, "_forced_leader_override", False) and self.leader_id == self.node_id:
                                continue
                            if self.leader_id != info.node_id or self.role != NodeRole.FOLLOWER:
                                logger.info(f"Following configured leader from heartbeat: {info.node_id}")
                            prev_leader = self.leader_id
                            # Provisional lease: allow time for the leader to send
                            # a /coordinator lease renewal after we discover it via
                            # heartbeat (prevents leaderless oscillation right after
                            # restarts/partitions).
                            if prev_leader != info.node_id or not self._is_leader_lease_valid():
                                self.leader_lease_id = ""
                                self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                            # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                            self._set_leader(info.node_id, reason="heartbeat_configured_leader", save_state=False)

                # Send to discovered peers (skip NAT-blocked peers and ambiguous endpoints).
                # Jan 12, 2026: Use cached snapshot to reduce lock contention (1s staleness OK for heartbeat)
                # Jan 30, 2026: Use network orchestrator directly
                peers_snapshot = self.network.get_cached_peer_snapshot(max_age_seconds=1.0)
                conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
                peer_list = [
                    p for p in peers_snapshot
                    if (
                        not p.nat_blocked
                        and self._endpoint_key(p) not in conflict_keys
                    )
                ]

                for peer in peer_list:
                    if peer.node_id != self.node_id:
                        if not peer.should_retry():
                            continue

                        # Jan 11, 2026: Phase 2 - Adaptive heartbeat timing based on consecutive failures
                        # This gives flaky peers time to recover without spamming them
                        failures = int(getattr(peer, "consecutive_failures", 0) or 0)
                        if failures == 0:
                            heartbeat_interval = HEARTBEAT_INTERVAL  # 15s for healthy peers
                        elif failures == 1:
                            heartbeat_interval = 5  # Quick retry after first failure
                        elif failures == 2:
                            heartbeat_interval = 10  # Second retry
                        elif failures < 5:
                            heartbeat_interval = 20  # Slower retries
                        else:
                            heartbeat_interval = 30  # Very slow for consistently failing

                        # Check if heartbeat is due for this peer
                        last_sent = float(getattr(peer, "last_heartbeat_sent", 0.0) or 0.0)
                        now = time.time()
                        if now - last_sent < heartbeat_interval:
                            continue  # Not time yet for this peer

                        # Mark the send time
                        peer.last_heartbeat_sent = now

                        peer_scheme = getattr(peer, "scheme", "http") or "http"
                        info = await self._send_heartbeat_to_peer(peer.host, peer.port, scheme=peer_scheme)
                        if not info and getattr(peer, "reported_host", "") and getattr(peer, "reported_port", 0):
                            # Multi-path retry: fall back to self-reported endpoint when the
                            # observed reachable endpoint fails (e.g., mixed overlays).
                            try:
                                rh = str(getattr(peer, "reported_host", "") or "").strip()
                                rp = int(getattr(peer, "reported_port", 0) or 0)
                            except (ValueError, AttributeError):
                                rh, rp = "", 0
                            if rh and rp and (rh != peer.host or rp != peer.port):
                                info = await self._send_heartbeat_to_peer(rh, rp, scheme=peer_scheme)
                        # Self-healing: Tailscale IP fallback when both primary and reported fail
                        if not info:
                            ts_ip = self._get_tailscale_ip_for_peer(peer.node_id)
                            if ts_ip and ts_ip != peer.host:
                                # Try Tailscale mesh IP (100.x.x.x)
                                info = await self._send_heartbeat_to_peer(ts_ip, peer.port, scheme=peer_scheme)
                                if info:
                                    logger.info(f"Reached {peer.node_id} via Tailscale ({ts_ip})")
                        if info:
                            info.consecutive_failures = 0
                            info.last_failure_time = 0.0
                            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                                info.last_heartbeat = time.time()
                                self.peers[info.node_id] = info
                                self._publish_peers_snapshot()
                            if info.role == NodeRole.LEADER and self.role != NodeRole.LEADER:
                                if not self._is_leader_eligible(info, conflict_keys, require_alive=False):
                                    continue
                                if (
                                    self.leader_id
                                    and self.leader_id != info.node_id
                                    and self._is_leader_lease_valid()
                                    and info.node_id <= self.leader_id
                                ):
                                    continue
                                # Feb 2026: Skip leader adoption if we have forced leader override
                                if getattr(self, "_forced_leader_override", False) and self.leader_id == self.node_id:
                                    pass
                                else:
                                    if self.leader_id != info.node_id:
                                        logger.info(f"Adopted leader from heartbeat: {info.node_id}")
                                    prev_leader = self.leader_id
                                    if prev_leader != info.node_id or not self._is_leader_lease_valid():
                                        self.leader_lease_id = ""
                                        self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                                    # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                                    self._set_leader(info.node_id, reason="heartbeat_adopt_leader", save_state=False)
                        else:
                            async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
                                existing = self.peers.get(peer.node_id)
                                if existing:
                                    existing.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0) + 1
                                    existing.last_failure_time = time.time()

                # If we're only connected to a seed peer (or lost cluster membership),
                # pull a fresh peer snapshot so leader election converges quickly.
                await self._bootstrap_from_known_peers()

                # Get current time for all time-based checks in this cycle
                now = time.time()

                # NAT-blocked nodes: poll a relay endpoint for peer snapshots + commands.
                if getattr(self.self_info, "nat_blocked", False):
                    if now - self.last_relay_heartbeat >= RELAY_HEARTBEAT_INTERVAL:
                        relay_urls: list[str] = []
                        leader_peer = self._get_leader_peer()
                        if leader_peer and leader_peer.node_id != self.node_id:
                            relay_urls.append(f"{leader_peer.scheme}://{leader_peer.host}:{leader_peer.port}")
                        for peer_addr in self.known_peers:
                            try:
                                scheme, host, port = self._parse_peer_address(peer_addr)
                            except (AttributeError):
                                continue
                            relay_urls.append(f"{scheme}://{host}:{port}")
                        seen: set[str] = set()
                        relay_urls = [u for u in relay_urls if not (u in seen or seen.add(u))]

                        for relay_url in relay_urls:
                            result = await self._send_relay_heartbeat(relay_url)
                            if result.get("success"):
                                self.last_relay_heartbeat = now
                                break

                # Check for dead peers
                await self._check_dead_peers_async()

                # Dec 30, 2025: Probe retired peers periodically to detect recovery
                # This runs every PEER_RECOVERY_RETRY_INTERVAL (120s) to actively probe
                # retired nodes that may have come back online after cluster restart.
                last_probe = getattr(self, "_last_retired_probe", 0.0)
                if now - last_probe >= PEER_RECOVERY_RETRY_INTERVAL:
                    self._last_retired_probe = now
                    try:
                        await self.network.probe_retired_peers_async()
                    except Exception as e:
                        logger.warning(f"Error in retired peer probe: {e}")

                # Self-healing: detect network partition and trigger Tailscale-priority mode
                # Jan 30, 2026: Use network orchestrator directly
                if self.network.detect_network_partition():
                    self.network.enable_tailscale_priority()
                    # Also enable partition-local election if no voters reachable
                    if not self._has_voter_quorum():
                        self.leadership.enable_partition_local_election()
                    # Force refresh all IP sources to discover alternative paths
                    last_refresh = getattr(self, "_last_partition_ip_refresh", 0)
                    if time.time() - last_refresh > 60:  # Refresh at most once per minute
                        self._last_partition_ip_refresh = time.time()
                        # Jan 28, 2026: Uses ip_discovery_manager directly
                        fire_and_forget(
                            self.ip_discovery_manager.force_ip_refresh_all_sources(),
                            name=f"force_ip_refresh:{self.node_id}",
                        )

                    # Jan 13, 2026: Exponential backoff during isolation
                    # Check if we're completely isolated (no alive peers)
                    alive_peers = self._peer_query.alive_count().unwrap_or(0)
                    if alive_peers == 0:
                        # Track isolation start time
                        if not hasattr(self, "_isolation_start"):
                            self._isolation_start = time.time()
                            self._isolation_backoff_seconds = HEARTBEAT_INTERVAL
                            logger.warning("Node is isolated - no alive peers, starting exponential backoff")

                        # Calculate exponential backoff based on isolation duration
                        isolation_duration = time.time() - self._isolation_start
                        if isolation_duration > 60:  # After 1 min
                            self._isolation_backoff_seconds = min(30, self._isolation_backoff_seconds * 1.5)
                        if isolation_duration > 180:  # After 3 min
                            self._isolation_backoff_seconds = min(60, self._isolation_backoff_seconds * 1.5)
                        if isolation_duration > 300:  # After 5 min
                            self._isolation_backoff_seconds = min(120, self._isolation_backoff_seconds)

                        logger.debug(f"Isolated for {isolation_duration:.0f}s, backoff={self._isolation_backoff_seconds:.0f}s")
                        # Apply additional backoff sleep (on top of normal HEARTBEAT_INTERVAL)
                        extra_backoff = self._isolation_backoff_seconds - HEARTBEAT_INTERVAL
                        if extra_backoff > 0:
                            await asyncio.sleep(extra_backoff)
                    else:
                        # Reset isolation tracking when peers are reachable
                        if hasattr(self, "_isolation_start"):
                            logger.info(f"Isolation ended after {time.time() - self._isolation_start:.0f}s, {alive_peers} peers alive")
                            delattr(self, "_isolation_start")
                            if hasattr(self, "_isolation_backoff_seconds"):
                                delattr(self, "_isolation_backoff_seconds")
                elif getattr(self, "_tailscale_priority", False):
                    # Check if priority mode should expire
                    if time.time() > getattr(self, "_tailscale_priority_until", 0):
                        # Check if connectivity recovered
                        alive_count = self._peer_query.alive_count().unwrap_or(0)
                        if alive_count > 0:
                            self.network.disable_tailscale_priority()

                # Self-healing: check if partition healed and restore original voters
                if hasattr(self, "_original_voters"):
                    self.leadership.restore_original_voters()

                # Dynamic voter management: promote/demote voters based on health
                # Only the leader manages voters to ensure consistency
                if self.role == NodeRole.LEADER:
                    self.network.manage_dynamic_voters()

                # Health-based leadership: step down if we can't reach enough peers
                if self.role == NodeRole.LEADER and not self._check_leader_health():
                    logger.info("Stepping down due to degraded health")
                    # Jan 3, 2026: Use _set_leader() for atomic leadership assignment (Phase 4)
                    self._set_leader(None, reason="degraded_health", save_state=True)
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self._release_voter_grant_if_self()
                    # Jan 13, 2026: Add sleep before continue to prevent busy loop
                    # when repeatedly stepping down due to degraded health
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                    continue  # Skip leader duties this cycle

                # P0 Dec 2025: Monitor leader heartbeat for early warning
                # Emit LEADER_HEARTBEAT_MISSING if leader lease is approaching expiry
                if self.role == NodeRole.FOLLOWER and self.leader_id:
                    now = time.time()
                    # Warning threshold: 45 seconds (3x lease renewal interval)
                    heartbeat_warning_threshold = LEADER_LEASE_RENEW_INTERVAL * 3
                    time_until_expiry = self.leader_lease_expires - now
                    # Emit warning if lease will expire within warning threshold
                    if 0 < time_until_expiry < heartbeat_warning_threshold:
                        last_warning = getattr(self, "_last_heartbeat_missing_warning", 0.0)
                        # Only warn once per 30 seconds to avoid spam
                        if now - last_warning > 30:
                            self._last_heartbeat_missing_warning = now
                            delay_seconds = (LEADER_LEASE_DURATION - time_until_expiry)
                            try:
                                from app.distributed.data_events import emit_leader_heartbeat_missing

                                fire_and_forget(
                                    emit_leader_heartbeat_missing(
                                        leader_id=self.leader_id,
                                        last_heartbeat=(
                                            self.leader_lease_expires - LEADER_LEASE_DURATION
                                        ),
                                        expected_interval=LEADER_LEASE_RENEW_INTERVAL,
                                        delay_seconds=delay_seconds,
                                        source=self.node_id,
                                    ),
                                    name=f"leader_heartbeat_missing:{self.leader_id}",
                                )
                            except ImportError:
                                pass  # Graceful degradation if event system not available

                # LEARNED LESSONS - Lease renewal to maintain leadership
                if self.role == NodeRole.LEADER:
                    await self._renew_leader_lease()

                # P2P monitoring: start/stop services based on leadership
                await self._stop_monitoring_if_not_leader()
                if self.role == NodeRole.LEADER:
                    await self._start_monitoring_if_leader()

                # P2P auto-deployer: start/stop based on leadership
                if self.role != NodeRole.LEADER and self._auto_deployer_task:
                    await self._stop_p2p_auto_deployer()
                elif self.role == NodeRole.LEADER and not self._auto_deployer_task:
                    await self._start_p2p_auto_deployer()

                # Report node resources to resource_optimizer for cluster-wide utilization tracking
                # This enables cooperative 60-80% utilization targeting across orchestrators
                if _has_new_coordination and get_resource_optimizer is not None:
                    try:
                        optimizer = get_resource_optimizer()
                        # Mar 2026: Use cached async version instead of blocking
                        # asyncio.to_thread(self._update_self_info). The sync version
                        # takes 10-30s on macOS (pgrep, psutil, NFS checks) and consumes
                        # a thread pool slot every heartbeat (10-15s). With only 8 threads,
                        # this starves queue_populator and voter_heartbeat, causing cascading
                        # 600s timeouts that eventually trigger P2P recovery daemon to
                        # pkill the orchestrator after ~2 hours.
                        # cache_ttl=30s is sufficient for resource metrics (they don't
                        # change rapidly), and dramatically reduces thread pool pressure.
                        await self._update_self_info_async(cache_ttl=30.0)
                        node_resources = NodeResources(
                            node_id=self.node_id,
                            cpu_percent=self.self_info.cpu_percent,
                            memory_percent=self.self_info.memory_percent,
                            active_jobs=self.self_info.selfplay_jobs + self.self_info.training_jobs,
                            has_gpu=self.self_info.has_gpu,
                            gpu_name=self.self_info.gpu_type or "",
                        )
                        optimizer.report_node_resources(node_resources)
                    except (AttributeError):
                        pass  # Non-critical, don't disrupt heartbeat

                # Save state periodically
                self._save_state()

            except Exception as e:  # noqa: BLE001
                logger.info(f"Heartbeat error: {e}")

            # Notify systemd watchdog that we're still alive
            systemd_notify_watchdog()

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _send_voter_heartbeat(self, voter_peer) -> bool:
        """Send a heartbeat to a voter peer with shorter timeout.

        Jan 27, 2026: Phase 16A - Delegates to HeartbeatManager.
        """
        return await self.heartbeat_manager.send_voter_heartbeat(voter_peer)

    async def _try_voter_alternative_endpoints(self, voter_peer) -> bool:
        """Try alternative endpoints for a voter peer."""
        from scripts.p2p.constants import VOTER_HEARTBEAT_TIMEOUT

        peer_scheme = getattr(voter_peer, "scheme", "http") or "http"

        # Try 1: Tailscale IP
        ts_ip = self._get_tailscale_ip_for_peer(voter_peer.node_id)
        if ts_ip and ts_ip != voter_peer.host:
            info = await self._send_heartbeat_to_peer(ts_ip, voter_peer.port, scheme=peer_scheme, timeout=VOTER_HEARTBEAT_TIMEOUT)
            if info:
                logger.info(f"Reached voter {voter_peer.node_id} via Tailscale ({ts_ip})")
                with self.peers_lock:
                    info.last_heartbeat = time.time()
                    info.consecutive_failures = 0
                    self.peers[info.node_id] = info
                    self._publish_peers_snapshot()
                return True

        # Try 2: Reported host/port
        rh = str(getattr(voter_peer, "reported_host", "") or "").strip()
        rp = int(getattr(voter_peer, "reported_port", 0) or 0)
        if rh and rp and (rh != voter_peer.host or rp != voter_peer.port):
            info = await self._send_heartbeat_to_peer(rh, rp, scheme=peer_scheme, timeout=VOTER_HEARTBEAT_TIMEOUT)
            if info:
                logger.info(f"Reached voter {voter_peer.node_id} via reported endpoint ({rh}:{rp})")
                with self.peers_lock:
                    info.last_heartbeat = time.time()
                    info.consecutive_failures = 0
                    self.peers[info.node_id] = info
                    self._publish_peers_snapshot()
                return True

        return False

    async def _discover_voter_peer(self, voter_id: str):
        """Discover a voter peer from known peers."""
        from scripts.p2p.models import NodeInfo

        # Ask known peers for the voter's endpoint
        for peer_addr in self.known_peers:
            try:
                scheme, host, port = self._parse_peer_address(peer_addr)
                async with aiohttp.ClientSession() as session, session.get(
                    f"{scheme}://{host}:{port}/relay/peers",
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers=self._auth_headers()
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        peers_data = data.get("peers", {})
                        if voter_id in peers_data:
                            peer_info = NodeInfo.from_dict(peers_data[voter_id])
                            with self.peers_lock:
                                self.peers[voter_id] = peer_info
                                self._publish_peers_snapshot()
                            logger.info(f"Discovered voter {voter_id} from {host}")
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError, ImportError):
                continue

    async def _refresh_voter_mesh(self):
        """Ensure all voters have knowledge of each other."""
        if not self.voter_node_ids:
            return

        # Jan 12, 2026: Use lock-free PeerSnapshot for read-only access
        peers_snapshot = self._peer_snapshot.get_snapshot()

        # Check how many voters we know about (outside lock)
        known_voters = [v for v in self.voter_node_ids if v in peers_snapshot or v == self.node_id]

        if len(known_voters) < len(self.voter_node_ids):
            missing_voters = [v for v in self.voter_node_ids if v not in known_voters]
            logger.info(f"Voter mesh incomplete, missing: {missing_voters}")

            # Try to discover missing voters
            for voter_id in missing_voters:
                await self._discover_voter_peer(voter_id)

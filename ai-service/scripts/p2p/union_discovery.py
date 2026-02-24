"""
Union-based peer discovery.

Dec 30, 2025: Combines ALL peer discovery methods into a single unified view.

Philosophy: Union over intersection - if ANY discovery method finds a peer,
include it in the known peers list. More sources = higher confidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml

from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredPeer:
    """A peer discovered by one or more discovery methods."""

    node_id: str
    addresses: list[str] = field(default_factory=list)  # All known addresses
    discovery_sources: set[str] = field(default_factory=set)  # Which methods found this
    last_seen: float = 0.0
    confidence: float = 0.5  # 0-1, higher if multiple sources agree
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra info

    @property
    def best_address(self) -> str | None:
        """Get the best address to use (Tailscale preferred)."""
        for addr in self.addresses:
            if addr.startswith("100."):  # Tailscale IP
                return addr
        return self.addresses[0] if self.addresses else None


@dataclass
class DiscoveryResult:
    """Result of a discovery method."""

    source: str
    peers: list[DiscoveredPeer]
    error: str | None = None
    duration_ms: float = 0.0


class UnionDiscovery:
    """
    Union of all peer discovery methods.

    Runs all discovery sources in parallel and merges results.
    Multiple sources finding the same peer increases confidence.
    """

    def __init__(self):
        self._known_peers: dict[str, DiscoveredPeer] = {}
        self._discovery_sources: list[tuple[str, Callable[[], Coroutine[Any, Any, list[DiscoveredPeer]]]]] = []
        self._last_discovery_time: float = 0.0
        self._discovery_interval: float = float(
            os.environ.get("RINGRIFT_DISCOVERY_INTERVAL", "30")
        )

        # Register built-in discovery sources
        self._register_builtin_sources()

    def _register_builtin_sources(self) -> None:
        """Register all built-in discovery sources."""
        self.register_source("yaml_config", self._discover_from_yaml)
        self.register_source("tailscale_cli", self._discover_from_tailscale_cli)
        self.register_source("gossip", self._discover_from_gossip)
        self.register_source("swim", self._discover_from_swim)
        self.register_source("raft", self._discover_from_raft)
        self.register_source("bully_election", self._discover_from_bully_election)
        self.register_source("dns_srv", self._discover_from_dns_srv)
        self.register_source("bonjour", self._discover_from_bonjour)

    def register_source(
        self,
        name: str,
        discovery_func: Callable[[], Coroutine[Any, Any, list[DiscoveredPeer]]],
    ) -> None:
        """Register a discovery source."""
        self._discovery_sources.append((name, discovery_func))
        logger.debug(f"Registered discovery source: {name}")

    async def discover_all_peers(
        self, timeout: float = 30.0
    ) -> dict[str, DiscoveredPeer]:
        """
        Run all discovery methods and return union of results.

        Returns:
            Dict mapping node_id to DiscoveredPeer with merged info
        """
        start_time = time.time()

        # Run all discovery sources in parallel
        tasks = []
        for name, func in self._discovery_sources:
            task = safe_create_task(self._run_discovery_source(name, func, timeout), name=f"union-discovery-{name}")
            tasks.append((name, task))

        # Wait for all to complete
        results: list[DiscoveryResult] = []
        for name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                results.append(result)
            except asyncio.TimeoutError:
                results.append(DiscoveryResult(
                    source=name,
                    peers=[],
                    error=f"timeout ({timeout}s)",
                ))
            except Exception as e:
                results.append(DiscoveryResult(
                    source=name,
                    peers=[],
                    error=f"{type(e).__name__}: {e}",
                ))

        # Merge all results
        for result in results:
            if result.error:
                logger.debug(f"Discovery source {result.source} failed: {result.error}")
            else:
                logger.debug(
                    f"Discovery source {result.source} found {len(result.peers)} peers "
                    f"in {result.duration_ms:.1f}ms"
                )
                for peer in result.peers:
                    self._merge_peer(peer)

        self._last_discovery_time = time.time()
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Union discovery complete: {len(self._known_peers)} peers from "
            f"{len(results)} sources in {duration_ms:.1f}ms"
        )

        return dict(self._known_peers)

    async def _run_discovery_source(
        self,
        name: str,
        func: Callable[[], Coroutine[Any, Any, list[DiscoveredPeer]]],
        timeout: float,
    ) -> DiscoveryResult:
        """Run a single discovery source with error handling."""
        start_time = time.time()
        try:
            peers = await asyncio.wait_for(func(), timeout=timeout)
            return DiscoveryResult(
                source=name,
                peers=peers,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except asyncio.TimeoutError:
            return DiscoveryResult(
                source=name,
                peers=[],
                error=f"timeout ({timeout}s)",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return DiscoveryResult(
                source=name,
                peers=[],
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _merge_peer(self, peer: DiscoveredPeer) -> None:
        """Merge peer info, taking union of addresses and sources."""
        if peer.node_id in self._known_peers:
            existing = self._known_peers[peer.node_id]
            # Union of addresses (preserve uniqueness)
            all_addrs = set(existing.addresses) | set(peer.addresses)
            existing.addresses = list(all_addrs)
            # Union of sources
            existing.discovery_sources |= peer.discovery_sources
            # Update confidence (more sources = higher confidence)
            existing.confidence = min(1.0, len(existing.discovery_sources) * 0.2)
            # Update last seen
            existing.last_seen = max(existing.last_seen, peer.last_seen)
            # Merge metadata
            existing.metadata.update(peer.metadata)
        else:
            self._known_peers[peer.node_id] = peer

    def get_peer(self, node_id: str) -> DiscoveredPeer | None:
        """Get a specific peer by node_id."""
        return self._known_peers.get(node_id)

    def get_all_peers(self) -> dict[str, DiscoveredPeer]:
        """Get all known peers."""
        return dict(self._known_peers)

    def get_peers_by_source(self, source: str) -> list[DiscoveredPeer]:
        """Get peers discovered by a specific source."""
        return [
            p for p in self._known_peers.values()
            if source in p.discovery_sources
        ]

    def get_high_confidence_peers(self, min_confidence: float = 0.6) -> list[DiscoveredPeer]:
        """Get peers with high confidence (multiple sources)."""
        return [
            p for p in self._known_peers.values()
            if p.confidence >= min_confidence
        ]

    # =========================================================================
    # Discovery Source Implementations
    # =========================================================================

    async def _discover_from_yaml(self) -> list[DiscoveredPeer]:
        """Discover peers from distributed_hosts.yaml config."""
        peers = []
        config_path = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            return []

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            for host in config.get("hosts", []):
                node_id = host.get("name")
                if not node_id:
                    continue

                addresses = []
                if host.get("tailscale_ip"):
                    addresses.append(host["tailscale_ip"])
                if host.get("ssh_host"):
                    addresses.append(host["ssh_host"])
                if host.get("http_endpoint"):
                    addresses.append(host["http_endpoint"])

                peers.append(DiscoveredPeer(
                    node_id=node_id,
                    addresses=addresses,
                    discovery_sources={"yaml_config"},
                    last_seen=time.time(),
                    confidence=0.8,  # Config is reliable
                    metadata={
                        "role": host.get("role", ""),
                        "provider": host.get("provider", ""),
                        "status": host.get("status", "active"),
                    },
                ))

        except Exception as e:
            logger.debug(f"YAML discovery failed: {e}")

        return peers

    async def _discover_from_tailscale_cli(self) -> list[DiscoveredPeer]:
        """Discover peers from Tailscale CLI."""
        peers = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode != 0:
                return []

            data = json.loads(stdout.decode())

            for peer_key, info in data.get("Peer", {}).items():
                hostname = info.get("HostName", "")
                if not hostname:
                    continue

                addresses = []
                # Add Tailscale IPs
                for ip in info.get("TailscaleIPs", []):
                    addresses.append(ip)
                # Add current direct address if available
                if info.get("CurAddr"):
                    addresses.append(info["CurAddr"].split(":")[0])

                is_online = info.get("Online", False)

                peers.append(DiscoveredPeer(
                    node_id=hostname,
                    addresses=addresses,
                    discovery_sources={"tailscale_cli"},
                    last_seen=time.time() if is_online else 0,
                    confidence=0.9 if is_online else 0.3,
                    metadata={
                        "tailscale_online": is_online,
                        "tailscale_key": peer_key[:8],  # Truncated for privacy
                        "os": info.get("OS", ""),
                    },
                ))

        except FileNotFoundError:
            logger.debug("Tailscale CLI not available")
        except asyncio.TimeoutError:
            logger.debug("Tailscale CLI timeout")
        except Exception as e:
            logger.debug(f"Tailscale CLI discovery failed: {e}")

        return peers

    async def _discover_from_gossip(self) -> list[DiscoveredPeer]:
        """Discover peers known to gossip protocol."""
        peers = []

        try:
            # Query local P2P status endpoint
            import aiohttp
            from aiohttp import ClientTimeout

            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8770/status") as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()
                    peer_list = data.get("peers", {})

                    for node_id, info in peer_list.items():
                        addresses = []
                        if info.get("http_endpoint"):
                            addresses.append(info["http_endpoint"])
                        if info.get("tailscale_ip"):
                            addresses.append(info["tailscale_ip"])

                        is_alive = info.get("is_alive", False)

                        peers.append(DiscoveredPeer(
                            node_id=node_id,
                            addresses=addresses,
                            discovery_sources={"gossip"},
                            last_seen=info.get("last_heartbeat", 0),
                            confidence=0.95 if is_alive else 0.4,
                            metadata={
                                "has_gpu": info.get("has_gpu", False),
                                "selfplay_jobs": info.get("selfplay_jobs", 0),
                            },
                        ))

        except Exception as e:
            logger.debug(f"Gossip discovery failed: {e}")

        return peers

    async def _discover_from_swim(self) -> list[DiscoveredPeer]:
        """Discover peers from SWIM membership protocol."""
        peers = []

        try:
            # Query SWIM status endpoint
            import aiohttp
            from aiohttp import ClientTimeout

            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8770/swim/members") as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()
                    members = data.get("members", [])

                    for member in members:
                        node_id = member.get("node_id", "")
                        if not node_id:
                            continue

                        addresses = []
                        if member.get("address"):
                            addresses.append(member["address"])

                        state = member.get("state", "unknown")
                        confidence = {"alive": 0.95, "suspected": 0.5, "failed": 0.1}.get(state, 0.3)

                        peers.append(DiscoveredPeer(
                            node_id=node_id,
                            addresses=addresses,
                            discovery_sources={"swim"},
                            last_seen=member.get("last_seen", 0),
                            confidence=confidence,
                            metadata={"swim_state": state},
                        ))

        except Exception as e:
            logger.debug(f"SWIM discovery failed: {e}")

        return peers

    async def _discover_from_raft(self) -> list[DiscoveredPeer]:
        """Discover peers from Raft consensus."""
        peers = []

        try:
            # Query Raft status endpoint
            import aiohttp
            from aiohttp import ClientTimeout

            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8770/raft/status") as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()
                    partners = data.get("partners", [])

                    for partner in partners:
                        # Parse partner address (host:port format)
                        if ":" in partner:
                            host = partner.rsplit(":", 1)[0]
                        else:
                            host = partner

                        peers.append(DiscoveredPeer(
                            node_id=host,
                            addresses=[partner],
                            discovery_sources={"raft"},
                            last_seen=time.time(),
                            confidence=0.9,  # Raft partners are configured
                            metadata={"raft_partner": True},
                        ))

        except Exception as e:
            logger.debug(f"Raft discovery failed: {e}")

        return peers

    async def _discover_from_bully_election(self) -> list[DiscoveredPeer]:
        """Discover peers from Bully election participants."""
        peers = []

        try:
            # Query election status endpoint
            import aiohttp
            from aiohttp import ClientTimeout

            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8770/election/status") as resp:
                    if resp.status != 200:
                        return []

                    data = await resp.json()

                    # Get voters and candidates
                    voters = data.get("voters", [])
                    candidates = data.get("candidates", [])
                    leader = data.get("leader_id")

                    all_nodes = set(voters) | set(candidates)
                    if leader:
                        all_nodes.add(leader)

                    for node_id in all_nodes:
                        peers.append(DiscoveredPeer(
                            node_id=node_id,
                            addresses=[],  # Election doesn't have addresses
                            discovery_sources={"bully_election"},
                            last_seen=time.time(),
                            confidence=0.7,
                            metadata={
                                "is_voter": node_id in voters,
                                "is_leader": node_id == leader,
                            },
                        ))

        except Exception as e:
            logger.debug(f"Bully election discovery failed: {e}")

        return peers

    async def _discover_from_dns_srv(self) -> list[DiscoveredPeer]:
        """Discover peers from DNS SRV records."""
        peers = []

        domain = os.environ.get("RINGRIFT_DNS_SRV_DOMAIN")
        if not domain:
            return []

        try:
            import dns.resolver

            # Query SRV records for _ringrift._tcp.{domain}
            srv_name = f"_ringrift._tcp.{domain}"
            answers = dns.resolver.resolve(srv_name, "SRV")

            for rdata in answers:
                target = str(rdata.target).rstrip(".")
                port = rdata.port

                peers.append(DiscoveredPeer(
                    node_id=target,
                    addresses=[f"{target}:{port}"],
                    discovery_sources={"dns_srv"},
                    last_seen=time.time(),
                    confidence=0.85,
                    metadata={"srv_priority": rdata.priority, "srv_weight": rdata.weight},
                ))

        except ImportError:
            logger.debug("dnspython not installed, skipping DNS SRV discovery")
        except Exception as e:
            logger.debug(f"DNS SRV discovery failed: {e}")

        return peers

    async def _discover_from_bonjour(self) -> list[DiscoveredPeer]:
        """Discover peers via Bonjour/mDNS (local network only)."""
        peers = []

        if not os.environ.get("RINGRIFT_BONJOUR_ENABLED", "").lower() == "true":
            return []

        try:
            from zeroconf import Zeroconf, ServiceBrowser, ServiceListener

            class RingRiftListener(ServiceListener):
                def __init__(self):
                    self.discovered: list[DiscoveredPeer] = []

                def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    info = zc.get_service_info(type_, name)
                    if info:
                        addresses = [str(addr) for addr in info.parsed_addresses()]
                        node_id = info.server.rstrip(".")
                        self.discovered.append(DiscoveredPeer(
                            node_id=node_id,
                            addresses=addresses,
                            discovery_sources={"bonjour"},
                            last_seen=time.time(),
                            confidence=0.8,
                            metadata={"port": info.port},
                        ))

                def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    pass

                def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    pass

            # Browse for a short time
            zc = Zeroconf()
            listener = RingRiftListener()
            browser = ServiceBrowser(zc, "_ringrift-worker._tcp.local.", listener)

            await asyncio.sleep(2.0)  # Wait for discoveries

            browser.cancel()
            zc.close()

            peers = listener.discovered

        except ImportError:
            logger.debug("zeroconf not installed, skipping Bonjour discovery")
        except Exception as e:
            logger.debug(f"Bonjour discovery failed: {e}")

        return peers


# Singleton instance
_union_discovery: UnionDiscovery | None = None


def get_union_discovery() -> UnionDiscovery:
    """Get the singleton UnionDiscovery instance."""
    global _union_discovery
    if _union_discovery is None:
        _union_discovery = UnionDiscovery()
    return _union_discovery


def reset_singleton() -> None:
    """Reset singleton (for testing)."""
    global _union_discovery
    _union_discovery = None

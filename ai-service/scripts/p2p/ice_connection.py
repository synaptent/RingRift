"""
ICE-Style Connection Establishment.

Dec 30, 2025: Part of Phase 4 - Advanced NAT Traversal.

Implements ICE (Interactive Connectivity Establishment) style connection
negotiation between peers. Gathers connectivity candidates from multiple
sources and tries them in priority order until one succeeds.

Candidate Types:
- host: Local IP addresses (fastest, but may be unreachable)
- srflx: Server-reflexive (external IP from STUN)
- relay: Via P2P leader or TURN server (always works, but slower)
- tailscale: Via Tailscale mesh network

Usage:
    from scripts.p2p.ice_connection import ICEGatherer, establish_connection

    gatherer = ICEGatherer(node_id="my-node")
    candidates = await gatherer.gather_candidates()

    # Try to connect to peer
    connection = await establish_connection(
        target_node="peer-node",
        local_candidates=candidates,
        remote_candidates=peer_candidates,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CandidateType(str, Enum):
    """ICE candidate types in order of preference."""

    HOST = "host"  # Local IP address
    SRFLX = "srflx"  # Server-reflexive (from STUN)
    PRFLX = "prflx"  # Peer-reflexive (discovered during connectivity check)
    RELAY = "relay"  # Via relay server
    TAILSCALE = "tailscale"  # Via Tailscale mesh


class TransportType(str, Enum):
    """Transport protocols for candidates."""

    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    TAILSCALE = "tailscale"


@dataclass
class ICECandidate:
    """An ICE candidate representing a potential connection path."""

    type: CandidateType
    transport: TransportType
    address: str
    port: int
    priority: int  # Higher = better (0-2^31)
    foundation: str = ""  # Unique ID for similar candidates
    component_id: int = 1  # 1 = RTP, 2 = RTCP (we use 1 for data)
    related_address: str | None = None  # For srflx/relay: the base address
    related_port: int | None = None

    def __post_init__(self) -> None:
        if not self.foundation:
            self.foundation = f"{self.type.value}-{self.transport.value}-{self.address}"

    @property
    def key(self) -> str:
        """Unique key for this candidate."""
        return f"{self.type.value}:{self.transport.value}:{self.address}:{self.port}"

    def to_sdp(self) -> str:
        """Convert to SDP-style candidate string."""
        return (
            f"candidate:{self.foundation} {self.component_id} {self.transport.value} "
            f"{self.priority} {self.address} {self.port} typ {self.type.value}"
        )


@dataclass
class ICEConnectionState:
    """State of an ICE connection attempt."""

    local_candidates: list[ICECandidate] = field(default_factory=list)
    remote_candidates: list[ICECandidate] = field(default_factory=list)
    selected_pair: tuple[ICECandidate, ICECandidate] | None = None
    state: Literal["new", "checking", "connected", "completed", "failed"] = "new"
    error: str | None = None


class ICEGatherer:
    """
    Gathers ICE candidates from multiple sources.

    Collects all possible connection paths to this node:
    - Local network interfaces (host candidates)
    - Public IP via STUN (server-reflexive candidates)
    - Tailscale mesh address
    - Relay endpoints (via P2P leader)
    """

    def __init__(
        self,
        node_id: str,
        http_port: int = 8770,
        enable_stun: bool = True,
        enable_tailscale: bool = True,
        enable_relay: bool = True,
    ):
        """Initialize ICE gatherer.

        Args:
            node_id: Local node identifier
            http_port: HTTP port for P2P communication
            enable_stun: Enable STUN-based candidate gathering
            enable_tailscale: Enable Tailscale candidate gathering
            enable_relay: Enable relay candidate gathering
        """
        self._node_id = node_id
        self._http_port = http_port
        self._enable_stun = enable_stun
        self._enable_tailscale = enable_tailscale
        self._enable_relay = enable_relay
        self._candidates: list[ICECandidate] = []
        self._gather_time: float = 0.0

    async def gather_candidates(self) -> list[ICECandidate]:
        """Gather all connectivity candidates.

        Returns:
            List of ICECandidate sorted by priority (highest first)
        """
        start = time.time()
        self._candidates = []

        # Gather from all sources in parallel
        tasks = [
            self._gather_host_candidates(),
            self._gather_tailscale_candidates() if self._enable_tailscale else asyncio.sleep(0),
            self._gather_stun_candidates() if self._enable_stun else asyncio.sleep(0),
            self._gather_relay_candidates() if self._enable_relay else asyncio.sleep(0),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Sort by priority (highest first)
        self._candidates.sort(key=lambda c: -c.priority)
        self._gather_time = time.time() - start

        logger.info(
            f"Gathered {len(self._candidates)} ICE candidates in {self._gather_time:.2f}s"
        )

        return self._candidates

    async def _gather_host_candidates(self) -> None:
        """Gather host candidates from local network interfaces."""
        try:
            # Get all local IP addresses
            hostname = socket.gethostname()
            addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)

            for addr in addrs:
                ip = addr[4][0]
                if ip.startswith("127."):
                    continue

                # Calculate priority: host candidates are highest
                # Priority = type_pref * 2^24 + local_pref * 2^8 + component
                priority = (126 << 24) + (65535 << 8) + 1

                self._candidates.append(
                    ICECandidate(
                        type=CandidateType.HOST,
                        transport=TransportType.HTTP,
                        address=ip,
                        port=self._http_port,
                        priority=priority,
                    )
                )

            # Also try to get the default route IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                default_ip = s.getsockname()[0]
                s.close()

                # Check if already added
                if not any(c.address == default_ip for c in self._candidates):
                    self._candidates.append(
                        ICECandidate(
                            type=CandidateType.HOST,
                            transport=TransportType.HTTP,
                            address=default_ip,
                            port=self._http_port,
                            priority=(126 << 24) + (65535 << 8) + 1,
                        )
                    )
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Failed to gather host candidates: {e}")

    async def _gather_tailscale_candidates(self) -> None:
        """Gather Tailscale mesh address if available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "ip", "-4",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                tailscale_ip = stdout.decode().strip().split("\n")[0]
                if tailscale_ip:
                    # Tailscale is very reliable but slightly higher latency
                    priority = (110 << 24) + (65535 << 8) + 1

                    self._candidates.append(
                        ICECandidate(
                            type=CandidateType.TAILSCALE,
                            transport=TransportType.TAILSCALE,
                            address=tailscale_ip,
                            port=self._http_port,
                            priority=priority,
                        )
                    )

        except asyncio.TimeoutError:
            logger.debug("Tailscale candidate gathering timeout")
        except Exception as e:
            logger.debug(f"Failed to gather Tailscale candidates: {e}")

    async def _gather_stun_candidates(self) -> None:
        """Gather server-reflexive candidates via STUN."""
        try:
            from scripts.p2p.nat_detection import detect_nat_type

            result = await detect_nat_type()

            if result.external_ip and result.external_port:
                # srflx has lower priority than host
                priority = (100 << 24) + (65535 << 8) + 1

                self._candidates.append(
                    ICECandidate(
                        type=CandidateType.SRFLX,
                        transport=TransportType.HTTP,
                        address=result.external_ip,
                        port=self._http_port,  # Use our HTTP port, not STUN port
                        priority=priority,
                        related_address=self._get_local_ip(),
                        related_port=self._http_port,
                    )
                )

        except Exception as e:
            logger.debug(f"Failed to gather STUN candidates: {e}")

    async def _gather_relay_candidates(self) -> None:
        """Gather relay candidates via P2P leader."""
        try:
            # Check for configured relay endpoints
            relay_hosts = os.environ.get("RINGRIFT_RELAY_HOSTS", "").split(",")
            relay_hosts = [h.strip() for h in relay_hosts if h.strip()]

            for relay in relay_hosts:
                if ":" in relay:
                    host, port_str = relay.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = relay
                    port = 8770

                # Relay has lowest priority but highest reliability
                priority = (0 << 24) + (65535 << 8) + 1

                self._candidates.append(
                    ICECandidate(
                        type=CandidateType.RELAY,
                        transport=TransportType.HTTP,
                        address=host,
                        port=port,
                        priority=priority,
                    )
                )

        except Exception as e:
            logger.debug(f"Failed to gather relay candidates: {e}")

    def _get_local_ip(self) -> str:
        """Get default local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"


class ICEChecker:
    """
    Performs connectivity checks between candidate pairs.

    Tries each local/remote candidate pair in priority order
    until one succeeds.
    """

    def __init__(
        self,
        check_timeout: float = 3.0,
        max_checks: int = 20,
    ):
        """Initialize connectivity checker.

        Args:
            check_timeout: Timeout per connectivity check
            max_checks: Maximum number of pairs to check
        """
        self._check_timeout = check_timeout
        self._max_checks = max_checks

    async def check_connectivity(
        self,
        local_candidates: list[ICECandidate],
        remote_candidates: list[ICECandidate],
    ) -> ICEConnectionState:
        """Check connectivity between local and remote candidates.

        Args:
            local_candidates: Our gathered candidates
            remote_candidates: Peer's gathered candidates

        Returns:
            ICEConnectionState with selected candidate pair
        """
        state = ICEConnectionState(
            local_candidates=local_candidates,
            remote_candidates=remote_candidates,
            state="checking",
        )

        # Generate candidate pairs sorted by combined priority
        pairs = self._generate_pairs(local_candidates, remote_candidates)

        for i, (local, remote) in enumerate(pairs[: self._max_checks]):
            try:
                success = await self._check_pair(local, remote)
                if success:
                    state.selected_pair = (local, remote)
                    state.state = "connected"
                    logger.info(
                        f"ICE connected via {local.type.value}:{local.address} -> "
                        f"{remote.type.value}:{remote.address} (pair {i+1}/{len(pairs)})"
                    )
                    return state

            except Exception as e:
                logger.debug(f"Check failed for pair {i+1}: {e}")

        state.state = "failed"
        state.error = f"All {len(pairs)} candidate pairs failed"
        return state

    def _generate_pairs(
        self,
        local: list[ICECandidate],
        remote: list[ICECandidate],
    ) -> list[tuple[ICECandidate, ICECandidate]]:
        """Generate candidate pairs sorted by priority."""
        pairs = []

        for l in local:
            for r in remote:
                # Only pair compatible transports
                if l.transport != r.transport and l.transport != TransportType.HTTP:
                    continue

                # Combined priority (simple sum)
                combined = l.priority + r.priority
                pairs.append((l, r, combined))

        # Sort by combined priority (highest first)
        pairs.sort(key=lambda x: -x[2])
        return [(l, r) for l, r, _ in pairs]

    async def _check_pair(
        self,
        local: ICECandidate,
        remote: ICECandidate,
    ) -> bool:
        """Check if a candidate pair works.

        Args:
            local: Local candidate (not used for outbound check)
            remote: Remote candidate to check

        Returns:
            True if connectivity check succeeded
        """
        try:
            if remote.transport == TransportType.HTTP:
                return await self._check_http(remote.address, remote.port)
            elif remote.transport == TransportType.TAILSCALE:
                return await self._check_http(remote.address, remote.port)
            else:
                # For other transports, try TCP connect
                return await self._check_tcp(remote.address, remote.port)

        except Exception as e:
            logger.debug(f"Pair check failed: {e}")
            return False

    async def _check_http(self, host: str, port: int) -> bool:
        """Check HTTP connectivity."""
        try:
            import aiohttp
            from aiohttp import ClientTimeout

            url = f"http://{host}:{port}/health"
            timeout = ClientTimeout(total=self._check_timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    return resp.status == 200

        except Exception:
            return False

    async def _check_tcp(self, host: str, port: int) -> bool:
        """Check raw TCP connectivity."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._check_timeout,
            )
            writer.close()
            await writer.wait_closed()
            return True

        except Exception:
            return False


async def establish_connection(
    target_node: str,
    local_candidates: list[ICECandidate] | None = None,
    remote_candidates: list[ICECandidate] | None = None,
    orchestrator: Any = None,
) -> ICEConnectionState:
    """Establish ICE connection to a peer.

    Args:
        target_node: Target node ID
        local_candidates: Pre-gathered local candidates (or gather automatically)
        remote_candidates: Peer's candidates (or fetch from peer)
        orchestrator: P2P orchestrator for peer lookup

    Returns:
        ICEConnectionState with connection result
    """
    # Gather local candidates if not provided
    if local_candidates is None:
        gatherer = ICEGatherer(node_id="local")
        local_candidates = await gatherer.gather_candidates()

    # Fetch remote candidates if not provided
    if remote_candidates is None and orchestrator:
        remote_candidates = await _fetch_remote_candidates(target_node, orchestrator)

    if not remote_candidates:
        return ICEConnectionState(
            local_candidates=local_candidates,
            state="failed",
            error="No remote candidates available",
        )

    # Run connectivity checks
    checker = ICEChecker()
    return await checker.check_connectivity(local_candidates, remote_candidates)


async def _fetch_remote_candidates(
    target_node: str,
    orchestrator: Any,
) -> list[ICECandidate]:
    """Fetch ICE candidates from a remote peer."""
    try:
        # Try to get peer info from orchestrator
        peer_info = orchestrator.peers.get(target_node)
        if not peer_info:
            return []

        # Build candidates from peer endpoints
        candidates = []

        # Primary endpoint
        if peer_info.host:
            candidates.append(
                ICECandidate(
                    type=CandidateType.HOST,
                    transport=TransportType.HTTP,
                    address=peer_info.host,
                    port=peer_info.port or 8770,
                    priority=(126 << 24) + (65535 << 8) + 1,
                )
            )

        # Tailscale endpoint
        if hasattr(peer_info, "tailscale_ip") and peer_info.tailscale_ip:
            candidates.append(
                ICECandidate(
                    type=CandidateType.TAILSCALE,
                    transport=TransportType.TAILSCALE,
                    address=peer_info.tailscale_ip,
                    port=peer_info.port or 8770,
                    priority=(110 << 24) + (65535 << 8) + 1,
                )
            )

        # Additional endpoints from gossip
        if hasattr(peer_info, "endpoints"):
            for endpoint in peer_info.endpoints:
                if ":" in endpoint:
                    host, port_str = endpoint.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = endpoint
                    port = 8770

                if not any(c.address == host for c in candidates):
                    candidates.append(
                        ICECandidate(
                            type=CandidateType.SRFLX,
                            transport=TransportType.HTTP,
                            address=host,
                            port=port,
                            priority=(100 << 24) + (65535 << 8) + 1,
                        )
                    )

        return candidates

    except Exception as e:
        logger.error(f"Failed to fetch remote candidates: {e}")
        return []

"""
NAT Type Detection Module.

Dec 30, 2025: Part of Phase 4 - Advanced NAT Traversal.

Detects the NAT type of the local network using STUN-style probing.
This information is used to select the optimal transport for each peer.

NAT Types (in order of connectivity difficulty):
- OPEN: Direct connectivity, no NAT
- FULL_CONE: Any external host can reach after outbound connection
- RESTRICTED_CONE: Only hosts we've sent to can reply
- PORT_RESTRICTED: Only host+port we've sent to can reply
- SYMMETRIC: Different external mapping per destination (hardest)
- BLOCKED: No UDP connectivity

Usage:
    from scripts.p2p.nat_detection import detect_nat_type, NATType

    nat_type = await detect_nat_type()
    if nat_type == NATType.SYMMETRIC:
        # Use relay-based transport
        pass
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Default STUN servers for NAT detection
DEFAULT_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun2.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
]

# STUN message types
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_RESPONSE = 0x0101
STUN_MAGIC_COOKIE = 0x2112A442

# STUN attribute types
ATTR_MAPPED_ADDRESS = 0x0001
ATTR_XOR_MAPPED_ADDRESS = 0x0020
ATTR_CHANGE_REQUEST = 0x0003
ATTR_RESPONSE_ORIGIN = 0x802B
ATTR_OTHER_ADDRESS = 0x802C


class NATType(str, Enum):
    """NAT types in order of connectivity difficulty."""

    OPEN = "open"  # No NAT, direct connectivity
    FULL_CONE = "full_cone"  # Any external can reach after outbound
    RESTRICTED_CONE = "restricted_cone"  # Only replied-to hosts
    PORT_RESTRICTED = "port_restricted"  # Only replied-to host+port
    SYMMETRIC = "symmetric"  # Different mapping per destination
    BLOCKED = "blocked"  # No UDP connectivity
    UNKNOWN = "unknown"  # Detection failed


@dataclass
class NATDetectionResult:
    """Result of NAT type detection."""

    nat_type: NATType
    external_ip: str | None = None
    external_port: int | None = None
    detection_time: float = 0.0
    stun_server_used: str | None = None
    error: str | None = None
    raw_results: dict[str, Any] = field(default_factory=dict)


@dataclass
class STUNResponse:
    """Parsed STUN response."""

    success: bool
    mapped_ip: str | None = None
    mapped_port: int | None = None
    response_origin_ip: str | None = None
    response_origin_port: int | None = None
    other_address_ip: str | None = None
    other_address_port: int | None = None
    error: str | None = None


class NATDetector:
    """
    Detects NAT type using STUN protocol.

    Uses multiple STUN servers and tests to determine:
    1. If we have any external connectivity
    2. What our external IP:port mapping is
    3. Whether the mapping changes per destination (symmetric)
    4. How restrictive our NAT is (cone type)
    """

    def __init__(
        self,
        stun_servers: list[tuple[str, int]] | None = None,
        timeout: float = 5.0,
        bind_port: int = 0,
    ):
        """Initialize NAT detector.

        Args:
            stun_servers: List of (host, port) STUN servers to use
            timeout: Timeout per STUN request
            bind_port: Local port to bind (0 = ephemeral)
        """
        self._stun_servers = stun_servers or DEFAULT_STUN_SERVERS
        self._timeout = timeout
        self._bind_port = bind_port
        self._socket: socket.socket | None = None
        self._transaction_id: bytes = b""

    async def detect(self) -> NATDetectionResult:
        """Detect NAT type using STUN probing.

        Returns:
            NATDetectionResult with detected NAT type and external address
        """
        start_time = time.time()

        try:
            # Create UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
            self._socket.bind(("0.0.0.0", self._bind_port))

            # Test 1: Basic connectivity to first STUN server
            result1 = await self._stun_request(self._stun_servers[0])
            if not result1.success:
                return NATDetectionResult(
                    nat_type=NATType.BLOCKED,
                    detection_time=time.time() - start_time,
                    error=result1.error,
                )

            external_ip = result1.mapped_ip
            external_port = result1.mapped_port

            # Test 2: Same request to second STUN server
            if len(self._stun_servers) > 1:
                result2 = await self._stun_request(self._stun_servers[1])
                if result2.success:
                    # If mapping changed, it's symmetric NAT
                    if (
                        result2.mapped_ip != external_ip
                        or result2.mapped_port != external_port
                    ):
                        return NATDetectionResult(
                            nat_type=NATType.SYMMETRIC,
                            external_ip=external_ip,
                            external_port=external_port,
                            detection_time=time.time() - start_time,
                            stun_server_used=f"{self._stun_servers[0][0]}:{self._stun_servers[0][1]}",
                            raw_results={
                                "test1": {"ip": external_ip, "port": external_port},
                                "test2": {
                                    "ip": result2.mapped_ip,
                                    "port": result2.mapped_port,
                                },
                            },
                        )

            # Test 3: Check if port matches local port (OPEN)
            local_port = self._socket.getsockname()[1]
            if external_port == local_port:
                # Could be OPEN or FULL_CONE, need more tests
                # For now, assume FULL_CONE if behind NAT
                if self._is_private_ip(self._get_local_ip()):
                    nat_type = NATType.FULL_CONE
                else:
                    nat_type = NATType.OPEN
            else:
                # Port mapping changed, likely PORT_RESTRICTED or RESTRICTED_CONE
                # Would need change-request STUN to distinguish
                nat_type = NATType.PORT_RESTRICTED

            return NATDetectionResult(
                nat_type=nat_type,
                external_ip=external_ip,
                external_port=external_port,
                detection_time=time.time() - start_time,
                stun_server_used=f"{self._stun_servers[0][0]}:{self._stun_servers[0][1]}",
            )

        except Exception as e:
            logger.error(f"NAT detection failed: {e}")
            return NATDetectionResult(
                nat_type=NATType.UNKNOWN,
                detection_time=time.time() - start_time,
                error=str(e),
            )
        finally:
            if self._socket:
                self._socket.close()
                self._socket = None

    async def _stun_request(
        self, server: tuple[str, int]
    ) -> STUNResponse:
        """Send STUN binding request and parse response.

        Args:
            server: (host, port) of STUN server

        Returns:
            STUNResponse with mapped address
        """
        try:
            # Generate transaction ID
            self._transaction_id = os.urandom(12)

            # Build STUN binding request
            request = self._build_binding_request()

            # Resolve server address
            try:
                server_ip = socket.gethostbyname(server[0])
            except socket.gaierror as e:
                return STUNResponse(success=False, error=f"DNS resolution failed: {e}")

            # Send request
            loop = asyncio.get_event_loop()
            await loop.sock_sendto(self._socket, request, (server_ip, server[1]))

            # Wait for response
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(self._socket, 1024),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                return STUNResponse(success=False, error="STUN request timeout")

            # Parse response
            return self._parse_response(data)

        except Exception as e:
            return STUNResponse(success=False, error=str(e))

    def _build_binding_request(self) -> bytes:
        """Build a STUN Binding Request message."""
        # Message type (2 bytes) + Length (2 bytes) + Magic Cookie (4 bytes) + Transaction ID (12 bytes)
        header = struct.pack(
            ">HHI",
            STUN_BINDING_REQUEST,
            0,  # Message length (no attributes)
            STUN_MAGIC_COOKIE,
        )
        return header + self._transaction_id

    def _parse_response(self, data: bytes) -> STUNResponse:
        """Parse STUN response and extract mapped address."""
        if len(data) < 20:
            return STUNResponse(success=False, error="Response too short")

        # Parse header
        msg_type, msg_len, magic = struct.unpack(">HHI", data[:8])
        transaction_id = data[8:20]

        if msg_type != STUN_BINDING_RESPONSE:
            return STUNResponse(
                success=False, error=f"Unexpected message type: {msg_type}"
            )

        if transaction_id != self._transaction_id:
            return STUNResponse(success=False, error="Transaction ID mismatch")

        # Parse attributes
        result = STUNResponse(success=True)
        offset = 20

        while offset < len(data):
            if offset + 4 > len(data):
                break

            attr_type, attr_len = struct.unpack(">HH", data[offset : offset + 4])
            offset += 4

            if offset + attr_len > len(data):
                break

            attr_data = data[offset : offset + attr_len]

            if attr_type == ATTR_XOR_MAPPED_ADDRESS:
                ip, port = self._parse_xor_mapped_address(attr_data)
                result.mapped_ip = ip
                result.mapped_port = port
            elif attr_type == ATTR_MAPPED_ADDRESS:
                ip, port = self._parse_mapped_address(attr_data)
                if not result.mapped_ip:  # Prefer XOR-mapped
                    result.mapped_ip = ip
                    result.mapped_port = port
            elif attr_type == ATTR_RESPONSE_ORIGIN:
                ip, port = self._parse_mapped_address(attr_data)
                result.response_origin_ip = ip
                result.response_origin_port = port
            elif attr_type == ATTR_OTHER_ADDRESS:
                ip, port = self._parse_mapped_address(attr_data)
                result.other_address_ip = ip
                result.other_address_port = port

            # Align to 4-byte boundary
            offset += attr_len + (4 - attr_len % 4) % 4

        return result

    def _parse_mapped_address(self, data: bytes) -> tuple[str | None, int | None]:
        """Parse MAPPED-ADDRESS attribute."""
        if len(data) < 8:
            return None, None

        family = data[1]
        port = struct.unpack(">H", data[2:4])[0]

        if family == 0x01:  # IPv4
            ip = socket.inet_ntoa(data[4:8])
        else:
            return None, None

        return ip, port

    def _parse_xor_mapped_address(self, data: bytes) -> tuple[str | None, int | None]:
        """Parse XOR-MAPPED-ADDRESS attribute."""
        if len(data) < 8:
            return None, None

        family = data[1]
        xor_port = struct.unpack(">H", data[2:4])[0]
        port = xor_port ^ (STUN_MAGIC_COOKIE >> 16)

        if family == 0x01:  # IPv4
            xor_ip = struct.unpack(">I", data[4:8])[0]
            ip_int = xor_ip ^ STUN_MAGIC_COOKIE
            ip = socket.inet_ntoa(struct.pack(">I", ip_int))
        else:
            return None, None

        return ip, port

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is a private address."""
        try:
            parts = [int(p) for p in ip.split(".")]
            if parts[0] == 10:
                return True
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            if parts[0] == 192 and parts[1] == 168:
                return True
            if parts[0] == 100 and 64 <= parts[1] <= 127:
                # Tailscale CGNAT range
                return True
            return False
        except Exception:
            return False


# Singleton for caching result
_cached_result: NATDetectionResult | None = None
_cache_time: float = 0.0
_cache_ttl: float = 300.0  # 5 minutes


async def detect_nat_type(
    force_refresh: bool = False,
    stun_servers: list[tuple[str, int]] | None = None,
) -> NATDetectionResult:
    """Detect NAT type (cached).

    Args:
        force_refresh: Force re-detection even if cached
        stun_servers: Optional list of STUN servers

    Returns:
        NATDetectionResult with NAT type and external address
    """
    global _cached_result, _cache_time

    if not force_refresh and _cached_result:
        if time.time() - _cache_time < _cache_ttl:
            return _cached_result

    detector = NATDetector(stun_servers=stun_servers)
    result = await detector.detect()

    _cached_result = result
    _cache_time = time.time()

    logger.info(
        f"NAT type detected: {result.nat_type.value} "
        f"(external: {result.external_ip}:{result.external_port})"
    )

    return result


def get_cached_nat_type() -> NATType:
    """Get cached NAT type without re-detection.

    Returns:
        Cached NATType or UNKNOWN if not cached
    """
    if _cached_result:
        return _cached_result.nat_type
    return NATType.UNKNOWN


def clear_nat_cache() -> None:
    """Clear the NAT detection cache."""
    global _cached_result, _cache_time
    _cached_result = None
    _cache_time = 0.0

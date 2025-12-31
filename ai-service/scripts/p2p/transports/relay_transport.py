"""
P2P relay transport implementation.

Tier 4 (RELAY): Route through P2P leader or other relay nodes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from ..transport_cascade import BaseTransport, TransportResult, TransportTier

logger = logging.getLogger(__name__)


class P2PRelayTransport(BaseTransport):
    """
    Relay transport through P2P leader or relay nodes.

    Tier 4 (RELAY): For nodes that can't be reached directly.
    The payload is sent to a relay node which forwards to the target.
    """

    name = "p2p_relay"
    tier = TransportTier.TIER_4_RELAY

    def __init__(
        self,
        relay_nodes: list[str] | None = None,
        port: int = 8770,
        timeout: float = 20.0,
    ):
        self._relay_nodes = relay_nodes or []
        self._port = port
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._leader_node: str | None = None

    def set_leader_node(self, leader_node: str) -> None:
        """Set the current P2P leader for relay."""
        self._leader_node = leader_node

    def add_relay_node(self, node: str) -> None:
        """Add a relay node."""
        if node not in self._relay_nodes:
            self._relay_nodes.append(node)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via relay."""
        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        # Build relay list: leader first, then configured relays
        relay_list = []
        if self._leader_node and self._leader_node != target:
            relay_list.append(self._leader_node)
        relay_list.extend([r for r in self._relay_nodes if r != target])

        if not relay_list:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="No relay nodes available",
            )

        # Try each relay in order
        errors = []
        for relay in relay_list:
            result = await self._send_via_relay(relay, target, payload)
            if result.success:
                return result
            errors.append(f"{relay}: {result.error}")

        return self._make_result(
            success=False,
            latency_ms=0,
            error=f"All relays failed: {'; '.join(errors)}",
        )

    async def _send_via_relay(
        self, relay: str, target: str, payload: bytes
    ) -> TransportResult:
        """Send payload through a specific relay node."""
        # Relay endpoint expects target in header
        url = f"http://{relay}:{self._port}/relay/forward"
        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.post(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Relay-Target": target,
                },
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                response_data = await resp.read()

                if resp.status == 200:
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=response_data,
                        relay_node=relay,
                    )
                else:
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=f"Relay HTTP {resp.status}",
                    )

        except Exception as e:
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    async def is_available(self, target: str) -> bool:
        """Check if relay transport is available."""
        # Available if we have any relay nodes
        return bool(self._leader_node) or bool(self._relay_nodes)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

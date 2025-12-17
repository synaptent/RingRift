"""P2P Orchestrator Network Utilities.

This module contains network-related utilities for the P2P orchestrator,
including HTTP client helpers and circuit breaker functions.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

# HTTP client imports
try:
    from aiohttp import ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    ClientSession = None
    ClientTimeout = None

# SOCKS proxy support for userspace Tailscale networking
try:
    from aiohttp_socks import ProxyConnector
    HAS_SOCKS = True
except ImportError:
    HAS_SOCKS = False
    ProxyConnector = None

# Get SOCKS proxy from environment (e.g., socks5://localhost:1055)
SOCKS_PROXY = os.environ.get("RINGRIFT_SOCKS_PROXY", "")

# Circuit breaker for fault-tolerant peer communication
try:
    from app.distributed.circuit_breaker import (
        get_host_breaker,
        CircuitOpenError,
        CircuitState,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_host_breaker = None
    CircuitOpenError = Exception
    CircuitState = None


class AsyncLockWrapper:
    """Synchronous context manager wrapper for threading locks in async code.

    This wraps a threading.RLock for use in async handlers. Since the critical
    sections protected by this lock are typically fast (reading/copying dicts),
    we use synchronous locking which briefly blocks the event loop but guarantees
    correct RLock semantics (same-thread acquire/release).

    For long-running critical sections, consider using asyncio.Lock instead.

    Usage:
        # Instead of: with self.peers_lock:
        async with AsyncLockWrapper(self.peers_lock):
            # ... critical section (keep it fast!)
    """

    def __init__(self, lock: threading.RLock):
        self._lock = lock

    async def __aenter__(self):
        # Synchronous acquire - blocks event loop briefly but guarantees correctness
        self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


def get_client_session(timeout: "ClientTimeout" = None) -> "ClientSession":
    """Create an aiohttp ClientSession with optional SOCKS proxy support.

    Args:
        timeout: Optional ClientTimeout configuration

    Returns:
        Configured ClientSession instance
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required for HTTP client functionality")

    if SOCKS_PROXY and HAS_SOCKS:
        connector = ProxyConnector.from_url(SOCKS_PROXY)
        return ClientSession(connector=connector, timeout=timeout)
    return ClientSession(timeout=timeout)


def check_peer_circuit(peer_host: str) -> bool:
    """Check if a peer's circuit is open.

    Args:
        peer_host: Hostname or IP of the peer

    Returns:
        True if request is allowed (circuit closed/half-open), False if open
    """
    if not HAS_CIRCUIT_BREAKER:
        return True
    return get_host_breaker().can_execute(peer_host)


def record_peer_success(peer_host: str) -> None:
    """Record successful communication with a peer.

    This helps the circuit breaker transition from half-open to closed.

    Args:
        peer_host: Hostname or IP of the peer
    """
    if HAS_CIRCUIT_BREAKER:
        get_host_breaker().record_success(peer_host)


def record_peer_failure(peer_host: str, error: Optional[Exception] = None) -> None:
    """Record failed communication with a peer.

    This increments the failure count and may trip the circuit breaker.

    Args:
        peer_host: Hostname or IP of the peer
        error: Optional exception that caused the failure
    """
    if HAS_CIRCUIT_BREAKER:
        get_host_breaker().record_failure(peer_host, error)


async def peer_request(
    session: "ClientSession",
    method: str,
    url: str,
    peer_host: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Make a circuit-breaker-protected request to a peer.

    This function wraps HTTP requests with circuit breaker protection,
    automatically tracking successes and failures for each peer.

    Args:
        session: aiohttp ClientSession to use
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        peer_host: Hostname/IP for circuit breaker tracking
        headers: Optional headers dict
        json: Optional JSON payload for POST/PUT
        timeout: Optional request timeout in seconds

    Returns:
        Response JSON if successful, None if circuit open or request failed
    """
    # Check circuit first
    if not check_peer_circuit(peer_host):
        return None

    try:
        kwargs = {"headers": headers} if headers else {}
        if json is not None:
            kwargs["json"] = json
        if timeout:
            kwargs["timeout"] = ClientTimeout(total=timeout)

        async with session.request(method, url, **kwargs) as resp:
            if resp.status == 200:
                record_peer_success(peer_host)
                return await resp.json()
            else:
                # Non-200 isn't necessarily a failure (might be expected)
                return {"status": resp.status, "error": await resp.text()}
    except Exception as e:
        record_peer_failure(peer_host, e)
        return None

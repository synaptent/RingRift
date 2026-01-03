"""P2P Orchestrator Network Utilities.

This module contains network-related utilities for the P2P orchestrator,
including HTTP client helpers and circuit breaker functions.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

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
        CircuitOpenError,
        CircuitState,
        get_host_breaker,
        # Per-transport circuit breakers (January 2026 - Phase 1)
        get_transport_breaker,
        check_transport_circuit,
        record_transport_success,
        record_transport_failure,
    )
    HAS_CIRCUIT_BREAKER = True
    HAS_TRANSPORT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    HAS_TRANSPORT_BREAKER = False
    get_host_breaker = None
    get_transport_breaker = None
    check_transport_circuit = None
    record_transport_success = None
    record_transport_failure = None
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


class TimeoutAsyncLockWrapper:
    """Async lock with configurable timeout to prevent deadlocks.

    December 30, 2025: Added to fix P2P cluster connectivity issues caused by
    indefinite lock acquisition blocking HTTP handlers and event loops.

    Unlike AsyncLockWrapper which wraps threading.RLock, this uses asyncio.Lock
    and supports timeout-based acquisition to prevent indefinite blocking.

    Usage:
        lock = TimeoutAsyncLockWrapper(timeout=5.0)

        # Context manager (raises TimeoutError on timeout)
        async with lock:
            # ... critical section

        # Manual acquisition (returns bool)
        if await lock.acquire(timeout=2.0):
            try:
                # ... critical section
            finally:
                lock.release()
        else:
            # Handle timeout gracefully
            pass
    """

    def __init__(self, timeout: float = 5.0):
        """Initialize lock with default timeout.

        Args:
            timeout: Default timeout in seconds for lock acquisition.
        """
        import asyncio
        self._lock = asyncio.Lock()
        self._timeout = timeout

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire lock with timeout.

        Args:
            timeout: Override timeout in seconds. Uses default if None.

        Returns:
            True if lock acquired, False if timeout exceeded.
        """
        import asyncio
        wait_time = timeout if timeout is not None else self._timeout
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=wait_time)
            return True
        except asyncio.TimeoutError:
            return False

    def release(self) -> None:
        """Release the lock."""
        self._lock.release()

    def locked(self) -> bool:
        """Check if lock is currently held."""
        return self._lock.locked()

    async def __aenter__(self):
        """Async context manager entry. Raises TimeoutError on timeout."""
        acquired = await self.acquire()
        if not acquired:
            import asyncio
            raise asyncio.TimeoutError(
                f"Lock acquisition timed out after {self._timeout}s"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
        return False


def get_client_session(timeout: ClientTimeout = None) -> ClientSession:
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


def record_peer_failure(peer_host: str, error: Exception | None = None) -> None:
    """Record failed communication with a peer.

    This increments the failure count and may trip the circuit breaker.

    Args:
        peer_host: Hostname or IP of the peer
        error: Optional exception that caused the failure
    """
    if HAS_CIRCUIT_BREAKER:
        get_host_breaker().record_failure(peer_host, error)


async def peer_request(
    session: ClientSession,
    method: str,
    url: str,
    peer_host: str,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
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


# =============================================================================
# Per-Transport Circuit Breaker Functions (January 2026 - Phase 1)
# =============================================================================

def check_peer_transport_circuit(peer_host: str, transport: str = "http") -> bool:
    """Check if a specific transport to a peer is available.

    Unlike check_peer_circuit() which uses a global breaker, this checks
    a per-(host, transport) circuit breaker, enabling failover between
    transports when one fails.

    January 2026: Created as part of P2P critical hardening (Phase 1).

    Args:
        peer_host: Hostname or IP of the peer
        transport: Transport type ("http", "ssh", "rsync", "p2p", etc.)

    Returns:
        True if transport is available (circuit closed/half-open), False if open
    """
    if not HAS_TRANSPORT_BREAKER:
        return check_peer_circuit(peer_host)  # Fallback to global breaker
    return check_transport_circuit(peer_host, transport)


def record_peer_transport_success(peer_host: str, transport: str = "http") -> None:
    """Record successful transport operation with a peer.

    Args:
        peer_host: Hostname or IP of the peer
        transport: Transport type ("http", "ssh", "rsync", etc.)
    """
    if HAS_TRANSPORT_BREAKER:
        record_transport_success(peer_host, transport)
    else:
        record_peer_success(peer_host)


def record_peer_transport_failure(
    peer_host: str,
    transport: str = "http",
    error: Exception | None = None
) -> None:
    """Record failed transport operation with a peer.

    Args:
        peer_host: Hostname or IP of the peer
        transport: Transport type ("http", "ssh", "rsync", etc.)
        error: Optional exception that caused the failure
    """
    if HAS_TRANSPORT_BREAKER:
        record_transport_failure(peer_host, transport, error)
    else:
        record_peer_failure(peer_host, error)


async def peer_request_with_transport(
    session: ClientSession,
    method: str,
    url: str,
    peer_host: str,
    transport: str = "http",
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """Make a transport-specific circuit-breaker-protected request to a peer.

    Like peer_request() but uses per-transport circuit breakers, enabling
    failover between transports when one fails.

    January 2026: Created as part of P2P critical hardening (Phase 1).

    Args:
        session: aiohttp ClientSession to use
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        peer_host: Hostname/IP for circuit breaker tracking
        transport: Transport type for circuit breaker isolation
        headers: Optional headers dict
        json: Optional JSON payload for POST/PUT
        timeout: Optional request timeout in seconds

    Returns:
        Response JSON if successful, None if circuit open or request failed
    """
    # Check transport-specific circuit first
    if not check_peer_transport_circuit(peer_host, transport):
        return None

    try:
        kwargs = {"headers": headers} if headers else {}
        if json is not None:
            kwargs["json"] = json
        if timeout:
            kwargs["timeout"] = ClientTimeout(total=timeout)

        async with session.request(method, url, **kwargs) as resp:
            if resp.status == 200:
                record_peer_transport_success(peer_host, transport)
                return await resp.json()
            else:
                # Non-200 isn't necessarily a failure (might be expected)
                return {"status": resp.status, "error": await resp.text()}
    except Exception as e:
        record_peer_transport_failure(peer_host, transport, e)
        return None

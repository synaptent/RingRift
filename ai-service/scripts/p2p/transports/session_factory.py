"""
Centralized HTTP session factory for P2P transports.

Jan 13, 2026: Created to fix CLOSE_WAIT connection leaks by ensuring all
HTTP sessions use consistent, properly-configured TCPConnectors with:
- Keepalive timeout coordination
- Connection cleanup on close
- Consistent timeout settings
- Proper resource lifecycle management

Root Cause Fixed:
- DirectHTTPTransport, TailscaleHTTPTransport, etc. were creating bare
  ClientSession() without TCPConnector, leading to:
  1. No keepalive timeout coordination with server
  2. Connections left in CLOSE_WAIT when client closes before server
  3. Inconsistent timeout behavior across transports

Usage:
    from .session_factory import create_managed_session, ManagedSessionContext

    # Option 1: Context manager (preferred - automatic cleanup)
    async with ManagedSessionContext(timeout=10.0) as session:
        async with session.get(url) as resp:
            data = await resp.json()

    # Option 2: Manual management (when session reuse needed)
    session = await create_managed_session(timeout=10.0)
    try:
        # use session...
    finally:
        await close_session(session)
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from contextlib import asynccontextmanager
from typing import AsyncIterator

try:
    import aiohttp
    from aiohttp import ClientTimeout, TCPConnector
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore
    TCPConnector = None  # type: ignore

logger = logging.getLogger(__name__)

# Default configuration from environment
# January 2026 - P2P Stability Plan Phase 3: Increased connection limit from 100 to 200
# Supports 20+ node clusters with multiple concurrent connections per node
DEFAULT_KEEPALIVE_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_KEEPALIVE_TIMEOUT", "30"))
DEFAULT_CONNECTION_LIMIT = int(os.environ.get("RINGRIFT_P2P_CONNECTION_LIMIT", "200"))
DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("RINGRIFT_P2P_CONNECT_TIMEOUT", "10.0"))
DEFAULT_TOTAL_TIMEOUT = float(os.environ.get("RINGRIFT_P2P_TOTAL_TIMEOUT", "30.0"))
DNS_CACHE_TTL = int(os.environ.get("RINGRIFT_P2P_DNS_CACHE_TTL", "300"))


async def create_managed_session(
    timeout: float = DEFAULT_TOTAL_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    keepalive_timeout: int = DEFAULT_KEEPALIVE_TIMEOUT,
    limit: int = DEFAULT_CONNECTION_LIMIT,
    enable_keepalive: bool = True,
) -> aiohttp.ClientSession:
    """
    Create a properly-configured ClientSession for P2P communication.

    This function centralizes HTTP session configuration to ensure:
    1. Consistent keepalive settings across all transports
    2. Proper connection cleanup on close (enable_cleanup_closed=True)
    3. DNS caching to reduce lookup latency
    4. Connection limits to prevent resource exhaustion

    Args:
        timeout: Total request timeout in seconds
        connect_timeout: TCP connection timeout in seconds
        keepalive_timeout: Keepalive timeout in seconds (0 to disable)
        limit: Maximum concurrent connections
        enable_keepalive: Whether to enable HTTP keepalive

    Returns:
        Configured ClientSession with proper lifecycle management

    Example:
        session = await create_managed_session(timeout=10.0)
        try:
            async with session.get("http://peer:8770/health") as resp:
                return resp.status == 200
        finally:
            await close_session(session)
    """
    if aiohttp is None:
        raise RuntimeError("aiohttp required for P2P HTTP transport")

    # Configure TCP connector with proper keepalive and cleanup
    connector = TCPConnector(
        # Connection limits
        limit=limit,
        limit_per_host=min(limit, 10),  # Prevent single host from consuming all slots
        # Keepalive configuration - CRITICAL for preventing CLOSE_WAIT
        keepalive_timeout=keepalive_timeout if enable_keepalive else 0,
        enable_cleanup_closed=True,  # CRITICAL: Clean up closed connections
        # DNS configuration
        ttl_dns_cache=DNS_CACHE_TTL,
        use_dns_cache=True,
        # Socket configuration
        family=socket.AF_UNSPEC,  # Support both IPv4 and IPv6
        # SSL - use default context
        ssl=None,
    )

    # Configure timeouts
    client_timeout = ClientTimeout(
        total=timeout,
        connect=connect_timeout,
        sock_connect=connect_timeout,
        sock_read=timeout,
    )

    return aiohttp.ClientSession(
        connector=connector,
        timeout=client_timeout,
        # Don't raise for status - let caller handle
        raise_for_status=False,
    )


async def close_session(session: aiohttp.ClientSession | None) -> None:
    """
    Safely close an aiohttp ClientSession.

    Handles edge cases:
    - None session
    - Already closed session
    - Exceptions during close

    Args:
        session: Session to close, or None
    """
    if session is None:
        return

    if session.closed:
        return

    try:
        await session.close()
        # Give time for graceful connection shutdown
        # This prevents CLOSE_WAIT by allowing FIN/ACK handshake to complete
        await asyncio.sleep(0.1)
    except Exception as e:
        logger.debug(f"Error closing session: {e}")


@asynccontextmanager
async def ManagedSessionContext(
    timeout: float = DEFAULT_TOTAL_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    keepalive_timeout: int = DEFAULT_KEEPALIVE_TIMEOUT,
    limit: int = DEFAULT_CONNECTION_LIMIT,
    enable_keepalive: bool = True,
) -> AsyncIterator[aiohttp.ClientSession]:
    """
    Context manager for managed HTTP sessions.

    Guarantees cleanup even if an exception occurs. This is the preferred
    way to use HTTP sessions in the P2P layer.

    Args:
        timeout: Total request timeout in seconds
        connect_timeout: TCP connection timeout in seconds
        keepalive_timeout: Keepalive timeout in seconds
        limit: Maximum concurrent connections
        enable_keepalive: Whether to enable HTTP keepalive

    Yields:
        Configured ClientSession

    Example:
        async with ManagedSessionContext(timeout=5.0) as session:
            async with session.get("http://peer:8770/health") as resp:
                if resp.status == 200:
                    return await resp.json()
    """
    session = await create_managed_session(
        timeout=timeout,
        connect_timeout=connect_timeout,
        keepalive_timeout=keepalive_timeout,
        limit=limit,
        enable_keepalive=enable_keepalive,
    )
    try:
        yield session
    finally:
        await close_session(session)


class ReusableSessionManager:
    """
    Manager for reusable sessions in long-lived transports.

    For transports that need to reuse sessions across multiple requests
    (like DirectHTTPTransport), this class provides lifecycle management
    with automatic recreation on errors.

    Usage:
        class MyTransport:
            def __init__(self):
                self._session_manager = ReusableSessionManager(timeout=10.0)

            async def send(self, url: str) -> bytes:
                session = await self._session_manager.get_session()
                async with session.get(url) as resp:
                    return await resp.read()

            async def close(self):
                await self._session_manager.close()
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TOTAL_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        keepalive_timeout: int = DEFAULT_KEEPALIVE_TIMEOUT,
        limit: int = DEFAULT_CONNECTION_LIMIT,
        enable_keepalive: bool = True,
    ):
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._keepalive_timeout = keepalive_timeout
        self._limit = limit
        self._enable_keepalive = enable_keepalive
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()
        self._error_count = 0
        self._max_errors_before_recreate = 3

    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get or create a managed session.

        Thread-safe via lock. Automatically recreates session if:
        - Session is None
        - Session is closed
        - Too many errors have occurred (suggests connection issues)

        Returns:
            Active ClientSession
        """
        async with self._lock:
            if self._should_recreate():
                await self._recreate_session()
            return self._session  # type: ignore

    def mark_error(self) -> None:
        """
        Mark that an error occurred using the session.

        After _max_errors_before_recreate errors, the session will be
        recreated on the next get_session() call.
        """
        self._error_count += 1

    def mark_success(self) -> None:
        """Mark that a request succeeded, resetting error count."""
        self._error_count = 0

    async def close(self) -> None:
        """Close the managed session."""
        async with self._lock:
            await close_session(self._session)
            self._session = None
            self._error_count = 0

    def _should_recreate(self) -> bool:
        """Check if session should be recreated."""
        if self._session is None:
            return True
        if self._session.closed:
            return True
        if self._error_count >= self._max_errors_before_recreate:
            logger.info(
                f"Recreating session after {self._error_count} errors"
            )
            return True
        return False

    async def _recreate_session(self) -> None:
        """Close existing session and create a new one."""
        await close_session(self._session)
        self._session = await create_managed_session(
            timeout=self._timeout,
            connect_timeout=self._connect_timeout,
            keepalive_timeout=self._keepalive_timeout,
            limit=self._limit,
            enable_keepalive=self._enable_keepalive,
        )
        self._error_count = 0


# Convenience exports
__all__ = [
    "create_managed_session",
    "close_session",
    "ManagedSessionContext",
    "ReusableSessionManager",
    "DEFAULT_KEEPALIVE_TIMEOUT",
    "DEFAULT_CONNECTION_LIMIT",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_TOTAL_TIMEOUT",
]

"""
Connection Pooling Module.

Dec 30, 2025: Part of Phase 5 - Enhanced Transport Failover.

Provides persistent connection pooling for reduced latency to frequently
accessed peers. Supports HTTP sessions with automatic health checking.

Usage:
    from scripts.p2p.connection_pool import (
        PeerConnectionPool,
        get_connection_pool,
        get_pooled_session,
    )

    # Get a pooled session for a peer
    pool = get_connection_pool()
    async with pool.get_session("node-1") as session:
        async with session.get(f"http://{peer_addr}/status") as resp:
            data = await resp.json()

    # Or use convenience function
    session = await get_pooled_session("node-1")
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# Jan 16, 2026: Use centralized provider timeout configuration
from app.config.provider_timeouts import ProviderTimeouts

logger = logging.getLogger(__name__)

# Pool configuration
# Jan 16, 2026: Increased from 3/100 to 8/250 to reduce connection exhaustion
# on 40+ node clusters during high-activity periods (model sync, gauntlet runs)
# Jan 20, 2026: Made idle timeout configurable, reduced default from 300s to 120s
# to prevent connection pool exhaustion (40 nodes × 8 conn = 320 > pool limit 250)
DEFAULT_MAX_CONNECTIONS_PER_PEER = 8
DEFAULT_MAX_TOTAL_CONNECTIONS = 250
DEFAULT_CONNECTION_TIMEOUT = 30.0  # seconds (base timeout, multiplied by provider factor)
DEFAULT_IDLE_TIMEOUT = float(os.environ.get("RINGRIFT_P2P_CONNECTION_IDLE_TIMEOUT", "120.0"))
DEFAULT_HEALTH_CHECK_INTERVAL = 60.0  # 1 minute

# =============================================================================
# Provider-Specific Connection Timeout Multipliers (Phase 7.11 - Jan 5, 2026)
# =============================================================================
#
# Problem: Connection pool uses hardcoded 30s timeout while SSH transport uses
# provider-specific multipliers. This causes timeouts on slow providers (Vast.ai,
# Lambda NAT relay) while being overly generous on fast providers.
#
# Solution: Apply provider-specific multipliers to align with peer_recovery_loop.py
# and ssh_transport.py timeout behavior for consistent network operations.
#
# Jan 16, 2026: Now delegated to centralized ProviderTimeouts config.
# These aliases are kept for backward compatibility with existing callers.

# Backward-compatible alias using centralized config
PROVIDER_TIMEOUT_MULTIPLIERS: dict[str, float] = ProviderTimeouts.MULTIPLIERS

# Default multiplier for unknown providers (conservative)
DEFAULT_PROVIDER_MULTIPLIER: float = ProviderTimeouts.DEFAULT_MULTIPLIER


def _extract_provider_from_peer_id(peer_id: str) -> str:
    """Extract provider name from peer_id prefix.

    Jan 16, 2026: Delegated to centralized ProviderTimeouts.extract_provider().
    """
    return ProviderTimeouts.extract_provider(peer_id)


def get_provider_connection_timeout(
    peer_id: str, base_timeout: float = DEFAULT_CONNECTION_TIMEOUT
) -> float:
    """Get provider-specific connection timeout for a peer.

    Phase 7.11 (Jan 5, 2026): Aligns connection pool timeouts with provider
    multipliers used in SSH transport and peer recovery for consistent
    behavior across all P2P operations.

    Jan 16, 2026: Delegated to centralized ProviderTimeouts.get_timeout().

    Args:
        peer_id: Peer identifier to look up provider for
        base_timeout: Base timeout to multiply (default: DEFAULT_CONNECTION_TIMEOUT)

    Returns:
        Provider-appropriate connection timeout in seconds
    """
    return ProviderTimeouts.get_timeout(peer_id, base_timeout)


@dataclass
class ConnectionConfig:
    """Configuration for the connection pool."""

    max_connections_per_peer: int = DEFAULT_MAX_CONNECTIONS_PER_PEER
    max_total_connections: int = DEFAULT_MAX_TOTAL_CONNECTIONS
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT
    health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL
    enable_keepalive: bool = True
    keepalive_timeout: int = 30


@dataclass
class DynamicPoolConfig:
    """Dynamic pool limits that scale with cluster size.

    January 20, 2026: Added to replace static 250 connection limit.
    The static limit caused connection exhaustion with 40+ node clusters:
    40 nodes × 8 connections = 320 > 250 limit = exhaustion

    This config automatically scales limits based on cluster size.
    """

    base_per_peer: int = 6  # Base connections per peer
    max_per_peer: int = 12  # Maximum connections per peer (for large clusters)
    cluster_multiplier: float = 8.0  # Connections per node in cluster
    min_total: int = 200  # Minimum total connections
    max_total: int = 600  # Maximum total connections (prevents unbounded growth)
    resize_interval: float = 60.0  # Minimum seconds between resizes

    def get_limits(self, cluster_size: int) -> tuple[int, int]:
        """Calculate pool limits for the given cluster size.

        Args:
            cluster_size: Number of nodes in the cluster

        Returns:
            Tuple of (per_peer_limit, total_limit)
        """
        # Per-peer: 6 base, +2 per 50 nodes, max 12
        per_peer = min(
            self.max_per_peer,
            self.base_per_peer + (cluster_size // 50) * 2
        )

        # Total: 8 per node, clamped to [200, 600]
        total = int(max(
            self.min_total,
            min(cluster_size * self.cluster_multiplier, self.max_total)
        ))

        return per_peer, total


@dataclass
class PooledConnection:
    """A pooled connection to a peer."""

    peer_id: str
    session: Any  # aiohttp.ClientSession
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_healthy: bool = True

    def mark_used(self) -> None:
        """Mark the connection as recently used."""
        self.last_used = time.time()
        self.use_count += 1

    def is_idle(self, timeout: float) -> bool:
        """Check if connection has been idle too long."""
        return time.time() - self.last_used > timeout

    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at


@dataclass
class PeerPoolStats:
    """Statistics for a peer's connection pool."""

    peer_id: str
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_health_check: float = 0.0
    health_status: str = "unknown"


class PeerConnectionPool:
    """
    Connection pool for persistent HTTP sessions to peers.

    Features:
    - Per-peer connection limits
    - Automatic idle connection cleanup
    - Health checking of pooled connections
    - Thread-safe access
    - Dynamic pool sizing based on cluster size (Jan 2026)
    """

    _instance: "PeerConnectionPool | None" = None
    _lock = asyncio.Lock()

    def __new__(cls, config: ConnectionConfig | None = None) -> "PeerConnectionPool":
        # Note: Singleton pattern, but __init__ handles configuration
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: ConnectionConfig | None = None) -> None:
        if self._initialized:
            return

        self._config = config or ConnectionConfig()
        self._pools: dict[str, asyncio.Queue[PooledConnection]] = {}
        self._active_connections: dict[str, set[PooledConnection]] = {}
        self._stats: dict[str, PeerPoolStats] = {}
        self._data_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._running = False
        self._initialized = True

        # Jan 20, 2026: Dynamic pool sizing based on cluster size
        self._dynamic_config = DynamicPoolConfig()
        self._get_cluster_size: callable | None = None
        self._last_resize: float = 0.0
        self._resize_count: int = 0

        logger.debug(
            f"PeerConnectionPool initialized: "
            f"max_per_peer={self._config.max_connections_per_peer}, "
            f"max_total={self._config.max_total_connections}"
        )

    def set_cluster_size_callback(self, callback: callable) -> None:
        """Set callback to get current cluster size for dynamic scaling.

        Args:
            callback: Callable that returns the current cluster size (int)
        """
        self._get_cluster_size = callback
        logger.info("Connection pool cluster size callback configured")

    async def start(self) -> None:
        """Start background tasks for cleanup and health checks."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.debug("Connection pool background tasks started")

    async def stop(self) -> None:
        """Stop background tasks and close all connections."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all pooled connections
        await self._close_all_connections()
        logger.debug("Connection pool stopped")

    async def _close_all_connections(self) -> None:
        """Close all pooled connections."""
        async with self._data_lock:
            for peer_id, pool in self._pools.items():
                while not pool.empty():
                    try:
                        conn = pool.get_nowait()
                        await self._close_connection(conn)
                    except asyncio.QueueEmpty:
                        break

            for peer_id, active in self._active_connections.items():
                for conn in list(active):
                    await self._close_connection(conn)

            self._pools.clear()
            self._active_connections.clear()

    async def _close_connection(self, conn: PooledConnection) -> None:
        """Close a single pooled connection."""
        try:
            if conn.session and not conn.session.closed:
                await conn.session.close()
        except Exception as e:
            logger.debug(f"Error closing connection to {conn.peer_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        while self._running:
            try:
                await asyncio.sleep(self._config.idle_timeout / 2)
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Connection cleanup error: {e}")

    async def _health_check_loop(self) -> None:
        """Background task to health check connections."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await self._health_check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Health check error: {e}")

    def _maybe_resize_pool(self) -> None:
        """Resize pool based on current cluster size.

        January 20, 2026: Added for dynamic pool sizing.
        Called during cleanup loop (every idle_timeout/2 seconds).
        Actual resize only happens if resize_interval has passed.
        """
        if not self._get_cluster_size:
            return  # No callback configured

        now = time.time()
        if now - self._last_resize < self._dynamic_config.resize_interval:
            return  # Too soon since last resize

        try:
            cluster_size = self._get_cluster_size()
            if cluster_size <= 0:
                return  # Invalid cluster size

            per_peer, total = self._dynamic_config.get_limits(cluster_size)

            # Check if resize is needed
            if (total != self._config.max_total_connections or
                    per_peer != self._config.max_connections_per_peer):
                old_total = self._config.max_total_connections
                old_per_peer = self._config.max_connections_per_peer

                self._config.max_total_connections = total
                self._config.max_connections_per_peer = per_peer
                self._resize_count += 1

                logger.info(
                    f"Connection pool resized: "
                    f"total {old_total} -> {total}, "
                    f"per_peer {old_per_peer} -> {per_peer} "
                    f"(cluster size: {cluster_size}, resize #{self._resize_count})"
                )

            self._last_resize = now

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Pool resize error: {e}")

    def get_pool_stats(self) -> dict[str, Any]:
        """Get pool statistics for monitoring.

        Returns:
            Dict with pool configuration and stats
        """
        total_pooled = sum(p.qsize() for p in self._pools.values())
        total_active = sum(len(a) for a in self._active_connections.values())

        return {
            "max_per_peer": self._config.max_connections_per_peer,
            "max_total": self._config.max_total_connections,
            "current_pooled": total_pooled,
            "current_active": total_active,
            "current_total": total_pooled + total_active,
            "peer_count": len(self._pools),
            "dynamic_scaling_enabled": self._get_cluster_size is not None,
            "resize_count": self._resize_count,
            "last_resize": self._last_resize,
        }

    async def _cleanup_idle_connections(self) -> None:
        """Remove connections that have been idle too long."""
        # Jan 20, 2026: Check if pool needs resizing based on cluster size
        self._maybe_resize_pool()

        async with self._data_lock:
            for peer_id, pool in list(self._pools.items()):
                # Check each connection in the pool
                connections_to_keep = []
                while not pool.empty():
                    try:
                        conn = pool.get_nowait()
                        if conn.is_idle(self._config.idle_timeout):
                            await self._close_connection(conn)
                            logger.debug(
                                f"Closed idle connection to {peer_id} "
                                f"(idle {time.time() - conn.last_used:.0f}s)"
                            )
                        else:
                            connections_to_keep.append(conn)
                    except asyncio.QueueEmpty:
                        break

                # Put back non-idle connections
                for conn in connections_to_keep:
                    await pool.put(conn)

    async def _health_check_all(self) -> None:
        """Health check all pooled connections."""
        async with self._data_lock:
            for peer_id in list(self._pools.keys()):
                await self._health_check_peer(peer_id)

    async def _health_check_peer(self, peer_id: str) -> bool:
        """Health check connections to a specific peer.

        January 4, 2026 - Sprint 17.10: Added CB pre-check to skip
        health checks on circuit-broken peers.
        """
        if peer_id not in self._stats:
            self._stats[peer_id] = PeerPoolStats(peer_id=peer_id)

        stats = self._stats[peer_id]
        stats.last_health_check = time.time()

        # Skip health check if peer has OPEN circuit breaker
        try:
            from scripts.p2p.health_coordinator import get_health_coordinator

            coordinator = get_health_coordinator()
            if coordinator.is_node_circuit_broken(peer_id):
                stats.health_status = "circuit_broken"
                return False
        except ImportError:
            pass  # Health coordinator not available

        # Check if we have any connections
        pool = self._pools.get(peer_id)
        if pool and not pool.empty():
            stats.health_status = "healthy"
            return True

        stats.health_status = "no_connections"
        return False

    @asynccontextmanager
    async def get_session(
        self,
        peer_id: str,
        peer_address: str | None = None,
    ) -> AsyncIterator[Any]:
        """Get a pooled session for a peer.

        Args:
            peer_id: Target peer node ID
            peer_address: Optional peer address (host:port) for new connections

        Yields:
            aiohttp.ClientSession

        Usage:
            async with pool.get_session("node-1", "192.168.1.10:8770") as session:
                async with session.get("/status") as resp:
                    ...
        """
        conn = await self._acquire_connection(peer_id, peer_address)
        try:
            yield conn.session
        finally:
            await self._release_connection(conn)

    async def _acquire_connection(
        self,
        peer_id: str,
        peer_address: str | None = None,
    ) -> PooledConnection:
        """Acquire a connection from the pool or create a new one."""
        async with self._data_lock:
            # Initialize pool for peer if needed
            if peer_id not in self._pools:
                self._pools[peer_id] = asyncio.Queue(
                    maxsize=self._config.max_connections_per_peer
                )
                self._active_connections[peer_id] = set()
                self._stats[peer_id] = PeerPoolStats(peer_id=peer_id)

            pool = self._pools[peer_id]
            active = self._active_connections[peer_id]
            stats = self._stats[peer_id]

            # Try to get an existing connection
            try:
                conn = pool.get_nowait()
                if conn.is_healthy:
                    conn.mark_used()
                    active.add(conn)
                    stats.active_connections = len(active)
                    stats.idle_connections = pool.qsize()
                    stats.total_requests += 1
                    return conn
                else:
                    # Connection unhealthy, close it
                    await self._close_connection(conn)
            except asyncio.QueueEmpty:
                pass

            # Need to create a new connection
            if len(active) >= self._config.max_connections_per_peer:
                # Wait for a connection to be released
                conn = await asyncio.wait_for(
                    pool.get(),
                    timeout=self._config.connection_timeout,
                )
                conn.mark_used()
                active.add(conn)
                stats.active_connections = len(active)
                stats.idle_connections = pool.qsize()
                stats.total_requests += 1
                return conn

            # Check global limit before creating new connection
            total_connections = self._get_total_connections()
            if total_connections >= self._config.max_total_connections:
                # Global limit reached - wait for any connection to be released
                logger.debug(
                    f"Global connection limit reached ({total_connections}/{self._config.max_total_connections}), "
                    f"waiting for connection to {peer_id}"
                )
                # Wait on this peer's pool - connections will be returned as others finish
                conn = await asyncio.wait_for(
                    pool.get(),
                    timeout=self._config.connection_timeout,
                )
                conn.mark_used()
                active.add(conn)
                stats.active_connections = len(active)
                stats.idle_connections = pool.qsize()
                stats.total_requests += 1
                return conn

            # Create new connection
            conn = await self._create_connection(peer_id, peer_address)
            active.add(conn)
            stats.active_connections = len(active)
            stats.total_requests += 1
            return conn

    async def _release_connection(self, conn: PooledConnection) -> None:
        """Release a connection back to the pool."""
        async with self._data_lock:
            peer_id = conn.peer_id

            if peer_id not in self._active_connections:
                # Pool was cleared, just close
                await self._close_connection(conn)
                return

            active = self._active_connections[peer_id]
            pool = self._pools[peer_id]
            stats = self._stats[peer_id]

            active.discard(conn)

            if conn.is_healthy:
                try:
                    pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool full, close this connection
                    await self._close_connection(conn)
            else:
                await self._close_connection(conn)

            stats.active_connections = len(active)
            stats.idle_connections = pool.qsize()

    async def _create_connection(
        self,
        peer_id: str,
        peer_address: str | None = None,
    ) -> PooledConnection:
        """Create a new connection to a peer.

        Phase 7.11 (Jan 5, 2026): Uses provider-specific timeouts for consistent
        behavior across all P2P operations. Different providers have different
        network characteristics:
        - Vast.ai: 60s (consumer networks, high variance)
        - Lambda: 45s (NAT relay latency)
        - Nebius/Hetzner: 30s (stable cloud/bare metal)
        """
        try:
            import aiohttp

            # Configure connector with keepalive
            connector = aiohttp.TCPConnector(
                limit=1,  # Single connection per session
                keepalive_timeout=self._config.keepalive_timeout if self._config.enable_keepalive else 0,
                enable_cleanup_closed=True,
            )

            # Phase 7.11: Use provider-specific timeout instead of fixed value
            provider_timeout = get_provider_connection_timeout(
                peer_id, self._config.connection_timeout
            )

            timeout = aiohttp.ClientTimeout(
                total=provider_timeout,
                connect=10.0,
            )

            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )

            conn = PooledConnection(
                peer_id=peer_id,
                session=session,
            )

            logger.debug(
                f"Created new connection to {peer_id} "
                f"(timeout={provider_timeout:.1f}s)"
            )
            return conn

        except ImportError:
            # aiohttp not available
            logger.warning("aiohttp not available for connection pooling")
            raise RuntimeError("aiohttp required for connection pooling")

    def _get_total_connections(self) -> int:
        """Get total connection count across all peers.

        Returns:
            Total number of active + idle connections.
        """
        total_active = sum(len(active) for active in self._active_connections.values())
        total_idle = sum(pool.qsize() for pool in self._pools.values())
        return total_active + total_idle

    def mark_connection_unhealthy(self, peer_id: str) -> None:
        """Mark all connections to a peer as unhealthy.

        Called when a peer is known to be down.
        """
        if peer_id in self._stats:
            self._stats[peer_id].health_status = "unhealthy"

    def get_stats(self, peer_id: str | None = None) -> dict[str, Any]:
        """Get pool statistics.

        Args:
            peer_id: Optional peer to get stats for. If None, returns all.

        Returns:
            Statistics dictionary
        """
        if peer_id:
            stats = self._stats.get(peer_id)
            if stats:
                return {
                    "peer_id": stats.peer_id,
                    "active_connections": stats.active_connections,
                    "idle_connections": stats.idle_connections,
                    "total_requests": stats.total_requests,
                    "failed_requests": stats.failed_requests,
                    "health_status": stats.health_status,
                }
            return {}

        return {
            "total_peers": len(self._pools),
            "total_active": sum(s.active_connections for s in self._stats.values()),
            "total_idle": sum(s.idle_connections for s in self._stats.values()),
            "peers": {
                pid: {
                    "active": s.active_connections,
                    "idle": s.idle_connections,
                    "requests": s.total_requests,
                    "health": s.health_status,
                }
                for pid, s in self._stats.items()
            },
        }

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:
            await cls._instance.stop()
            cls._instance = None


# Module-level convenience functions
_pool: PeerConnectionPool | None = None


def get_connection_pool(config: ConnectionConfig | None = None) -> PeerConnectionPool:
    """Get the global connection pool singleton."""
    global _pool
    if _pool is None:
        _pool = PeerConnectionPool(config)
    return _pool


async def get_pooled_session(
    peer_id: str,
    peer_address: str | None = None,
) -> Any:
    """Get a pooled session for a peer.

    Note: This returns a context manager. Use as:
        async with get_pooled_session("node-1") as session:
            ...
    """
    pool = get_connection_pool()
    return pool.get_session(peer_id, peer_address)


async def start_connection_pool() -> None:
    """Start the connection pool background tasks."""
    pool = get_connection_pool()
    await pool.start()


async def stop_connection_pool() -> None:
    """Stop the connection pool and close all connections."""
    pool = get_connection_pool()
    await pool.stop()


@asynccontextmanager
async def get_client_session(
    timeout: "aiohttp.ClientTimeout | None" = None,
) -> "AsyncIterator[aiohttp.ClientSession]":
    """Get a simple aiohttp client session with optional timeout.

    This is a convenience function for one-off HTTP requests that don't need
    connection pooling. Use get_pooled_session() for repeated requests to
    the same peer.

    Usage:
        async with get_client_session(aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url) as resp:
                ...
    """
    import aiohttp
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        yield session

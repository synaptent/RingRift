"""P2P Orchestrator Network Utilities.

This module contains network-related utilities for the P2P orchestrator,
including HTTP client helpers and circuit breaker functions.
Extracted from p2p_orchestrator.py for better modularity.

January 10, 2026: Added NonBlockingAsyncLockWrapper and lock ordering protocol
to fix lock contention issues on 40+ node clusters.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from .models import NodeInfo

# Generic type for PeerSnapshot to work with any peer info type
T = TypeVar("T")

logger = logging.getLogger(__name__)

# =============================================================================
# Lock Ordering Protocol (January 10, 2026)
# =============================================================================
# To prevent deadlocks on large clusters, locks must always be acquired in this
# order (lower number = acquire first). Never acquire a lower-numbered lock
# while holding a higher-numbered one.
#
# Example: If holding jobs_lock (2), you can acquire training_lock (3), but not
# peers_lock (1). If you need both, acquire peers_lock first, then jobs_lock.

LOCK_ORDER: dict[str, int] = {
    "peers_lock": 1,
    "jobs_lock": 2,
    "training_lock": 3,
    "manifest_lock": 4,
    "sync_lock": 5,
    "relay_lock": 6,
    "ssh_tournament_lock": 7,
}

# Thread-local storage to track currently held locks for ordering validation
_held_locks: threading.local = threading.local()

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

    DEPRECATED: Use NonBlockingAsyncLockWrapper instead, which doesn't block
    the event loop during lock acquisition.

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


class NonBlockingAsyncLockWrapper:
    """Non-blocking async wrapper for threading.RLock with timeout support.

    January 10, 2026: Created to fix lock contention issues on 40+ node clusters.
    Unlike AsyncLockWrapper which blocks the event loop during acquire, this
    wrapper uses asyncio.to_thread() for non-blocking lock acquisition.

    Features:
    - Non-blocking acquire via asyncio.to_thread()
    - Configurable timeout to prevent indefinite waiting
    - Lock ordering validation to prevent deadlocks
    - Debug logging for lock contention analysis

    Usage:
        # Basic usage (recommended for P2P handlers)
        async with NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
            # ... critical section

        # Manual acquisition with timeout check
        wrapper = NonBlockingAsyncLockWrapper(self.peers_lock, "peers_lock", timeout=5.0)
        if await wrapper.acquire():
            try:
                # ... critical section
            finally:
                wrapper.release()
        else:
            # Handle timeout gracefully
            logger.warning("Lock acquisition timed out")
    """

    def __init__(
        self,
        lock: threading.RLock,
        lock_name: str = "unknown",
        timeout: float = 10.0,  # Jan 2026: Increased from 5s to 10s for 40+ node clusters
        validate_order: bool = True,
    ):
        """Initialize non-blocking lock wrapper.

        Args:
            lock: The threading.RLock to wrap
            lock_name: Name of the lock (for ordering validation and logging)
            timeout: Maximum time to wait for lock acquisition in seconds
            validate_order: If True, validate lock ordering to prevent deadlocks
        """
        self._lock = lock
        self._lock_name = lock_name
        self._timeout = timeout
        self._validate_order = validate_order
        self._acquired = False

    def _get_held_locks(self) -> set[str]:
        """Get the set of locks currently held by this thread."""
        if not hasattr(_held_locks, "locks"):
            _held_locks.locks = set()
        return _held_locks.locks

    def _check_lock_order(self) -> bool:
        """Check if acquiring this lock would violate ordering protocol.

        Returns:
            True if ordering is valid, False if violation detected.
        """
        if not self._validate_order:
            return True

        held = self._get_held_locks()
        if not held:
            return True

        my_order = LOCK_ORDER.get(self._lock_name, 999)
        for held_lock in held:
            held_order = LOCK_ORDER.get(held_lock, 999)
            if held_order > my_order:
                logger.warning(
                    f"Lock ordering violation: attempting to acquire {self._lock_name} "
                    f"(order={my_order}) while holding {held_lock} (order={held_order}). "
                    f"This may cause deadlocks on large clusters."
                )
                return False
        return True

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire the lock with timeout, non-blocking for event loop.

        Args:
            timeout: Override timeout in seconds. Uses default if None.

        Returns:
            True if lock acquired, False if timeout exceeded.
        """
        # Check lock ordering before attempting acquire
        if not self._check_lock_order():
            # Log warning but still attempt acquire (don't break existing code)
            pass

        wait_time = timeout if timeout is not None else self._timeout

        def _blocking_acquire() -> bool:
            """Blocking acquire in thread pool."""
            return self._lock.acquire(blocking=True, timeout=wait_time)

        try:
            # Run blocking acquire in thread pool to avoid blocking event loop
            acquired = await asyncio.wait_for(
                asyncio.to_thread(_blocking_acquire),
                timeout=wait_time + 1.0,  # Slight buffer for thread scheduling
            )
            if acquired:
                self._acquired = True
                self._get_held_locks().add(self._lock_name)
            return acquired
        except asyncio.TimeoutError:
            logger.debug(f"Lock {self._lock_name} acquisition timed out after {wait_time}s")
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._acquired:
            self._lock.release()
            self._get_held_locks().discard(self._lock_name)
            self._acquired = False

    async def __aenter__(self):
        """Async context manager entry. Raises TimeoutError on timeout."""
        acquired = await self.acquire()
        if not acquired:
            raise asyncio.TimeoutError(
                f"Lock {self._lock_name} acquisition timed out after {self._timeout}s"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
        return False


class MetricsLockWrapper(NonBlockingAsyncLockWrapper):
    """Lock wrapper with contention metrics for observability.

    January 12, 2026: Added to provide visibility into lock contention issues.
    Extends NonBlockingAsyncLockWrapper with timing metrics and logging.

    Features:
    - Logs warnings when lock acquisition takes >100ms (configurable)
    - Tracks contention counts per lock
    - Provides metrics for proactive monitoring

    Usage:
        # Drop-in replacement for NonBlockingAsyncLockWrapper
        async with MetricsLockWrapper(self.peers_lock, "peers_lock", timeout=5.0):
            # ... critical section
    """

    # Class-level contention counters
    _contention_counts: dict[str, int] = {}
    _total_wait_time: dict[str, float] = {}

    def __init__(
        self,
        lock: threading.RLock,
        lock_name: str = "unknown",
        timeout: float = 5.0,
        validate_order: bool = True,
        contention_threshold: float = 0.1,  # 100ms threshold
    ):
        """Initialize metrics lock wrapper.

        Args:
            lock: The threading.RLock to wrap
            lock_name: Name of the lock
            timeout: Maximum time to wait for lock acquisition
            validate_order: If True, validate lock ordering
            contention_threshold: Log warning if wait exceeds this (seconds)
        """
        super().__init__(lock, lock_name, timeout, validate_order)
        self._contention_threshold = contention_threshold

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire lock with metrics tracking."""
        start = time.monotonic()
        result = await super().acquire(timeout)
        wait_time = time.monotonic() - start

        # Track metrics
        if self._lock_name not in MetricsLockWrapper._contention_counts:
            MetricsLockWrapper._contention_counts[self._lock_name] = 0
            MetricsLockWrapper._total_wait_time[self._lock_name] = 0.0

        MetricsLockWrapper._total_wait_time[self._lock_name] += wait_time

        # Log if contention exceeds threshold
        if wait_time > self._contention_threshold:
            MetricsLockWrapper._contention_counts[self._lock_name] += 1
            logger.warning(
                f"[LockMetrics] {self._lock_name} contention: {wait_time:.3f}s "
                f"(threshold: {self._contention_threshold}s, "
                f"total_contentions: {MetricsLockWrapper._contention_counts[self._lock_name]})"
            )

        return result

    @classmethod
    def get_contention_stats(cls) -> dict[str, Any]:
        """Get contention statistics for all locks."""
        return {
            "contention_counts": dict(cls._contention_counts),
            "total_wait_times": dict(cls._total_wait_time),
            "locks_with_contention": [
                name for name, count in cls._contention_counts.items() if count > 0
            ],
        }

    @classmethod
    def reset_stats(cls) -> None:
        """Reset contention statistics."""
        cls._contention_counts.clear()
        cls._total_wait_time.clear()


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


# =============================================================================
# Copy-on-Write Peer Snapshot (January 12, 2026)
# =============================================================================


class PeerSnapshot(Generic[T]):
    """Lock-free peer state access using copy-on-write pattern.

    January 12, 2026: Added to eliminate lock contention for read-heavy
    operations like /status endpoint. Before this change, /status could block
    for 6+ seconds under load waiting for peers_lock.

    Design:
    - Maintains internal mutable dictionary for writes (protected by lock)
    - Maintains immutable snapshot for reads (no lock needed)
    - On each write, atomically replaces the snapshot with a new copy
    - Read operations return the snapshot directly without locking

    Thread Safety:
    - Reads are lock-free and always return a consistent snapshot
    - Writes are protected by an RLock
    - The snapshot reference assignment is atomic in Python (GIL)

    Performance:
    - Reads: O(1), no lock acquisition
    - Writes: O(n) due to copy, but writes are infrequent
    - Snapshot refresh on every write ensures readers always see recent data

    Usage:
        # Create snapshot manager
        peers = PeerSnapshot[NodeInfo]()

        # Lock-free read (for /status endpoint)
        snapshot = peers.get_snapshot()
        for node_id, info in snapshot.items():
            ...

        # Write operations (updates trigger snapshot refresh)
        peers.update_peer("node-1", node_info)
        peers.remove_peer("node-2")

        # Bulk update (single lock acquisition, single snapshot refresh)
        with peers.bulk_update():
            peers.update_peer("node-1", info1)
            peers.update_peer("node-2", info2)
    """

    def __init__(self, initial_data: dict[str, T] | None = None):
        """Initialize PeerSnapshot with optional initial data.

        Args:
            initial_data: Optional initial dictionary of peers
        """
        # Internal mutable state (write-side)
        self._peers: dict[str, T] = dict(initial_data) if initial_data else {}

        # Immutable snapshot (read-side) - atomically replaced on writes
        self._snapshot: dict[str, T] = dict(self._peers)

        # Version counter for tracking changes
        self._snapshot_version: int = 0

        # Lock for write operations only
        self._write_lock = threading.RLock()

        # Timestamp of last snapshot update
        self._last_update: float = time.time()

        # Flag for bulk update mode
        self._in_bulk_update: bool = False

    def get_snapshot(self) -> dict[str, T]:
        """Get current peer state as an immutable snapshot.

        This is the primary read method - returns instantly without locking.
        The returned dictionary should be treated as read-only. Modifying it
        will not affect the internal state.

        Returns:
            Dictionary mapping node_id to peer info (immutable copy)
        """
        # No lock needed - snapshot is immutable and reference is atomic
        return self._snapshot

    def get_peer(self, node_id: str) -> T | None:
        """Get a single peer from the snapshot.

        Lock-free read of a specific peer.

        Args:
            node_id: ID of the peer to retrieve

        Returns:
            Peer info if found, None otherwise
        """
        return self._snapshot.get(node_id)

    def __contains__(self, node_id: str) -> bool:
        """Check if peer exists in snapshot (lock-free)."""
        return node_id in self._snapshot

    def __len__(self) -> int:
        """Return number of peers in snapshot (lock-free)."""
        return len(self._snapshot)

    def update_peer(self, node_id: str, info: T) -> None:
        """Update or add a peer.

        This acquires the write lock and refreshes the snapshot.
        If in bulk update mode, defers snapshot refresh until bulk completes.

        Args:
            node_id: Unique peer identifier
            info: Peer information object
        """
        with self._write_lock:
            self._peers[node_id] = info
            if not self._in_bulk_update:
                self._refresh_snapshot()

    def remove_peer(self, node_id: str) -> bool:
        """Remove a peer if it exists.

        Args:
            node_id: ID of peer to remove

        Returns:
            True if peer was removed, False if not found
        """
        with self._write_lock:
            if node_id in self._peers:
                del self._peers[node_id]
                if not self._in_bulk_update:
                    self._refresh_snapshot()
                return True
            return False

    def clear(self) -> None:
        """Remove all peers."""
        with self._write_lock:
            self._peers.clear()
            if not self._in_bulk_update:
                self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        """Create a new immutable snapshot from current state.

        Must be called while holding the write lock.
        Uses shallow copy - peer objects are shared between snapshots.
        For deep isolation, callers should not mutate returned NodeInfo objects.
        """
        self._snapshot = dict(self._peers)
        self._snapshot_version += 1
        self._last_update = time.time()

    class _BulkUpdateContext:
        """Context manager for bulk update operations."""

        def __init__(self, snapshot: "PeerSnapshot"):
            self._snapshot = snapshot

        def __enter__(self):
            self._snapshot._write_lock.acquire()
            self._snapshot._in_bulk_update = True
            return self._snapshot

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                self._snapshot._in_bulk_update = False
                self._snapshot._refresh_snapshot()
            finally:
                self._snapshot._write_lock.release()
            return False

    def bulk_update(self) -> "_BulkUpdateContext":
        """Context manager for efficient bulk updates.

        Acquires lock once and defers snapshot refresh until all updates complete.
        Use this when making multiple updates in sequence.

        Returns:
            Context manager that holds write lock

        Example:
            with peers.bulk_update():
                peers.update_peer("node-1", info1)
                peers.update_peer("node-2", info2)
                peers.remove_peer("node-3")
            # Single snapshot refresh happens here
        """
        return self._BulkUpdateContext(self)

    @property
    def version(self) -> int:
        """Get current snapshot version number.

        Increments on each update. Useful for change detection.
        """
        return self._snapshot_version

    @property
    def last_update_time(self) -> float:
        """Get timestamp of last snapshot update."""
        return self._last_update

    def get_stats(self) -> dict[str, Any]:
        """Get snapshot statistics for monitoring.

        Returns:
            Dictionary with peer count, version, and timing info
        """
        return {
            "peer_count": len(self._snapshot),
            "version": self._snapshot_version,
            "last_update": self._last_update,
            "age_seconds": time.time() - self._last_update,
        }


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

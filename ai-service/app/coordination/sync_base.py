"""Base class for sync managers with common functionality.

This module provides a unified base class for all sync managers in the system,
consolidating common patterns:
- Transport failover (Tailscale → SSH → HTTP)
- Circuit breaker per-node fault tolerance
- Async lock and running state management
- State persistence (JSON or SQLite)

Usage:
    from app.coordination.sync_base import SyncManagerBase

    class MySyncManager(SyncManagerBase):
        async def _do_sync(self, node: str) -> bool:
            # Implement specific sync logic
            pass

See: ai-service/docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.coordination_defaults import SyncDefaults
from app.coordination.sync_constants import (
    SyncDirection,
    SyncPriority,
    SyncResult,
    SyncState,
    SyncTarget,
)

logger = logging.getLogger(__name__)


@dataclass
class BaseSyncProgress:
    """Common sync progress tracking for all sync managers.

    Provides a unified schema for tracking sync progress.
    Note: This is a dataclass for state tracking, not an enum.
    For sync operation states (PENDING, IN_PROGRESS, etc.), use sync_constants.SyncState.
    """
    last_sync_timestamp: float = 0.0
    synced_nodes: set[str] = field(default_factory=set)
    pending_syncs: set[str] = field(default_factory=set)
    failed_nodes: set[str] = field(default_factory=set)
    sync_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "last_sync_timestamp": self.last_sync_timestamp,
            "synced_nodes": list(self.synced_nodes),
            "pending_syncs": list(self.pending_syncs),
            "failed_nodes": list(self.failed_nodes),
            "sync_count": self.sync_count,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseSyncProgress:
        """Deserialize state from dictionary."""
        return cls(
            last_sync_timestamp=data.get("last_sync_timestamp", 0.0),
            synced_nodes=set(data.get("synced_nodes", [])),
            pending_syncs=set(data.get("pending_syncs", [])),
            failed_nodes=set(data.get("failed_nodes", [])),
            sync_count=data.get("sync_count", 0),
            last_error=data.get("last_error"),
        )


# Use canonical circuit breaker from distributed module
from app.distributed.circuit_breaker import CircuitBreaker as CanonicalCircuitBreaker

# CircuitBreakerConfig consolidated to transport_base.py (December 2025)
# Import from canonical location to avoid duplicate definitions
from app.coordination.transport_base import CircuitBreakerConfig

# Alias for backward compatibility - use canonical CircuitBreaker directly
# Old code using SimpleCircuitBreaker should migrate to CanonicalCircuitBreaker
SimpleCircuitBreaker = CanonicalCircuitBreaker


class SyncManagerBase(ABC):
    """Base class for sync managers with common functionality.

    Provides:
    - Async lock for preventing concurrent syncs
    - Running state management
    - Circuit breaker per node
    - State persistence
    - Transport failover orchestration

    Subclasses must implement:
    - _do_sync(node: str) -> bool: Actual sync logic for a node
    - _get_nodes() -> List[str]: Get list of nodes to sync with
    """

    def __init__(
        self,
        state_path: Path | None = None,
        sync_interval: float | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """Initialize base sync manager.

        Args:
            state_path: Path to persist sync state (JSON file)
            sync_interval: Seconds between sync cycles (defaults to SyncDefaults.DATA_SYNC_INTERVAL)
            circuit_breaker_config: Config for per-node circuit breakers
        """
        # Use centralized default from coordination_defaults
        if sync_interval is None:
            sync_interval = SyncDefaults.DATA_SYNC_INTERVAL
        self.state_path = state_path
        self.sync_interval = sync_interval
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        # Core state
        self._sync_lock = asyncio.Lock()
        self._running = False
        self._state = BaseSyncProgress()
        # Use single canonical circuit breaker (tracks targets internally)
        self._circuit_breaker = CanonicalCircuitBreaker(
            failure_threshold=self.circuit_breaker_config.failure_threshold,
            recovery_timeout=self.circuit_breaker_config.recovery_timeout,
            half_open_max_calls=self.circuit_breaker_config.half_open_max_calls,
            operation_type="sync_manager",
        )

        # Load persisted state if available
        if self.state_path and self.state_path.exists():
            self._load_state()

    def _can_sync_with_node(self, node: str) -> bool:
        """Check if sync with node is allowed (circuit not open)."""
        return self._circuit_breaker.can_execute(node)

    def _record_sync_success(self, node: str) -> None:
        """Record successful sync with node."""
        self._circuit_breaker.record_success(node)

    def _record_sync_failure(self, node: str, error: Exception | None = None) -> None:
        """Record failed sync with node."""
        self._circuit_breaker.record_failure(node, error)

    def _load_state(self) -> None:
        """Load state from persistent storage."""
        try:
            if self.state_path and self.state_path.exists():
                with open(self.state_path) as f:
                    data = json.load(f)
                    self._state = BaseSyncProgress.from_dict(data)
                    logger.debug(f"Loaded sync state from {self.state_path}")
        except Exception as e:
            logger.warning(f"Failed to load sync state: {e}")

    def _save_state(self) -> None:
        """Save state to persistent storage."""
        try:
            if self.state_path:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_path, "w") as f:
                    json.dump(self._state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save sync state: {e}")

    @abstractmethod
    async def _do_sync(self, node: str) -> bool:
        """Perform sync with a specific node.

        Args:
            node: Node identifier to sync with

        Returns:
            True if sync succeeded, False otherwise
        """

    @abstractmethod
    def _get_nodes(self) -> list[str]:
        """Get list of nodes to sync with.

        Returns:
            List of node identifiers
        """

    async def sync_with_node(self, node: str) -> bool:
        """Sync with a specific node, respecting circuit breaker.

        Args:
            node: Node identifier

        Returns:
            True if sync succeeded, False if failed or skipped
        """
        if not self._can_sync_with_node(node):
            logger.debug(f"Circuit breaker open for {node}, skipping sync")
            return False

        try:
            success = await self._do_sync(node)
            if success:
                self._record_sync_success(node)
                self._state.synced_nodes.add(node)
                self._state.failed_nodes.discard(node)
            else:
                self._record_sync_failure(node)
                self._state.failed_nodes.add(node)
            return success
        except Exception as e:
            logger.error(f"Sync error with {node}: {e}")
            self._record_sync_failure(node, e)
            self._state.failed_nodes.add(node)
            self._state.last_error = str(e)
            return False

    async def sync_with_cluster(
        self,
        max_concurrent: int = 5,
    ) -> dict[str, bool]:
        """Sync with all nodes in the cluster.

        December 29, 2025: Parallelized sync operations for improved throughput.
        Uses asyncio.gather with semaphore to limit concurrent connections.

        Args:
            max_concurrent: Maximum concurrent sync operations (default 5)

        Returns:
            Dict mapping node -> success status
        """
        async with self._sync_lock:
            nodes = self._get_nodes()
            if not nodes:
                return {}

            # Use semaphore to limit concurrent syncs
            semaphore = asyncio.Semaphore(max_concurrent)

            async def sync_with_limit(node: str) -> tuple[str, bool]:
                async with semaphore:
                    result = await self.sync_with_node(node)
                    return (node, result)

            # Run syncs in parallel with limited concurrency
            tasks = [sync_with_limit(node) for node in nodes]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            results: dict[str, bool] = {}
            for item in completed:
                if isinstance(item, Exception):
                    logger.error(f"Sync task failed: {item}")
                elif isinstance(item, tuple):
                    node, success = item
                    results[node] = success

            self._state.last_sync_timestamp = time.time()
            self._state.sync_count += 1
            self._save_state()

            return results

    async def start(self) -> None:
        """Start the sync manager background task."""
        if self._running:
            logger.warning("Sync manager already running")
            return

        self._running = True
        logger.info(f"Starting sync manager with interval {self.sync_interval}s")

        while self._running:
            try:
                await self.sync_with_cluster()
            except Exception as e:
                logger.error(f"Sync cycle error: {e}")

            await asyncio.sleep(self.sync_interval)

    async def stop(self) -> None:
        """Stop the sync manager."""
        self._running = False
        self._save_state()
        logger.info("Sync manager stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current sync manager status.

        Returns:
            Status dict with state, circuit breaker info, etc.
        """
        # Get all circuit breaker states (targets tracked internally)
        cb_states = self._circuit_breaker.get_all_states()
        return {
            "running": self._running,
            "state": self._state.to_dict(),
            "circuit_breaker": {
                target: status.state.value if hasattr(status.state, 'value') else str(status.state)
                for target, status in cb_states.items()
            } if cb_states else {},
        }

    def health_check(self) -> "HealthCheckResult":
        """Check sync manager health for daemon monitoring.

        December 2025 Phase 4: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with sync status and error metrics.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            # Get current state
            synced_count = len(self._state.synced_nodes)
            failed_count = len(self._state.failed_nodes)
            pending_count = len(self._state.pending_syncs)

            if not self._running:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.STOPPED,
                    message="Sync manager not running",
                )

            # Check for high failure rate
            total_nodes = synced_count + failed_count
            if total_nodes > 0:
                failure_rate = failed_count / total_nodes
                if failure_rate > 0.5:
                    return HealthCheckResult(
                        healthy=False,
                        status=CoordinatorStatus.DEGRADED,
                        message=f"High sync failure rate: {failure_rate:.1%} ({failed_count}/{total_nodes} nodes)",
                        details={
                            "synced_nodes": synced_count,
                            "failed_nodes": failed_count,
                            "pending_syncs": pending_count,
                            "failure_rate": failure_rate,
                        },
                    )

            # Check if last sync was recent (within 2x interval)
            time_since_sync = time.time() - self._state.last_sync_timestamp
            if self._state.last_sync_timestamp > 0 and time_since_sync > self.sync_interval * 2:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Last sync was {time_since_sync:.0f}s ago (interval: {self.sync_interval}s)",
                    details={
                        "time_since_sync": time_since_sync,
                        "sync_interval": self.sync_interval,
                    },
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Sync manager healthy: {synced_count} synced, {failed_count} failed",
                details={
                    "synced_nodes": synced_count,
                    "failed_nodes": failed_count,
                    "pending_syncs": pending_count,
                    "sync_count": self._state.sync_count,
                },
            )

        except Exception as e:
            logger.error(f"Error checking SyncManagerBase health: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
            )


# Transport helper functions for common sync patterns

async def try_transports(
    node: str,
    transports: list[tuple[str, Callable]],
    timeout: float = 30.0,
) -> tuple[bool, str]:
    """Try multiple transport methods with failover.

    Args:
        node: Target node
        transports: List of (name, async_callable) pairs
        timeout: Timeout per transport attempt

    Returns:
        Tuple of (success, transport_used)
    """
    for transport_name, transport_func in transports:
        try:
            result = await asyncio.wait_for(
                transport_func(node),
                timeout=timeout,
            )
            if result:
                logger.debug(f"Sync to {node} succeeded via {transport_name}")
                return True, transport_name
        except asyncio.TimeoutError:
            logger.warning(f"Sync to {node} timed out via {transport_name}")
        except Exception as e:
            logger.warning(f"Sync to {node} failed via {transport_name}: {e}")

    return False, ""


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Base class and config
    "CircuitBreakerConfig",
    "SyncManagerBase",
    # Progress tracking
    "BaseSyncProgress",
    # Re-exports from sync_constants (for convenience imports)
    "SyncDirection",
    "SyncPriority",
    "SyncResult",
    "SyncState",
    "SyncTarget",
    # Utility functions
    "try_transports",
]

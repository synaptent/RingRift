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

See: docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """Common sync state for all sync managers.

    Provides a unified schema for tracking sync progress.
    """
    last_sync_timestamp: float = 0.0
    synced_nodes: Set[str] = field(default_factory=set)
    pending_syncs: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    sync_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "SyncState":
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


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Kept for backward compatibility - maps to canonical CircuitBreaker parameters.
    """
    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 1


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
        state_path: Optional[Path] = None,
        sync_interval: float = 300.0,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize base sync manager.

        Args:
            state_path: Path to persist sync state (JSON file)
            sync_interval: Seconds between sync cycles
            circuit_breaker_config: Config for per-node circuit breakers
        """
        self.state_path = state_path
        self.sync_interval = sync_interval
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        # Core state
        self._sync_lock = asyncio.Lock()
        self._running = False
        self._state = SyncState()
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

    def _record_sync_failure(self, node: str, error: Optional[Exception] = None) -> None:
        """Record failed sync with node."""
        self._circuit_breaker.record_failure(node, error)

    def _load_state(self) -> None:
        """Load state from persistent storage."""
        try:
            if self.state_path and self.state_path.exists():
                with open(self.state_path) as f:
                    data = json.load(f)
                    self._state = SyncState.from_dict(data)
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
        pass

    @abstractmethod
    def _get_nodes(self) -> List[str]:
        """Get list of nodes to sync with.

        Returns:
            List of node identifiers
        """
        pass

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

    async def sync_with_cluster(self) -> Dict[str, bool]:
        """Sync with all nodes in the cluster.

        Returns:
            Dict mapping node -> success status
        """
        async with self._sync_lock:
            results: Dict[str, bool] = {}
            nodes = self._get_nodes()

            for node in nodes:
                results[node] = await self.sync_with_node(node)

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

    def get_status(self) -> Dict[str, Any]:
        """Get current sync manager status.

        Returns:
            Status dict with state, circuit breaker info, etc.
        """
        return {
            "running": self._running,
            "state": self._state.to_dict(),
            "circuit_breakers": {
                node: breaker.get_status()
                for node, breaker in self._circuit_breakers.items()
            },
        }


# Transport helper functions for common sync patterns

async def try_transports(
    node: str,
    transports: List[Tuple[str, Callable]],
    timeout: float = 30.0,
) -> Tuple[bool, str]:
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
    # Data classes
    "SyncState",
    "CircuitBreakerConfig",
    # Classes
    "SyncManagerBase",
    # Functions
    "try_transports",
]

"""Transport Health Tracking for Cluster Network Stability.

Tracks per-node, per-transport (Tailscale vs Direct IP) success/failure rates
to enable smart transport selection and automatic failover.

Features:
- Track success/failure rates per node per transport
- Automatically disable failing transports after consecutive failures
- Re-test disabled transports periodically
- Provide adaptive timeouts based on historical latency

Usage:
    from app.distributed.transport_health import TransportHealthTracker

    tracker = TransportHealthTracker()

    # Record results
    tracker.record_success("lambda-gh200-a", "tailscale", latency_ms=45.0)
    tracker.record_failure("lambda-gh200-a", "direct")

    # Get best transport for a node
    transport = tracker.get_best_transport("lambda-gh200-a")

    # Get adaptive timeout
    timeout = tracker.get_adaptive_timeout("lambda-gh200-a", "tailscale")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class TransportType(str, Enum):
    """Available transport types for SSH connections."""
    TAILSCALE = "tailscale"  # 100.x.x.x Tailscale mesh
    DIRECT = "direct"        # Direct IP (192.222.x.x for Lambda)
    CLOUDFLARE = "cloudflare"  # Cloudflare Zero Trust tunnel


@dataclass
class TransportStats:
    """Statistics for a single transport on a single node."""
    transport: TransportType
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_attempt_time: float = 0.0
    disabled_until: float = 0.0  # Unix timestamp when transport can be re-enabled
    total_latency_ms: float = 0.0
    latency_samples: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        total = self.successes + self.failures
        if total == 0:
            return 1.0  # Assume good until proven otherwise
        return self.successes / total

    @property
    def average_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.latency_samples == 0:
            return 0.0
        return self.total_latency_ms / self.latency_samples

    @property
    def is_disabled(self) -> bool:
        """Check if transport is currently disabled."""
        return time.time() < self.disabled_until


@dataclass
class NodeHealth:
    """Health tracking for a single node across all transports."""
    node_id: str
    transports: dict[TransportType, TransportStats] = field(default_factory=dict)
    preferred_transport: TransportType | None = None
    last_successful_transport: TransportType | None = None

    def get_or_create_transport(self, transport: TransportType) -> TransportStats:
        """Get transport stats, creating if not exists."""
        if transport not in self.transports:
            self.transports[transport] = TransportStats(transport=transport)
        return self.transports[transport]


class TransportHealthTracker:
    """Tracks transport health across all cluster nodes.

    Thread-safe singleton for tracking SSH/HTTP transport reliability.
    Used by SSHConnectionManager to make smart transport decisions.
    """

    _instance: TransportHealthTracker | None = None
    _lock = threading.Lock()

    # Configuration (overridable via environment)
    CONSECUTIVE_FAILURE_THRESHOLD = int(
        os.environ.get("RINGRIFT_TRANSPORT_FAILURE_THRESHOLD", "3")
    )
    DISABLE_DURATION_SECONDS = float(
        os.environ.get("RINGRIFT_TRANSPORT_DISABLE_DURATION", "300")  # 5 minutes
    )
    LATENCY_HISTORY_WEIGHT = 0.7  # Weight for exponential moving average
    MIN_SAMPLES_FOR_PREFERENCE = 3  # Minimum samples before preferring a transport

    def __new__(cls) -> TransportHealthTracker:
        """Singleton pattern for global access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._nodes: dict[str, NodeHealth] = {}
        self._data_lock = threading.RLock()
        self._initialized = True
        logger.info("TransportHealthTracker initialized")

    def record_success(
        self,
        node_id: str,
        transport: TransportType | str,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a successful connection attempt.

        Args:
            node_id: Node identifier (e.g., "lambda-gh200-a")
            transport: Transport type used
            latency_ms: Connection latency in milliseconds
        """
        transport = TransportType(transport) if isinstance(transport, str) else transport

        with self._data_lock:
            node = self._get_or_create_node(node_id)
            stats = node.get_or_create_transport(transport)

            now = time.time()
            stats.successes += 1
            stats.consecutive_failures = 0
            stats.last_success_time = now
            stats.last_attempt_time = now

            # Track latency with exponential moving average
            if latency_ms > 0:
                if stats.latency_samples == 0:
                    stats.total_latency_ms = latency_ms
                else:
                    # EMA update
                    stats.total_latency_ms = (
                        self.LATENCY_HISTORY_WEIGHT * stats.average_latency_ms +
                        (1 - self.LATENCY_HISTORY_WEIGHT) * latency_ms
                    )
                stats.latency_samples += 1

            # Clear disable status on success
            stats.disabled_until = 0.0

            # Update node's successful transport
            node.last_successful_transport = transport

            logger.debug(
                f"Transport success: {node_id}/{transport.value} "
                f"(rate={stats.success_rate:.1%}, latency={stats.average_latency_ms:.0f}ms)"
            )

    def record_failure(
        self,
        node_id: str,
        transport: TransportType | str,
        error: str | None = None,
    ) -> None:
        """Record a failed connection attempt.

        Args:
            node_id: Node identifier
            transport: Transport type that failed
            error: Optional error message
        """
        transport = TransportType(transport) if isinstance(transport, str) else transport

        with self._data_lock:
            node = self._get_or_create_node(node_id)
            stats = node.get_or_create_transport(transport)

            now = time.time()
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.last_failure_time = now
            stats.last_attempt_time = now

            # Auto-disable after consecutive failures
            if stats.consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
                if not stats.is_disabled:
                    stats.disabled_until = now + self.DISABLE_DURATION_SECONDS
                    logger.warning(
                        f"Transport disabled: {node_id}/{transport.value} "
                        f"({stats.consecutive_failures} consecutive failures, "
                        f"re-test in {self.DISABLE_DURATION_SECONDS}s)"
                    )

            logger.debug(
                f"Transport failure: {node_id}/{transport.value} "
                f"(consecutive={stats.consecutive_failures}, error={error})"
            )

    def get_best_transport(
        self,
        node_id: str,
        available_transports: list[TransportType] | None = None,
    ) -> TransportType | None:
        """Get the best available transport for a node.

        Priority:
        1. Last successful transport (if recent and not disabled)
        2. Highest success rate transport (if enough samples)
        3. Lowest latency transport
        4. First available non-disabled transport
        5. Any transport (for re-testing disabled ones)

        Args:
            node_id: Node identifier
            available_transports: List of transports to consider (default: all)

        Returns:
            Best transport type, or None if none available
        """
        if available_transports is None:
            available_transports = list(TransportType)

        with self._data_lock:
            node = self._nodes.get(node_id)
            if node is None:
                # No history - prefer Tailscale for new nodes
                if TransportType.TAILSCALE in available_transports:
                    return TransportType.TAILSCALE
                return available_transports[0] if available_transports else None

            # Filter to non-disabled transports
            enabled = [
                t for t in available_transports
                if t not in node.transports or not node.transports[t].is_disabled
            ]

            if not enabled:
                # All disabled - pick one for re-testing (oldest disable time)
                oldest_disable = None
                oldest_transport = None
                for t in available_transports:
                    if t in node.transports:
                        disable_time = node.transports[t].disabled_until
                        if oldest_disable is None or disable_time < oldest_disable:
                            oldest_disable = disable_time
                            oldest_transport = t
                return oldest_transport or (available_transports[0] if available_transports else None)

            # Check if last successful transport is available and recent
            if node.last_successful_transport in enabled:
                stats = node.transports.get(node.last_successful_transport)
                if stats and time.time() - stats.last_success_time < 300:  # Last 5 minutes
                    return node.last_successful_transport

            # Score transports by success rate and latency
            def transport_score(t: TransportType) -> tuple[float, float]:
                if t not in node.transports:
                    return (1.0, 0.0)  # Untested - assume good
                stats = node.transports[t]
                samples = stats.successes + stats.failures
                if samples < self.MIN_SAMPLES_FOR_PREFERENCE:
                    return (1.0, 0.0)  # Not enough data
                # Primary: success rate, Secondary: inverse latency (lower is better)
                latency_score = 1.0 / max(stats.average_latency_ms, 1.0)
                return (stats.success_rate, latency_score)

            best = max(enabled, key=transport_score)
            return best

    def get_adaptive_timeout(
        self,
        node_id: str,
        transport: TransportType | str,
        base_timeout: float = 30.0,
    ) -> float:
        """Get an adaptive timeout based on historical latency.

        Args:
            node_id: Node identifier
            transport: Transport type
            base_timeout: Base timeout in seconds

        Returns:
            Adjusted timeout in seconds
        """
        transport = TransportType(transport) if isinstance(transport, str) else transport

        with self._data_lock:
            node = self._nodes.get(node_id)
            if node is None or transport not in node.transports:
                return base_timeout

            stats = node.transports[transport]
            if stats.latency_samples < 3:
                return base_timeout

            # Scale timeout based on observed latency
            # Base: 30s for 100ms latency
            # Scale linearly with some padding
            avg_latency_s = stats.average_latency_ms / 1000.0
            # Timeout = base + 2 * observed latency (with min/max bounds)
            adaptive = base_timeout + (avg_latency_s * 2)
            return max(10.0, min(adaptive, 120.0))

    def get_node_health(self, node_id: str) -> NodeHealth | None:
        """Get health data for a specific node."""
        with self._data_lock:
            return self._nodes.get(node_id)

    def get_all_health_summary(self) -> dict[str, dict]:
        """Get health summary for all nodes."""
        with self._data_lock:
            summary = {}
            for node_id, node in self._nodes.items():
                transports = {}
                for transport, stats in node.transports.items():
                    transports[transport.value] = {
                        "success_rate": stats.success_rate,
                        "consecutive_failures": stats.consecutive_failures,
                        "average_latency_ms": stats.average_latency_ms,
                        "disabled": stats.is_disabled,
                        "samples": stats.successes + stats.failures,
                    }
                summary[node_id] = {
                    "transports": transports,
                    "preferred": node.preferred_transport.value if node.preferred_transport else None,
                    "last_success": node.last_successful_transport.value if node.last_successful_transport else None,
                }
            return summary

    def reset_node(self, node_id: str) -> None:
        """Reset all health data for a node."""
        with self._data_lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"Reset health data for node: {node_id}")

    def reset_all(self) -> None:
        """Reset all health data."""
        with self._data_lock:
            self._nodes.clear()
            logger.info("Reset all transport health data")

    def _get_or_create_node(self, node_id: str) -> NodeHealth:
        """Get or create node health entry."""
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeHealth(node_id=node_id)
        return self._nodes[node_id]


# Global convenience functions
def get_tracker() -> TransportHealthTracker:
    """Get the global TransportHealthTracker instance."""
    return TransportHealthTracker()


def record_transport_success(
    node_id: str,
    transport: Literal["tailscale", "direct", "cloudflare"],
    latency_ms: float = 0.0,
) -> None:
    """Record a successful transport connection."""
    get_tracker().record_success(node_id, transport, latency_ms)


def record_transport_failure(
    node_id: str,
    transport: Literal["tailscale", "direct", "cloudflare"],
    error: str | None = None,
) -> None:
    """Record a failed transport connection."""
    get_tracker().record_failure(node_id, transport, error)


def get_best_transport(
    node_id: str,
    available: list[str] | None = None,
) -> str | None:
    """Get the best transport for a node."""
    available_types = [TransportType(t) for t in available] if available else None
    result = get_tracker().get_best_transport(node_id, available_types)
    return result.value if result else None

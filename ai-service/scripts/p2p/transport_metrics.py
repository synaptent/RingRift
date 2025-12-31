"""
Transport Metrics Tracking Module.

Dec 30, 2025: Part of Phase 5 - Enhanced Transport Failover.

Tracks per-transport performance metrics for intelligent transport selection.
Uses exponentially weighted moving averages for latency and success rate.

Usage:
    from scripts.p2p.transport_metrics import (
        TransportMetricsTracker,
        get_transport_metrics,
        record_transport_request,
    )

    # Record a request
    record_transport_request(
        peer_id="node-1",
        transport="tailscale",
        latency_ms=45.0,
        success=True,
    )

    # Get recommended transport for a peer
    tracker = get_transport_metrics()
    best = tracker.get_recommended_transport("node-1")
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Metrics configuration
DEFAULT_EWMA_ALPHA = 0.2  # Weight for new observations (higher = more reactive)
LATENCY_PERCENTILES = [50, 90, 99]
MIN_SAMPLES_FOR_RECOMMENDATION = 5  # Need at least this many samples
STALE_THRESHOLD_SECONDS = 600  # Metrics older than 10 min are stale
SUCCESS_RATE_THRESHOLD = 0.7  # Below this, transport is considered unhealthy
LATENCY_PENALTY_THRESHOLD_MS = 1000  # Above this, prefer other transports


@dataclass
class TransportMetrics:
    """Per-transport, per-peer performance metrics."""

    transport_name: str
    peer_id: str

    # Running statistics
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Latency tracking (EWMA)
    latency_ewma_ms: float = 0.0
    latency_min_ms: float = float("inf")
    latency_max_ms: float = 0.0
    latency_samples: list[float] = field(default_factory=list)

    # Bandwidth tracking
    bytes_sent: int = 0
    bytes_received: int = 0

    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_success: float = 0.0
    last_failure: float = 0.0

    # Computed metrics (updated on each record)
    success_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0

    def record_request(
        self,
        latency_ms: float,
        success: bool,
        bytes_transferred: int = 0,
        alpha: float = DEFAULT_EWMA_ALPHA,
    ) -> None:
        """Record a transport request result.

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether the request succeeded
            bytes_transferred: Number of bytes transferred
            alpha: EWMA weight for new observations
        """
        self.sample_count += 1
        now = time.time()
        self.last_seen = now

        if success:
            self.success_count += 1
            self.last_success = now

            # Only track latency for successful requests
            if latency_ms > 0:
                # Update EWMA
                if self.latency_ewma_ms == 0:
                    self.latency_ewma_ms = latency_ms
                else:
                    self.latency_ewma_ms = (
                        alpha * latency_ms + (1 - alpha) * self.latency_ewma_ms
                    )

                # Track min/max
                self.latency_min_ms = min(self.latency_min_ms, latency_ms)
                self.latency_max_ms = max(self.latency_max_ms, latency_ms)

                # Keep recent samples for percentile calculation
                self.latency_samples.append(latency_ms)
                # Keep last 100 samples
                if len(self.latency_samples) > 100:
                    self.latency_samples = self.latency_samples[-100:]

                # Update percentiles
                self._update_percentiles()
        else:
            self.failure_count += 1
            self.last_failure = now

        # Update bytes
        self.bytes_sent += bytes_transferred

        # Update success rate
        self.success_rate = self.success_count / self.sample_count

    def _update_percentiles(self) -> None:
        """Update latency percentiles from samples."""
        if not self.latency_samples:
            return

        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)

        # P50
        idx = int(n * 0.50)
        self.latency_p50_ms = sorted_samples[min(idx, n - 1)]

        # P90
        idx = int(n * 0.90)
        self.latency_p90_ms = sorted_samples[min(idx, n - 1)]

        # P99
        idx = int(n * 0.99)
        self.latency_p99_ms = sorted_samples[min(idx, n - 1)]

    def is_healthy(self) -> bool:
        """Check if this transport is healthy for this peer."""
        if self.sample_count < MIN_SAMPLES_FOR_RECOMMENDATION:
            return True  # Not enough data, assume healthy

        # Check success rate
        if self.success_rate < SUCCESS_RATE_THRESHOLD:
            return False

        # Check latency
        if self.latency_p90_ms > LATENCY_PENALTY_THRESHOLD_MS:
            return False

        return True

    def is_stale(self) -> bool:
        """Check if metrics are stale (no recent data)."""
        return time.time() - self.last_seen > STALE_THRESHOLD_SECONDS

    def get_score(self) -> float:
        """Get a composite score for transport ranking (higher = better).

        Score combines success rate and latency into a single metric.
        """
        if self.sample_count < MIN_SAMPLES_FOR_RECOMMENDATION:
            # Not enough data, return neutral score
            return 0.5

        # Success rate component (0-1)
        success_score = self.success_rate

        # Latency component (0-1, lower latency = higher score)
        # Normalize: 0ms = 1.0, 1000ms = 0.0
        latency_normalized = min(self.latency_ewma_ms / LATENCY_PENALTY_THRESHOLD_MS, 1.0)
        latency_score = 1.0 - latency_normalized

        # Combine: 60% success rate, 40% latency
        return 0.6 * success_score + 0.4 * latency_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transport_name": self.transport_name,
            "peer_id": self.peer_id,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 3),
            "latency_ewma_ms": round(self.latency_ewma_ms, 1),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p90_ms": round(self.latency_p90_ms, 1),
            "latency_p99_ms": round(self.latency_p99_ms, 1),
            "latency_min_ms": round(self.latency_min_ms, 1) if self.latency_min_ms != float("inf") else None,
            "latency_max_ms": round(self.latency_max_ms, 1),
            "bytes_sent": self.bytes_sent,
            "last_seen": self.last_seen,
            "is_healthy": self.is_healthy(),
            "is_stale": self.is_stale(),
            "score": round(self.get_score(), 3),
        }


@dataclass
class PeerTransportMetrics:
    """Aggregated metrics for all transports to a specific peer."""

    peer_id: str
    transports: dict[str, TransportMetrics] = field(default_factory=dict)

    def get_or_create(self, transport_name: str) -> TransportMetrics:
        """Get or create metrics for a transport."""
        if transport_name not in self.transports:
            self.transports[transport_name] = TransportMetrics(
                transport_name=transport_name,
                peer_id=self.peer_id,
            )
        return self.transports[transport_name]

    def get_best_transport(self) -> str | None:
        """Get the best transport for this peer based on historical performance.

        Returns:
            Transport name or None if no data
        """
        if not self.transports:
            return None

        # Filter to healthy, non-stale transports
        candidates = [
            (name, metrics)
            for name, metrics in self.transports.items()
            if metrics.is_healthy() and not metrics.is_stale()
        ]

        if not candidates:
            # Fall back to any transport with data
            candidates = list(self.transports.items())

        if not candidates:
            return None

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1].get_score(), reverse=True)
        return candidates[0][0]

    def get_ranked_transports(self) -> list[tuple[str, float]]:
        """Get all transports ranked by score.

        Returns:
            List of (transport_name, score) tuples, sorted descending
        """
        ranked = [
            (name, metrics.get_score())
            for name, metrics in self.transports.items()
            if not metrics.is_stale()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "peer_id": self.peer_id,
            "transports": {
                name: metrics.to_dict()
                for name, metrics in self.transports.items()
            },
            "best_transport": self.get_best_transport(),
            "ranked_transports": self.get_ranked_transports(),
        }


class TransportMetricsTracker:
    """
    Global tracker for transport performance across all peers.

    Thread-safe singleton that aggregates metrics from all transport operations.
    """

    _instance: "TransportMetricsTracker | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "TransportMetricsTracker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._peers: dict[str, PeerTransportMetrics] = {}
        self._global_transport_metrics: dict[str, TransportMetrics] = {}
        self._data_lock = threading.RLock()
        self._initialized = True

        logger.debug("TransportMetricsTracker initialized")

    def record_request(
        self,
        peer_id: str,
        transport: str,
        latency_ms: float,
        success: bool,
        bytes_transferred: int = 0,
    ) -> None:
        """Record a transport request result.

        Args:
            peer_id: Target peer node ID
            transport: Transport name (e.g., "tailscale", "http_direct")
            latency_ms: Request latency in milliseconds
            success: Whether the request succeeded
            bytes_transferred: Number of bytes transferred
        """
        with self._data_lock:
            # Per-peer metrics
            if peer_id not in self._peers:
                self._peers[peer_id] = PeerTransportMetrics(peer_id=peer_id)

            peer_metrics = self._peers[peer_id]
            transport_metrics = peer_metrics.get_or_create(transport)
            transport_metrics.record_request(
                latency_ms=latency_ms,
                success=success,
                bytes_transferred=bytes_transferred,
            )

            # Global transport metrics
            if transport not in self._global_transport_metrics:
                self._global_transport_metrics[transport] = TransportMetrics(
                    transport_name=transport,
                    peer_id="*global*",
                )

            self._global_transport_metrics[transport].record_request(
                latency_ms=latency_ms,
                success=success,
                bytes_transferred=bytes_transferred,
            )

    def get_recommended_transport(self, peer_id: str) -> str | None:
        """Get the recommended transport for a peer.

        Args:
            peer_id: Target peer node ID

        Returns:
            Best transport name or None
        """
        with self._data_lock:
            if peer_id not in self._peers:
                return None
            return self._peers[peer_id].get_best_transport()

    def get_ranked_transports(self, peer_id: str) -> list[tuple[str, float]]:
        """Get ranked transports for a peer.

        Args:
            peer_id: Target peer node ID

        Returns:
            List of (transport_name, score) tuples
        """
        with self._data_lock:
            if peer_id not in self._peers:
                return []
            return self._peers[peer_id].get_ranked_transports()

    def get_peer_metrics(self, peer_id: str) -> PeerTransportMetrics | None:
        """Get full metrics for a peer."""
        with self._data_lock:
            return self._peers.get(peer_id)

    def get_global_transport_metrics(self, transport: str) -> TransportMetrics | None:
        """Get global metrics for a transport (across all peers)."""
        with self._data_lock:
            return self._global_transport_metrics.get(transport)

    def get_all_transport_health(self) -> dict[str, dict[str, Any]]:
        """Get health status of all transports globally."""
        with self._data_lock:
            return {
                name: {
                    "success_rate": round(m.success_rate, 3),
                    "latency_p50_ms": round(m.latency_p50_ms, 1),
                    "latency_p90_ms": round(m.latency_p90_ms, 1),
                    "sample_count": m.sample_count,
                    "is_healthy": m.is_healthy(),
                    "score": round(m.get_score(), 3),
                }
                for name, m in self._global_transport_metrics.items()
            }

    def get_peer_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all peer transport metrics."""
        with self._data_lock:
            return {
                peer_id: {
                    "best_transport": pm.get_best_transport(),
                    "transports_count": len(pm.transports),
                    "ranked": pm.get_ranked_transports()[:3],  # Top 3
                }
                for peer_id, pm in self._peers.items()
            }

    def cleanup_stale(self) -> int:
        """Remove stale metrics entries.

        Returns:
            Number of entries removed
        """
        removed = 0
        now = time.time()
        stale_threshold = now - STALE_THRESHOLD_SECONDS * 10  # Very stale

        with self._data_lock:
            # Clean up per-peer metrics
            peers_to_remove = []
            for peer_id, pm in self._peers.items():
                transports_to_remove = []
                for name, tm in pm.transports.items():
                    if tm.last_seen < stale_threshold:
                        transports_to_remove.append(name)

                for name in transports_to_remove:
                    del pm.transports[name]
                    removed += 1

                if not pm.transports:
                    peers_to_remove.append(peer_id)

            for peer_id in peers_to_remove:
                del self._peers[peer_id]

        if removed > 0:
            logger.debug(f"Cleaned up {removed} stale transport metrics entries")

        return removed

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._data_lock:
            self._peers.clear()
            self._global_transport_metrics.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.reset()
                cls._instance = None


# Module-level functions for convenience
_tracker: TransportMetricsTracker | None = None


def get_transport_metrics() -> TransportMetricsTracker:
    """Get the global transport metrics tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = TransportMetricsTracker()
    return _tracker


def record_transport_request(
    peer_id: str,
    transport: str,
    latency_ms: float,
    success: bool,
    bytes_transferred: int = 0,
) -> None:
    """Record a transport request result.

    Convenience function that calls the singleton tracker.
    """
    get_transport_metrics().record_request(
        peer_id=peer_id,
        transport=transport,
        latency_ms=latency_ms,
        success=success,
        bytes_transferred=bytes_transferred,
    )


def get_recommended_transport(peer_id: str) -> str | None:
    """Get the recommended transport for a peer.

    Convenience function that calls the singleton tracker.
    """
    return get_transport_metrics().get_recommended_transport(peer_id)

"""Gossip Metrics Mixin - Metrics tracking for gossip protocol.

Extracted from gossip_protocol.py for modularity.

This mixin provides:
- _record_gossip_metrics: Record gossip events and latency
- _reset_gossip_metrics_hourly: Periodic metric reset
- _record_gossip_compression: Track compression efficiency
- _get_gossip_metrics_summary: Summary for /status endpoint
- _get_gossip_health_status: Health assessment with warnings

Note: Named gossip_metrics_mixin.py to avoid conflict with the deprecated
scripts/p2p/gossip_metrics.py module.

December 2025 (Phase 4): Merged from standalone GossipMetricsMixin.
February 2026: Re-extracted as part of gossip_protocol.py decomposition.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class GossipMetricsMixin:
    """Mixin providing gossip protocol metrics tracking.

    Expects the implementing class to provide:
    - _ensure_state_attr() from P2PMixinBase
    - _log_info/debug/warning() from P2PMixinBase
    - _gossip_metrics: dict (initialized by _init_gossip_protocol)
    - _gossip_compression_stats: dict (initialized by _init_gossip_protocol)
    - _gossip_health_tracker: GossipHealthTracker
    """

    def _record_gossip_metrics(
        self,
        event: str,
        peer_id: str | None = None,
        latency_ms: float = 0,
    ) -> None:
        """Record gossip protocol metrics for monitoring.

        GOSSIP METRICS: Track propagation efficiency and protocol health.
        - message_sent: Gossip messages sent
        - message_received: Gossip messages received
        - state_updates: Number of state updates from gossip
        - propagation_delay_ms: Average latency for gossip messages
        - anti_entropy_repairs: Full state reconciliations triggered

        Args:
            event: Event type (sent, received, update, anti_entropy, stale, latency)
            peer_id: Optional peer ID for context
            latency_ms: Latency in milliseconds (for latency events)
        """
        # Ensure metrics state exists
        # Jan 2026: Use deque(maxlen=100) for propagation_delay_ms to prevent memory leak
        self._ensure_state_attr("_gossip_metrics", {
            "message_sent": 0,
            "message_received": 0,
            "state_updates": 0,
            "propagation_delay_ms": deque(maxlen=100),
            "anti_entropy_repairs": 0,
            "stale_states_detected": 0,
            "last_reset": time.time(),
        })
        metrics = self._gossip_metrics

        # Use .get() with defaults to prevent KeyError in case of race conditions
        # with _reset_gossip_metrics_hourly() (Dec 2025)
        if event == "sent":
            metrics["message_sent"] = metrics.get("message_sent", 0) + 1
        elif event == "received":
            metrics["message_received"] = metrics.get("message_received", 0) + 1
        elif event == "update":
            metrics["state_updates"] = metrics.get("state_updates", 0) + 1
        elif event == "anti_entropy":
            metrics["anti_entropy_repairs"] = metrics.get("anti_entropy_repairs", 0) + 1
        elif event == "stale":
            metrics["stale_states_detected"] = metrics.get("stale_states_detected", 0) + 1
        elif event == "latency":
            # Keep last 100 latency measurements
            # Jan 2026: Use deque(maxlen=100) for automatic bounded size
            delays = metrics.get("propagation_delay_ms")
            if not isinstance(delays, deque):
                delays = deque(maxlen=100)
                metrics["propagation_delay_ms"] = delays
            delays.append(latency_ms)  # deque auto-removes oldest when full

        # Reset metrics every hour
        if time.time() - metrics.get("last_reset", 0) > 3600:
            self._reset_gossip_metrics_hourly()

    def _reset_gossip_metrics_hourly(self) -> dict[str, Any]:
        """Reset gossip metrics and return old values.

        Called automatically after 1 hour. Returns old metrics for logging.
        """
        self._ensure_state_attr("_gossip_metrics", {})
        old_metrics = self._gossip_metrics.copy()

        # Jan 2026: Use deque(maxlen=100) for propagation_delay_ms to prevent memory leak
        self._gossip_metrics = {
            "message_sent": 0,
            "message_received": 0,
            "state_updates": 0,
            "propagation_delay_ms": deque(maxlen=100),
            "anti_entropy_repairs": 0,
            "stale_states_detected": 0,
            "last_reset": time.time(),
        }

        # Log metrics before reset using base class helper
        delays = old_metrics.get("propagation_delay_ms", [])
        avg_latency = sum(delays) / max(1, len(delays)) if delays else 0

        self._log_debug(
            f"Hourly: sent={old_metrics.get('message_sent', 0)} "
            f"recv={old_metrics.get('message_received', 0)} "
            f"updates={old_metrics.get('state_updates', 0)} "
            f"repairs={old_metrics.get('anti_entropy_repairs', 0)} "
            f"stale={old_metrics.get('stale_states_detected', 0)} "
            f"avg_latency={avg_latency:.1f}ms"
        )

        return old_metrics

    def _record_gossip_compression(
        self,
        original_size: int,
        compressed_size: int,
    ) -> None:
        """Record gossip compression metrics.

        COMPRESSION METRICS: Track how effective compression is for gossip messages.
        Typical JSON gossip payloads compress 60-80% with gzip level 6.

        Args:
            original_size: Original message size in bytes
            compressed_size: Compressed message size in bytes
        """
        self._ensure_state_attr("_gossip_compression_stats", {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "messages_compressed": 0,
        })
        stats = self._gossip_compression_stats
        stats["total_original_bytes"] += original_size
        stats["total_compressed_bytes"] += compressed_size
        stats["messages_compressed"] += 1

    def _get_gossip_metrics_summary(self) -> dict[str, Any]:
        """Get summary of gossip metrics for /status endpoint.

        Returns:
            Dict with message counts, latency, and compression stats
        """
        self._ensure_state_attr("_gossip_metrics", {})
        self._ensure_state_attr("_gossip_compression_stats", {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "messages_compressed": 0,
        })
        metrics = self._gossip_metrics
        delays = metrics.get("propagation_delay_ms", [])

        # Include compression stats
        compression = self._gossip_compression_stats
        original = compression.get("total_original_bytes", 0)
        compressed = compression.get("total_compressed_bytes", 0)
        compression_ratio = 1.0 - (compressed / original) if original > 0 else 0

        return {
            "message_sent": metrics.get("message_sent", 0),
            "message_received": metrics.get("message_received", 0),
            "state_updates": metrics.get("state_updates", 0),
            "anti_entropy_repairs": metrics.get("anti_entropy_repairs", 0),
            "stale_states_detected": metrics.get("stale_states_detected", 0),
            "avg_latency_ms": sum(delays) / max(1, len(delays)) if delays else 0,
            "compression_ratio": round(compression_ratio, 3),
            "bytes_saved_kb": round((original - compressed) / 1024, 2),
            "messages_compressed": compression.get("messages_compressed", 0),
        }

    def _get_gossip_health_status(self) -> dict[str, Any]:
        """Get gossip protocol health status.

        Returns health indicators for monitoring:
        - is_healthy: True if gossip is functioning well
        - warnings: List of any warning conditions

        Dec 28, 2025 (Phase 6): Now includes peer health tracking stats.
        """
        summary = self._get_gossip_metrics_summary()
        warnings = []

        # Check for high latency
        avg_latency = summary.get("avg_latency_ms", 0)
        if avg_latency > 1000:
            warnings.append(f"High gossip latency: {avg_latency:.0f}ms")

        # Check for low message rate (stale cluster)
        sent = summary.get("message_sent", 0)
        received = summary.get("message_received", 0)
        if sent + received < 10:
            warnings.append("Low gossip activity")

        # Check for high stale rate
        stale = summary.get("stale_states_detected", 0)
        updates = summary.get("state_updates", 0)
        if updates > 0 and stale / updates > 0.5:
            warnings.append(f"High stale rate: {stale}/{updates}")

        # Dec 28, 2025 (Phase 6): Check for suspected peers via gossip failures
        health_tracker_stats: dict[str, Any] = {}
        if hasattr(self, "_gossip_health_tracker"):
            tracker = self._gossip_health_tracker
            health_tracker_stats = tracker.get_stats()
            suspected_count = health_tracker_stats.get("suspected_peers", 0)
            if suspected_count > 0:
                suspected_ids = health_tracker_stats.get("suspected_peer_ids", [])
                if suspected_count <= 3:
                    warnings.append(f"Gossip failures: {', '.join(suspected_ids)}")
                else:
                    warnings.append(f"Gossip failures: {suspected_count} peers unresponsive")

        return {
            "is_healthy": len(warnings) == 0,
            "warnings": warnings,
            "metrics": summary,
            "peer_health": health_tracker_stats,  # Phase 6
        }

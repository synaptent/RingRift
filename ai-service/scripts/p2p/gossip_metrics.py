"""Gossip Protocol Metrics Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides gossip protocol metrics tracking and compression stats.

Usage:
    class P2POrchestrator(GossipMetricsMixin, ...):
        pass

Phase 2 extraction - Dec 26, 2025
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class GossipMetricsMixin:
    """Mixin providing gossip protocol metrics tracking.

    Requires the implementing class to have:
    - node_id: str - This node's ID
    - notifier: Optional notifier for alerts (can be None)
    """

    # Type hints for IDE support
    node_id: str

    def _init_gossip_metrics(self) -> None:
        """Initialize gossip metrics storage.

        Call this in __init__ or on first use.
        """
        if not hasattr(self, "_gossip_metrics"):
            self._gossip_metrics = {
                "message_sent": 0,
                "message_received": 0,
                "state_updates": 0,
                "propagation_delay_ms": [],
                "anti_entropy_repairs": 0,
                "stale_states_detected": 0,
                "last_reset": time.time(),
            }
        if not hasattr(self, "_gossip_compression_stats"):
            self._gossip_compression_stats = {
                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
                "messages_compressed": 0,
            }

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
        self._init_gossip_metrics()
        metrics = self._gossip_metrics

        if event == "sent":
            metrics["message_sent"] += 1
        elif event == "received":
            metrics["message_received"] += 1
        elif event == "update":
            metrics["state_updates"] += 1
        elif event == "anti_entropy":
            metrics["anti_entropy_repairs"] += 1
        elif event == "stale":
            metrics["stale_states_detected"] += 1
        elif event == "latency":
            # Keep last 100 latency measurements
            metrics["propagation_delay_ms"].append(latency_ms)
            if len(metrics["propagation_delay_ms"]) > 100:
                metrics["propagation_delay_ms"] = metrics["propagation_delay_ms"][-100:]

        # Reset metrics every hour
        if time.time() - metrics["last_reset"] > 3600:
            self._reset_gossip_metrics_hourly()

    def _reset_gossip_metrics_hourly(self) -> dict[str, Any]:
        """Reset gossip metrics and return old values.

        Called automatically after 1 hour. Returns old metrics for logging.
        """
        self._init_gossip_metrics()
        old_metrics = self._gossip_metrics.copy()

        self._gossip_metrics = {
            "message_sent": 0,
            "message_received": 0,
            "state_updates": 0,
            "propagation_delay_ms": [],
            "anti_entropy_repairs": 0,
            "stale_states_detected": 0,
            "last_reset": time.time(),
        }

        # Log metrics before reset
        delays = old_metrics.get("propagation_delay_ms", [])
        avg_latency = sum(delays) / max(1, len(delays)) if delays else 0

        logger.debug(
            f"[GOSSIP] Hourly: sent={old_metrics['message_sent']} "
            f"recv={old_metrics['message_received']} "
            f"updates={old_metrics['state_updates']} "
            f"repairs={old_metrics['anti_entropy_repairs']} "
            f"stale={old_metrics['stale_states_detected']} "
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
        self._init_gossip_metrics()
        stats = self._gossip_compression_stats
        stats["total_original_bytes"] += original_size
        stats["total_compressed_bytes"] += compressed_size
        stats["messages_compressed"] += 1

    def _get_gossip_metrics_summary(self) -> dict[str, Any]:
        """Get summary of gossip metrics for /status endpoint.

        Returns:
            Dict with message counts, latency, and compression stats
        """
        self._init_gossip_metrics()
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

        return {
            "is_healthy": len(warnings) == 0,
            "warnings": warnings,
            "metrics": summary,
        }


# Standalone utility functions
def calculate_compression_ratio(original: int, compressed: int) -> float:
    """Calculate compression ratio.

    Args:
        original: Original size in bytes
        compressed: Compressed size in bytes

    Returns:
        Ratio of bytes saved (0.0 to 1.0)
    """
    if original <= 0:
        return 0.0
    return 1.0 - (compressed / original)

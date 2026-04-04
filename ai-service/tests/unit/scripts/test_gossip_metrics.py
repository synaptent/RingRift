"""Tests for gossip_metrics.py mixin.

Tests the GossipMetricsMixin extracted from p2p_orchestrator.py.
"""

from __future__ import annotations

import time
import warnings
from typing import Any
from unittest.mock import patch

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from scripts.p2p.gossip_metrics import (
        GossipMetricsMixin,
        calculate_compression_ratio,
    )


class MockOrchestrator(GossipMetricsMixin):
    """Mock orchestrator class that uses GossipMetricsMixin."""

    def __init__(self):
        self.node_id = "test-node"


class TestInitGossipMetrics:
    """Test _init_gossip_metrics method."""

    def test_initializes_metrics_on_first_call(self):
        """First call should initialize metrics."""
        orch = MockOrchestrator()
        assert not hasattr(orch, "_gossip_metrics")
        assert not hasattr(orch, "_gossip_compression_stats")

        orch._init_gossip_metrics()

        assert hasattr(orch, "_gossip_metrics")
        assert hasattr(orch, "_gossip_compression_stats")
        assert orch._gossip_metrics["message_sent"] == 0
        assert orch._gossip_metrics["message_received"] == 0
        assert orch._gossip_compression_stats["messages_compressed"] == 0

    def test_idempotent(self):
        """Multiple calls should not reset metrics."""
        orch = MockOrchestrator()
        orch._init_gossip_metrics()
        orch._gossip_metrics["message_sent"] = 5

        orch._init_gossip_metrics()

        assert orch._gossip_metrics["message_sent"] == 5


class TestRecordGossipMetrics:
    """Test _record_gossip_metrics method."""

    def test_record_sent(self):
        """Record sent event increments message_sent."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("sent")
        orch._record_gossip_metrics("sent")

        assert orch._gossip_metrics["message_sent"] == 2

    def test_record_received(self):
        """Record received event increments message_received."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("received")

        assert orch._gossip_metrics["message_received"] == 1

    def test_record_update(self):
        """Record update event increments state_updates."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("update")
        orch._record_gossip_metrics("update")
        orch._record_gossip_metrics("update")

        assert orch._gossip_metrics["state_updates"] == 3

    def test_record_anti_entropy(self):
        """Record anti_entropy event increments anti_entropy_repairs."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("anti_entropy")

        assert orch._gossip_metrics["anti_entropy_repairs"] == 1

    def test_record_stale(self):
        """Record stale event increments stale_states_detected."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("stale")

        assert orch._gossip_metrics["stale_states_detected"] == 1

    def test_record_latency(self):
        """Record latency event appends to propagation_delay_ms."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("latency", latency_ms=50.0)
        orch._record_gossip_metrics("latency", latency_ms=100.0)

        delays = orch._gossip_metrics["propagation_delay_ms"]
        assert list(delays) == [50.0, 100.0]

    def test_latency_capped_at_100(self):
        """Latency measurements capped at 100 most recent."""
        orch = MockOrchestrator()
        for i in range(150):
            orch._record_gossip_metrics("latency", latency_ms=float(i))

        delays = orch._gossip_metrics["propagation_delay_ms"]
        assert len(delays) == 100
        # Should have most recent 100 (50-149)
        assert list(delays)[0] == 50.0
        assert list(delays)[-1] == 149.0

    def test_hourly_reset_triggers(self):
        """Metrics reset after 1 hour."""
        orch = MockOrchestrator()
        orch._init_gossip_metrics()
        orch._gossip_metrics["message_sent"] = 100
        # Set last_reset to 2 hours ago
        orch._gossip_metrics["last_reset"] = time.time() - 7200

        orch._record_gossip_metrics("sent")

        # Event increments first (101), then reset triggers (0)
        # This is correct: the event was counted, then metrics reset
        assert orch._gossip_metrics["message_sent"] == 0
        # Verify last_reset was updated to recent time
        assert time.time() - orch._gossip_metrics["last_reset"] < 5


class TestResetGossipMetricsHourly:
    """Test _reset_gossip_metrics_hourly method."""

    def test_returns_old_metrics(self):
        """Reset returns copy of old metrics."""
        orch = MockOrchestrator()
        orch._init_gossip_metrics()
        orch._gossip_metrics["message_sent"] = 50
        orch._gossip_metrics["message_received"] = 25
        orch._gossip_metrics["propagation_delay_ms"] = [100.0, 200.0]

        old = orch._reset_gossip_metrics_hourly()

        assert old["message_sent"] == 50
        assert old["message_received"] == 25

    def test_resets_metrics(self):
        """Reset clears current metrics."""
        orch = MockOrchestrator()
        orch._init_gossip_metrics()
        orch._gossip_metrics["message_sent"] = 50

        orch._reset_gossip_metrics_hourly()

        assert orch._gossip_metrics["message_sent"] == 0
        assert list(orch._gossip_metrics["propagation_delay_ms"]) == []


class TestRecordGossipCompression:
    """Test _record_gossip_compression method."""

    def test_records_compression_stats(self):
        """Records original and compressed sizes."""
        orch = MockOrchestrator()
        orch._record_gossip_compression(1000, 200)
        orch._record_gossip_compression(500, 100)

        stats = orch._gossip_compression_stats
        assert stats["total_original_bytes"] == 1500
        assert stats["total_compressed_bytes"] == 300
        assert stats["messages_compressed"] == 2


class TestGetGossipMetricsSummary:
    """Test _get_gossip_metrics_summary method."""

    def test_summary_with_no_data(self):
        """Summary works with no recorded data."""
        orch = MockOrchestrator()
        summary = orch._get_gossip_metrics_summary()

        assert summary["message_sent"] == 0
        assert summary["message_received"] == 0
        assert summary["avg_latency_ms"] == 0
        assert summary["compression_ratio"] == 0

    def test_summary_with_data(self):
        """Summary correctly aggregates data."""
        orch = MockOrchestrator()
        orch._record_gossip_metrics("sent")
        orch._record_gossip_metrics("sent")
        orch._record_gossip_metrics("received")
        orch._record_gossip_metrics("latency", latency_ms=100.0)
        orch._record_gossip_metrics("latency", latency_ms=200.0)
        orch._record_gossip_compression(1000, 200)

        summary = orch._get_gossip_metrics_summary()

        assert summary["message_sent"] == 2
        assert summary["message_received"] == 1
        assert summary["avg_latency_ms"] == 150.0
        assert summary["compression_ratio"] == 0.8  # 1 - 200/1000
        assert summary["bytes_saved_kb"] == 0.78  # (1000-200)/1024 rounded


class TestGetGossipHealthStatus:
    """Test _get_gossip_health_status method."""

    def test_healthy_with_normal_metrics(self):
        """Healthy status with normal metrics."""
        orch = MockOrchestrator()
        # Record enough activity
        for _ in range(20):
            orch._record_gossip_metrics("sent")
            orch._record_gossip_metrics("received")
            orch._record_gossip_metrics("latency", latency_ms=50.0)

        status = orch._get_gossip_health_status()

        assert status["is_healthy"] is True
        assert status["warnings"] == []

    def test_warning_on_high_latency(self):
        """Warning when latency exceeds 1000ms."""
        orch = MockOrchestrator()
        for _ in range(20):
            orch._record_gossip_metrics("sent")
            orch._record_gossip_metrics("latency", latency_ms=1500.0)

        status = orch._get_gossip_health_status()

        assert status["is_healthy"] is False
        assert any("latency" in w.lower() for w in status["warnings"])

    def test_warning_on_low_activity(self):
        """Warning when gossip activity is low."""
        orch = MockOrchestrator()
        # Only 5 messages total (below 10 threshold)
        for _ in range(5):
            orch._record_gossip_metrics("sent")

        status = orch._get_gossip_health_status()

        assert status["is_healthy"] is False
        assert any("low" in w.lower() for w in status["warnings"])

    def test_warning_on_high_stale_rate(self):
        """Warning when stale rate exceeds 50%."""
        orch = MockOrchestrator()
        for _ in range(20):
            orch._record_gossip_metrics("sent")
        # 10 updates, 8 stale (80% stale rate)
        for _ in range(10):
            orch._record_gossip_metrics("update")
        for _ in range(8):
            orch._record_gossip_metrics("stale")

        status = orch._get_gossip_health_status()

        assert status["is_healthy"] is False
        assert any("stale" in w.lower() for w in status["warnings"])


class TestCalculateCompressionRatio:
    """Test standalone calculate_compression_ratio function."""

    def test_zero_original(self):
        """Returns 0 when original is 0."""
        assert calculate_compression_ratio(0, 0) == 0.0

    def test_negative_original(self):
        """Returns 0 when original is negative."""
        assert calculate_compression_ratio(-100, 50) == 0.0

    def test_normal_compression(self):
        """Calculates correct ratio for normal case."""
        # 1000 -> 200 = 80% compression
        assert calculate_compression_ratio(1000, 200) == 0.8

    def test_no_compression(self):
        """Returns 0 when no compression achieved."""
        assert calculate_compression_ratio(100, 100) == 0.0

    def test_expansion(self):
        """Handles expansion case (negative ratio)."""
        # 100 -> 150 = -50% (expansion)
        assert calculate_compression_ratio(100, 150) == -0.5

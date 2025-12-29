"""Unit tests for GossipHealthTracker.

December 28, 2025: Phase 6 - Gossip health tracking tests.
Tests the per-peer gossip failure tracking and event emission logic.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


class TestGossipHealthTrackerImport:
    """Test that GossipHealthTracker can be imported."""

    def test_import_gossip_health_tracker(self):
        """Test basic import."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        assert GossipHealthTracker is not None

    def test_import_threshold_constant(self):
        """Test threshold constant is available."""
        from scripts.p2p.gossip_protocol import GOSSIP_FAILURE_SUSPECT_THRESHOLD

        assert GOSSIP_FAILURE_SUSPECT_THRESHOLD == 5


class TestGossipHealthTrackerInit:
    """Test GossipHealthTracker initialization."""

    def test_default_threshold(self):
        """Test default failure threshold."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        assert tracker._failure_threshold == 5

    def test_custom_threshold(self):
        """Test custom failure threshold."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=10)
        assert tracker._failure_threshold == 10

    def test_initial_state_empty(self):
        """Test initial state is empty."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        assert len(tracker._failure_counts) == 0
        assert len(tracker._last_success) == 0
        assert len(tracker._suspect_emitted) == 0


class TestGossipHealthTrackerFailureTracking:
    """Test gossip failure tracking."""

    def test_first_failure_returns_count_1(self):
        """Test first failure returns count of 1."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        should_emit, count = tracker.record_gossip_failure("peer1")
        assert count == 1
        assert not should_emit

    def test_failures_accumulate(self):
        """Test failure count accumulates."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        for i in range(1, 5):
            should_emit, count = tracker.record_gossip_failure("peer1")
            assert count == i
            assert not should_emit  # Below threshold

    def test_threshold_triggers_emit(self):
        """Test reaching threshold triggers should_emit."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=3)
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")
        should_emit, count = tracker.record_gossip_failure("peer1")
        assert count == 3
        assert should_emit

    def test_emit_only_once(self):
        """Test should_emit is only True once at threshold."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=3)
        for _ in range(2):
            tracker.record_gossip_failure("peer1")

        # At threshold - should emit
        should_emit, _ = tracker.record_gossip_failure("peer1")
        assert should_emit

        # Past threshold - should not emit again
        should_emit, count = tracker.record_gossip_failure("peer1")
        assert count == 4
        assert not should_emit

    def test_independent_peer_tracking(self):
        """Test peers are tracked independently."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        _, count1 = tracker.record_gossip_failure("peer1")
        _, count2 = tracker.record_gossip_failure("peer2")
        _, count1b = tracker.record_gossip_failure("peer1")

        assert count1 == 1
        assert count2 == 1
        assert count1b == 2


class TestGossipHealthTrackerSuccess:
    """Test gossip success tracking."""

    def test_success_resets_failures(self):
        """Test success resets failure count."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_success("peer1")

        assert tracker.get_failure_count("peer1") == 0

    def test_success_records_last_time(self):
        """Test success records timestamp."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        before = time.time()
        tracker.record_gossip_success("peer1")
        after = time.time()

        last_success = tracker.get_last_success("peer1")
        assert before <= last_success <= after

    def test_success_returns_was_suspected(self):
        """Test success returns whether peer was suspected."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=2)

        # Not suspected yet
        was_suspected = tracker.record_gossip_success("peer1")
        assert not was_suspected

        # Make it suspected
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")

        # Now recover
        was_suspected = tracker.record_gossip_success("peer1")
        assert was_suspected

        # Not suspected after recovery
        was_suspected = tracker.record_gossip_success("peer1")
        assert not was_suspected

    def test_success_clears_suspect_status(self):
        """Test success clears suspect status."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=2)
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")

        assert tracker.is_suspected("peer1")

        tracker.record_gossip_success("peer1")

        assert not tracker.is_suspected("peer1")


class TestGossipHealthTrackerQueries:
    """Test query methods."""

    def test_get_failure_count_unknown_peer(self):
        """Test get_failure_count for unknown peer."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        assert tracker.get_failure_count("unknown") == 0

    def test_get_last_success_unknown_peer(self):
        """Test get_last_success for unknown peer returns None."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        assert tracker.get_last_success("unknown") is None

    def test_is_suspected_unknown_peer(self):
        """Test is_suspected for unknown peer."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        assert not tracker.is_suspected("unknown")

    def test_get_suspected_peers_empty(self):
        """Test get_suspected_peers when empty."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        assert tracker.get_suspected_peers() == set()

    def test_get_suspected_peers_with_suspects(self):
        """Test get_suspected_peers with suspects."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=2)
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer2")
        tracker.record_gossip_failure("peer2")

        suspects = tracker.get_suspected_peers()
        assert suspects == {"peer1", "peer2"}


class TestGossipHealthTrackerStats:
    """Test statistics methods."""

    def test_get_stats_empty(self):
        """Test get_stats when empty."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        stats = tracker.get_stats()

        assert stats["total_tracked_peers"] == 0
        assert stats["suspected_peers"] == 0
        assert stats["suspected_peer_ids"] == []
        assert stats["stale_peers"] == 0

    def test_get_stats_with_data(self):
        """Test get_stats with data."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=3)
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")
        tracker.record_gossip_failure("peer1")  # Now suspected
        tracker.record_gossip_failure("peer2")
        tracker.record_gossip_success("peer3")

        stats = tracker.get_stats()

        # Total tracked = failure_counts keys (peer1, peer2, peer3)
        # record_gossip_success also adds peer to _failure_counts with count=0
        assert stats["total_tracked_peers"] == 3
        assert stats["suspected_peers"] == 1
        assert "peer1" in stats["suspected_peer_ids"]


class TestGossipHealthTrackerCleanup:
    """Test cleanup methods."""

    def test_cleanup_stale_peers(self):
        """Test cleanup_stale_peers removes old entries."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()

        # Add some peers with old timestamps
        tracker._last_success["old_peer"] = time.time() - 7200  # 2 hours ago
        tracker._failure_counts["old_peer"] = 1
        tracker._last_success["new_peer"] = time.time()
        tracker._failure_counts["new_peer"] = 1

        cleaned = tracker.cleanup_stale_peers(max_age_seconds=3600)  # 1 hour

        assert cleaned == 1
        assert "old_peer" not in tracker._failure_counts
        assert "new_peer" in tracker._failure_counts

    def test_cleanup_stale_peers_no_cleanup_needed(self):
        """Test cleanup_stale_peers when nothing is stale."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        tracker.record_gossip_success("peer1")
        tracker.record_gossip_failure("peer1")

        cleaned = tracker.cleanup_stale_peers(max_age_seconds=3600)
        assert cleaned == 0


class TestGossipHealthTrackerEdgeCases:
    """Test edge cases."""

    def test_threshold_of_1(self):
        """Test threshold of 1 triggers immediately."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=1)
        should_emit, count = tracker.record_gossip_failure("peer1")
        assert count == 1
        assert should_emit

    def test_large_failure_count(self):
        """Test large failure counts work correctly."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker()
        for _ in range(100):
            tracker.record_gossip_failure("peer1")

        assert tracker.get_failure_count("peer1") == 100

    def test_rapid_success_failure_cycles(self):
        """Test rapid success/failure cycles."""
        from scripts.p2p.gossip_protocol import GossipHealthTracker

        tracker = GossipHealthTracker(failure_threshold=2)

        for _ in range(10):
            tracker.record_gossip_failure("peer1")
            tracker.record_gossip_failure("peer1")
            assert tracker.is_suspected("peer1")

            tracker.record_gossip_success("peer1")
            assert not tracker.is_suspected("peer1")
            assert tracker.get_failure_count("peer1") == 0

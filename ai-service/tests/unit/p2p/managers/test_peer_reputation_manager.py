"""Unit tests for PeerReputationManager.

Tests peer reputation tracking, score computation, and summary generation.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from scripts.p2p.managers.peer_reputation_manager import (
    PeerReputation,
    PeerReputationConfig,
    PeerReputationManager,
    ReputationSummary,
    create_peer_reputation_manager,
    get_peer_reputation_manager,
    set_peer_reputation_manager,
)


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestPeerReputationConfig:
    """Tests for PeerReputationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PeerReputationConfig()
        assert config.recent_window_seconds == 3600.0
        assert config.recency_decay_hours == 24.0
        assert config.recent_weight == 70.0
        assert config.historical_weight == 20.0
        assert config.recency_weight == 10.0
        assert config.reliable_threshold == 70.0
        assert config.unreliable_threshold == 30.0
        assert config.top_reliable_count == 5
        assert config.bottom_unreliable_count == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PeerReputationConfig(
            recent_window_seconds=1800.0,
            recency_decay_hours=12.0,
            recent_weight=50.0,
            historical_weight=30.0,
            recency_weight=20.0,
            reliable_threshold=80.0,
            unreliable_threshold=20.0,
            top_reliable_count=10,
            bottom_unreliable_count=5,
        )
        assert config.recent_window_seconds == 1800.0
        assert config.recency_decay_hours == 12.0
        assert config.recent_weight == 50.0
        assert config.historical_weight == 30.0
        assert config.recency_weight == 20.0
        assert config.reliable_threshold == 80.0
        assert config.unreliable_threshold == 20.0
        assert config.top_reliable_count == 10
        assert config.bottom_unreliable_count == 5


class TestPeerReputation:
    """Tests for PeerReputation dataclass."""

    def test_default_values(self):
        """Test default reputation values."""
        rep = PeerReputation()
        assert rep.total_success == 0
        assert rep.total_failure == 0
        assert rep.recent_success == 0
        assert rep.recent_failure == 0
        assert rep.last_success == 0.0
        assert rep.last_failure == 0.0
        assert rep.last_reset > 0  # Should be set to current time
        assert rep.by_type == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rep = PeerReputation(
            total_success=10,
            total_failure=2,
            recent_success=5,
            recent_failure=1,
            last_success=1000.0,
            last_failure=900.0,
            last_reset=800.0,
            by_type={"sync": {"success": 5, "failure": 1}},
        )
        d = rep.to_dict()
        assert d["total_success"] == 10
        assert d["total_failure"] == 2
        assert d["recent_success"] == 5
        assert d["recent_failure"] == 1
        assert d["last_success"] == 1000.0
        assert d["last_failure"] == 900.0
        assert d["last_reset"] == 800.0
        assert d["by_type"] == {"sync": {"success": 5, "failure": 1}}

    def test_to_dict_copies_by_type(self):
        """Test that to_dict returns a copy of by_type."""
        original_by_type = {"sync": {"success": 5, "failure": 1}}
        rep = PeerReputation(by_type=original_by_type)
        d = rep.to_dict()
        d["by_type"]["gossip"] = {"success": 1, "failure": 0}
        assert "gossip" not in original_by_type


class TestReputationSummary:
    """Tests for ReputationSummary dataclass."""

    def test_default_values(self):
        """Test default summary values."""
        summary = ReputationSummary()
        assert summary.reliable_peers == []
        assert summary.unreliable_peers == []

    def test_custom_values(self):
        """Test custom summary values."""
        summary = ReputationSummary(
            reliable_peers=[{"peer": "p1", "score": 90}],
            unreliable_peers=[{"peer": "p2", "score": 20}],
        )
        assert len(summary.reliable_peers) == 1
        assert len(summary.unreliable_peers) == 1
        assert summary.reliable_peers[0]["peer"] == "p1"
        assert summary.unreliable_peers[0]["peer"] == "p2"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = ReputationSummary(
            reliable_peers=[{"peer": "p1", "score": 90}],
            unreliable_peers=[{"peer": "p2", "score": 20}],
        )
        d = summary.to_dict()
        assert "reliable_peers" in d
        assert "unreliable_peers" in d
        assert d["reliable_peers"] == [{"peer": "p1", "score": 90}]
        assert d["unreliable_peers"] == [{"peer": "p2", "score": 20}]


# =============================================================================
# MANAGER INITIALIZATION TESTS
# =============================================================================


class TestPeerReputationManagerInit:
    """Tests for PeerReputationManager initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        manager = PeerReputationManager()
        assert manager._config is not None
        assert manager._config.recent_window_seconds == 3600.0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = PeerReputationConfig(recent_window_seconds=1800.0)
        manager = PeerReputationManager(config=config)
        assert manager._config.recent_window_seconds == 1800.0

    def test_empty_reputations_on_init(self):
        """Test that reputations dict is empty on init."""
        manager = PeerReputationManager()
        assert manager.get_peer_count() == 0
        assert manager.get_all_peers() == []


# =============================================================================
# INTERACTION RECORDING TESTS
# =============================================================================


class TestInteractionRecording:
    """Tests for interaction recording methods."""

    def test_record_success(self):
        """Test recording a successful interaction."""
        manager = PeerReputationManager()
        manager.record_success("peer-1", "sync")

        data = manager.get_peer_data("peer-1")
        assert data is not None
        assert data.total_success == 1
        assert data.total_failure == 0
        assert data.recent_success == 1
        assert data.recent_failure == 0
        assert data.by_type["sync"]["success"] == 1
        assert data.by_type["sync"]["failure"] == 0

    def test_record_failure(self):
        """Test recording a failed interaction."""
        manager = PeerReputationManager()
        manager.record_failure("peer-1", "gossip")

        data = manager.get_peer_data("peer-1")
        assert data is not None
        assert data.total_success == 0
        assert data.total_failure == 1
        assert data.recent_success == 0
        assert data.recent_failure == 1
        assert data.by_type["gossip"]["success"] == 0
        assert data.by_type["gossip"]["failure"] == 1

    def test_record_interaction_success(self):
        """Test record_interaction with success=True."""
        manager = PeerReputationManager()
        manager.record_interaction("peer-1", success=True, interaction_type="heartbeat")

        data = manager.get_peer_data("peer-1")
        assert data.total_success == 1
        assert data.by_type["heartbeat"]["success"] == 1

    def test_record_interaction_failure(self):
        """Test record_interaction with success=False."""
        manager = PeerReputationManager()
        manager.record_interaction("peer-1", success=False, interaction_type="command")

        data = manager.get_peer_data("peer-1")
        assert data.total_failure == 1
        assert data.by_type["command"]["failure"] == 1

    def test_multiple_interactions(self):
        """Test recording multiple interactions for same peer."""
        manager = PeerReputationManager()

        for _ in range(5):
            manager.record_success("peer-1")
        for _ in range(2):
            manager.record_failure("peer-1")

        data = manager.get_peer_data("peer-1")
        assert data.total_success == 5
        assert data.total_failure == 2
        assert data.recent_success == 5
        assert data.recent_failure == 2

    def test_multiple_peers(self):
        """Test recording interactions for multiple peers."""
        manager = PeerReputationManager()

        manager.record_success("peer-1")
        manager.record_success("peer-2")
        manager.record_failure("peer-3")

        assert manager.get_peer_count() == 3
        assert set(manager.get_all_peers()) == {"peer-1", "peer-2", "peer-3"}

    def test_multiple_interaction_types(self):
        """Test recording multiple interaction types."""
        manager = PeerReputationManager()

        manager.record_success("peer-1", "sync")
        manager.record_success("peer-1", "gossip")
        manager.record_failure("peer-1", "sync")

        data = manager.get_peer_data("peer-1")
        assert data.by_type["sync"]["success"] == 1
        assert data.by_type["sync"]["failure"] == 1
        assert data.by_type["gossip"]["success"] == 1
        assert data.by_type["gossip"]["failure"] == 0

    def test_default_interaction_type(self):
        """Test default interaction type is 'general'."""
        manager = PeerReputationManager()
        manager.record_success("peer-1")

        data = manager.get_peer_data("peer-1")
        assert "general" in data.by_type
        assert data.by_type["general"]["success"] == 1

    def test_last_success_timestamp_updated(self):
        """Test that last_success timestamp is updated."""
        manager = PeerReputationManager()
        before = time.time()
        manager.record_success("peer-1")
        after = time.time()

        data = manager.get_peer_data("peer-1")
        assert before <= data.last_success <= after

    def test_last_failure_timestamp_updated(self):
        """Test that last_failure timestamp is updated."""
        manager = PeerReputationManager()
        before = time.time()
        manager.record_failure("peer-1")
        after = time.time()

        data = manager.get_peer_data("peer-1")
        assert before <= data.last_failure <= after

    def test_recent_window_reset(self):
        """Test that recent counters reset when window expires."""
        config = PeerReputationConfig(recent_window_seconds=0.1)  # 100ms window
        manager = PeerReputationManager(config=config)

        manager.record_success("peer-1")
        manager.record_success("peer-1")

        data = manager.get_peer_data("peer-1")
        assert data.recent_success == 2

        # Wait for window to expire
        time.sleep(0.15)

        # Record another interaction - should reset recent counters
        manager.record_success("peer-1")

        data = manager.get_peer_data("peer-1")
        assert data.total_success == 3  # Historical preserved
        assert data.recent_success == 1  # Reset and new count


# =============================================================================
# SCORE COMPUTATION TESTS
# =============================================================================


class TestScoreComputation:
    """Tests for reputation score computation."""

    def test_neutral_score_for_unknown_peer(self):
        """Test that unknown peers get neutral 50 score."""
        manager = PeerReputationManager()
        score = manager.get_reputation_score("unknown-peer")
        assert score == 50.0

    def test_perfect_score(self):
        """Test score for peer with all successes."""
        manager = PeerReputationManager()

        for _ in range(10):
            manager.record_success("peer-1")

        score = manager.get_reputation_score("peer-1")
        # With all successes and recent activity, should be high
        assert score >= 90.0

    def test_low_score(self):
        """Test score for peer with all failures."""
        manager = PeerReputationManager()

        for _ in range(10):
            manager.record_failure("peer-1")

        score = manager.get_reputation_score("peer-1")
        # With all failures, should be low (but includes recency bonus)
        assert score <= 20.0

    def test_mixed_score(self):
        """Test score for peer with mixed results."""
        manager = PeerReputationManager()

        for _ in range(7):
            manager.record_success("peer-1")
        for _ in range(3):
            manager.record_failure("peer-1")

        score = manager.get_reputation_score("peer-1")
        # 70% success rate
        assert 50.0 <= score <= 90.0

    def test_score_bounds(self):
        """Test that scores are bounded 0-100."""
        manager = PeerReputationManager()

        manager.record_success("good-peer")
        manager.record_failure("bad-peer")

        good_score = manager.get_reputation_score("good-peer")
        bad_score = manager.get_reputation_score("bad-peer")

        assert 0.0 <= good_score <= 100.0
        assert 0.0 <= bad_score <= 100.0

    def test_get_all_scores(self):
        """Test getting scores for all peers."""
        manager = PeerReputationManager()

        manager.record_success("peer-1")
        manager.record_failure("peer-2")
        manager.record_success("peer-3")
        manager.record_success("peer-3")

        scores = manager.get_all_scores()

        assert len(scores) == 3
        assert "peer-1" in scores
        assert "peer-2" in scores
        assert "peer-3" in scores
        assert scores["peer-1"] > scores["peer-2"]  # Success vs failure

    def test_recency_bonus_decay(self):
        """Test that recency bonus decays over time."""
        config = PeerReputationConfig(recency_decay_hours=0.001)  # ~3.6 seconds
        manager = PeerReputationManager(config=config)

        manager.record_success("peer-1")
        initial_score = manager.get_reputation_score("peer-1")

        # Wait for some decay
        time.sleep(0.1)

        later_score = manager.get_reputation_score("peer-1")

        # Score should decrease due to recency decay
        assert later_score <= initial_score

    def test_historical_vs_recent_weight(self):
        """Test that recent interactions are weighted more heavily."""
        config = PeerReputationConfig(
            recent_window_seconds=0.05,  # 50ms window - very short
            recent_weight=70.0,
            historical_weight=20.0,
            recency_weight=10.0,
        )
        manager = PeerReputationManager(config=config)

        # Build historical failure record
        for _ in range(100):
            manager.record_failure("peer-1")

        # Wait for recent window to reset
        time.sleep(0.1)

        # Add recent successes (after window reset)
        for _ in range(10):
            manager.record_success("peer-1")

        # The peer has ~9% historical success but 100% recent success
        # Score should reflect the heavier recent weighting
        score = manager.get_reputation_score("peer-1")
        # Recent: 100% success (10/10), Historical: ~9% (10/110)
        # Weighted: 70*1.0 + 20*0.09 + 10*recency ≈ 71.8 + recency
        assert score > 65.0  # Recent success should pull score up significantly


# =============================================================================
# SUMMARY GENERATION TESTS
# =============================================================================


class TestSummaryGeneration:
    """Tests for reputation summary generation."""

    def test_empty_summary(self):
        """Test summary with no peers."""
        manager = PeerReputationManager()
        summary = manager.get_reputation_summary()

        assert summary.reliable_peers == []
        assert summary.unreliable_peers == []

    def test_reliable_peers_in_summary(self):
        """Test that reliable peers appear in summary."""
        config = PeerReputationConfig(
            reliable_threshold=70.0,
            top_reliable_count=3,
        )
        manager = PeerReputationManager(config=config)

        # Create reliable peers
        for i in range(5):
            peer_id = f"reliable-{i}"
            for _ in range(20):
                manager.record_success(peer_id)

        summary = manager.get_reputation_summary()

        assert len(summary.reliable_peers) <= 3  # Limited by top_reliable_count
        for p in summary.reliable_peers:
            assert p["score"] >= 70

    def test_unreliable_peers_in_summary(self):
        """Test that unreliable peers appear in summary."""
        config = PeerReputationConfig(
            unreliable_threshold=30.0,
            bottom_unreliable_count=2,
        )
        manager = PeerReputationManager(config=config)

        # Create unreliable peers
        for i in range(5):
            peer_id = f"unreliable-{i}"
            for _ in range(20):
                manager.record_failure(peer_id)

        summary = manager.get_reputation_summary()

        assert len(summary.unreliable_peers) <= 2
        for p in summary.unreliable_peers:
            assert p["score"] < 30

    def test_mixed_peers_summary(self):
        """Test summary with mix of reliable and unreliable peers."""
        manager = PeerReputationManager()

        # Reliable peer
        for _ in range(20):
            manager.record_success("reliable-1")

        # Unreliable peer
        for _ in range(20):
            manager.record_failure("unreliable-1")

        # Medium peer
        for _ in range(10):
            manager.record_success("medium-1")
            manager.record_failure("medium-1")

        summary = manager.get_reputation_summary()

        # Check structure
        assert isinstance(summary.reliable_peers, list)
        assert isinstance(summary.unreliable_peers, list)

    def test_summary_has_peer_and_score(self):
        """Test that summary entries have peer and score keys."""
        manager = PeerReputationManager()

        for _ in range(10):
            manager.record_success("peer-1")

        summary = manager.get_reputation_summary()

        if summary.reliable_peers:
            entry = summary.reliable_peers[0]
            assert "peer" in entry
            assert "score" in entry
            assert isinstance(entry["score"], int)  # Rounded


# =============================================================================
# PEER DATA ACCESS TESTS
# =============================================================================


class TestPeerDataAccess:
    """Tests for peer data access methods."""

    def test_get_peer_data_returns_none_for_unknown(self):
        """Test that get_peer_data returns None for unknown peer."""
        manager = PeerReputationManager()
        data = manager.get_peer_data("unknown-peer")
        assert data is None

    def test_get_peer_data_returns_copy(self):
        """Test that get_peer_data returns a copy."""
        manager = PeerReputationManager()
        manager.record_success("peer-1")

        data1 = manager.get_peer_data("peer-1")
        data1.total_success = 999

        data2 = manager.get_peer_data("peer-1")
        assert data2.total_success == 1  # Original value

    def test_get_all_peers(self):
        """Test getting list of all tracked peers."""
        manager = PeerReputationManager()

        manager.record_success("peer-1")
        manager.record_success("peer-2")
        manager.record_success("peer-3")

        peers = manager.get_all_peers()

        assert len(peers) == 3
        assert set(peers) == {"peer-1", "peer-2", "peer-3"}

    def test_get_peer_count(self):
        """Test getting peer count."""
        manager = PeerReputationManager()

        assert manager.get_peer_count() == 0

        manager.record_success("peer-1")
        assert manager.get_peer_count() == 1

        manager.record_success("peer-2")
        assert manager.get_peer_count() == 2

        manager.record_success("peer-1")  # Same peer again
        assert manager.get_peer_count() == 2

    def test_remove_peer(self):
        """Test removing a peer."""
        manager = PeerReputationManager()

        manager.record_success("peer-1")
        manager.record_success("peer-2")

        result = manager.remove_peer("peer-1")

        assert result is True
        assert manager.get_peer_count() == 1
        assert manager.get_peer_data("peer-1") is None

    def test_remove_unknown_peer(self):
        """Test removing unknown peer returns False."""
        manager = PeerReputationManager()
        result = manager.remove_peer("unknown-peer")
        assert result is False

    def test_clear_all(self):
        """Test clearing all reputation data."""
        manager = PeerReputationManager()

        manager.record_success("peer-1")
        manager.record_success("peer-2")
        manager.record_success("peer-3")

        manager.clear_all()

        assert manager.get_peer_count() == 0
        assert manager.get_all_peers() == []


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_healthy_no_peers(self):
        """Test health check with no peers tracked."""
        manager = PeerReputationManager()
        health = manager.health_check()

        assert health["status"] == "healthy"
        assert "operational" in health["message"].lower()
        assert health["details"]["peer_count"] == 0

    def test_healthy_with_peers(self):
        """Test health check with healthy peers."""
        manager = PeerReputationManager()

        for i in range(5):
            for _ in range(10):
                manager.record_success(f"peer-{i}")

        health = manager.health_check()

        assert health["status"] == "healthy"
        assert health["details"]["peer_count"] == 5
        assert health["details"]["avg_score"] > 80
        assert health["details"]["reliable_count"] > 0

    def test_degraded_with_many_unreliable(self):
        """Test degraded status when >50% unreliable peers."""
        manager = PeerReputationManager()

        # Create 3 unreliable peers
        for i in range(3):
            for _ in range(10):
                manager.record_failure(f"bad-{i}")

        # Create 1 reliable peer
        for _ in range(10):
            manager.record_success("good-1")

        health = manager.health_check()

        assert health["status"] == "degraded"
        assert "unreliable" in health["message"].lower()

    def test_health_check_details(self):
        """Test that health check returns expected details."""
        manager = PeerReputationManager()

        for _ in range(10):
            manager.record_success("peer-1")
        for _ in range(10):
            manager.record_failure("peer-2")

        health = manager.health_check()

        assert "peer_count" in health["details"]
        assert "avg_score" in health["details"]
        assert "reliable_count" in health["details"]
        assert "unreliable_count" in health["details"]
        assert health["details"]["peer_count"] == 2


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def setup_method(self):
        """Clear singleton before each test."""
        set_peer_reputation_manager(None)

    def teardown_method(self):
        """Clear singleton after each test."""
        set_peer_reputation_manager(None)

    def test_get_returns_none_initially(self):
        """Test that get_peer_reputation_manager returns None initially."""
        assert get_peer_reputation_manager() is None

    def test_set_and_get(self):
        """Test setting and getting the manager."""
        manager = PeerReputationManager()
        set_peer_reputation_manager(manager)

        retrieved = get_peer_reputation_manager()
        assert retrieved is manager

    def test_set_none_clears(self):
        """Test that setting None clears the manager."""
        manager = PeerReputationManager()
        set_peer_reputation_manager(manager)
        set_peer_reputation_manager(None)

        assert get_peer_reputation_manager() is None

    def test_create_sets_singleton(self):
        """Test that create_peer_reputation_manager sets singleton."""
        manager = create_peer_reputation_manager()

        retrieved = get_peer_reputation_manager()
        assert retrieved is manager

    def test_create_with_config(self):
        """Test creating manager with custom config."""
        config = PeerReputationConfig(recent_window_seconds=1800.0)
        manager = create_peer_reputation_manager(config=config)

        assert manager._config.recent_window_seconds == 1800.0


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_record_interactions(self):
        """Test concurrent interaction recording."""
        manager = PeerReputationManager()
        errors = []

        def record_many(peer_id: str, count: int, success: bool):
            try:
                for _ in range(count):
                    if success:
                        manager.record_success(peer_id)
                    else:
                        manager.record_failure(peer_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=("peer-1", 100, True)),
            threading.Thread(target=record_many, args=("peer-1", 100, False)),
            threading.Thread(target=record_many, args=("peer-2", 100, True)),
            threading.Thread(target=record_many, args=("peer-2", 100, False)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        data1 = manager.get_peer_data("peer-1")
        data2 = manager.get_peer_data("peer-2")

        assert data1.total_success + data1.total_failure == 200
        assert data2.total_success + data2.total_failure == 200

    def test_concurrent_score_computation(self):
        """Test concurrent score computation."""
        manager = PeerReputationManager()
        manager.record_success("peer-1")
        manager.record_failure("peer-1")

        scores = []
        errors = []

        def get_score_many(count: int):
            try:
                for _ in range(count):
                    score = manager.get_reputation_score("peer-1")
                    scores.append(score)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=get_score_many, args=(50,)),
            threading.Thread(target=get_score_many, args=(50,)),
            threading.Thread(target=get_score_many, args=(50,)),
            threading.Thread(target=get_score_many, args=(50,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(scores) == 200
        # Scores may vary slightly due to time-based recency bonus
        # But they should all be within a small range
        min_score = min(scores)
        max_score = max(scores)
        assert max_score - min_score < 1.0  # Less than 1 point variation

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        manager = PeerReputationManager()
        errors = []

        def writer():
            try:
                for i in range(100):
                    manager.record_success(f"peer-{i % 10}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    manager.get_all_scores()
                    manager.get_reputation_summary()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_peer_id(self):
        """Test handling empty peer ID."""
        manager = PeerReputationManager()
        manager.record_success("")

        data = manager.get_peer_data("")
        assert data is not None
        assert data.total_success == 1

    def test_special_characters_in_peer_id(self):
        """Test peer IDs with special characters."""
        manager = PeerReputationManager()
        special_ids = [
            "peer-with-dashes",
            "peer_with_underscores",
            "peer.with.dots",
            "peer:with:colons",
            "peer/with/slashes",
        ]

        for peer_id in special_ids:
            manager.record_success(peer_id)

        assert manager.get_peer_count() == len(special_ids)

    def test_unicode_peer_id(self):
        """Test unicode peer ID."""
        manager = PeerReputationManager()
        manager.record_success("peer-日本語")

        data = manager.get_peer_data("peer-日本語")
        assert data is not None
        assert data.total_success == 1

    def test_very_long_peer_id(self):
        """Test very long peer ID."""
        manager = PeerReputationManager()
        long_id = "peer-" + "x" * 1000
        manager.record_success(long_id)

        data = manager.get_peer_data(long_id)
        assert data is not None
        assert data.total_success == 1

    def test_zero_weight_config(self):
        """Test configuration with zero weights."""
        config = PeerReputationConfig(
            recent_weight=0.0,
            historical_weight=0.0,
            recency_weight=0.0,
        )
        manager = PeerReputationManager(config=config)

        manager.record_success("peer-1")
        score = manager.get_reputation_score("peer-1")

        assert score == 0.0  # All weights are zero

    def test_extreme_interactions(self):
        """Test with extreme number of interactions."""
        manager = PeerReputationManager()

        for _ in range(10000):
            manager.record_success("busy-peer")

        data = manager.get_peer_data("busy-peer")
        assert data.total_success == 10000

        score = manager.get_reputation_score("busy-peer")
        assert score >= 90.0

    def test_score_after_clear_all(self):
        """Test score computation after clearing all data."""
        manager = PeerReputationManager()

        manager.record_success("peer-1")
        manager.clear_all()

        score = manager.get_reputation_score("peer-1")
        assert score == 50.0  # Neutral for unknown peer

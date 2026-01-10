"""Tests for ClusterAnalytics manager.

January 2026: Tests for cluster analytics extracted from p2p_orchestrator.py.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.managers.cluster_analytics import (
    ClusterAnalytics,
    ClusterAnalyticsConfig,
    EloSummary,
    ReputationSummary,
    create_cluster_analytics,
    get_cluster_analytics,
    set_cluster_analytics,
)


class TestClusterAnalyticsConfig:
    """Tests for ClusterAnalyticsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClusterAnalyticsConfig()
        assert config.elo_cache_ttl == 600.0
        assert config.startup_grace_period == 60.0
        assert config.stale_gossip_threshold == 300.0
        assert config.top_models_limit == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClusterAnalyticsConfig(
            elo_cache_ttl=300.0,
            startup_grace_period=30.0,
            stale_gossip_threshold=120.0,
            top_models_limit=10,
        )
        assert config.elo_cache_ttl == 300.0
        assert config.startup_grace_period == 30.0
        assert config.stale_gossip_threshold == 120.0
        assert config.top_models_limit == 10


class TestEloSummary:
    """Tests for EloSummary dataclass."""

    def test_default_values(self):
        """Test default EloSummary values."""
        summary = EloSummary()
        assert summary.top_models == []
        assert summary.total_models == 0
        assert summary.last_update == 0.0
        assert summary.deferred is False

    def test_to_dict(self):
        """Test EloSummary.to_dict()."""
        summary = EloSummary(
            top_models=[{"model": "test", "elo": 1500}],
            total_models=10,
            last_update=123456.0,
            deferred=True,
        )
        result = summary.to_dict()
        assert result["top_models"] == [{"model": "test", "elo": 1500}]
        assert result["total_models"] == 10
        assert result["last_update"] == 123456.0
        assert result["deferred"] is True


class TestReputationSummary:
    """Tests for ReputationSummary dataclass."""

    def test_default_values(self):
        """Test default ReputationSummary values."""
        summary = ReputationSummary()
        assert summary.reliable_peers == []
        assert summary.unreliable_peers == []

    def test_to_dict(self):
        """Test ReputationSummary.to_dict()."""
        summary = ReputationSummary(
            reliable_peers=[{"peer": "node1", "score": 90}],
            unreliable_peers=[{"peer": "node2", "score": 20}],
        )
        result = summary.to_dict()
        assert result["reliable_peers"] == [{"peer": "node1", "score": 90}]
        assert result["unreliable_peers"] == [{"peer": "node2", "score": 20}]


class TestClusterAnalyticsInit:
    """Tests for ClusterAnalytics initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default callbacks."""
        analytics = ClusterAnalytics()
        assert analytics._config.elo_cache_ttl == 600.0
        assert analytics._get_gossip_states() == {}
        assert analytics._get_peer_reputation() == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ClusterAnalyticsConfig(elo_cache_ttl=300.0)
        analytics = ClusterAnalytics(config=config)
        assert analytics._config.elo_cache_ttl == 300.0

    def test_init_with_callbacks(self):
        """Test initialization with custom callbacks."""
        gossip_callback = MagicMock(return_value={"node1": {"data": "test"}})
        analytics = ClusterAnalytics(get_gossip_states=gossip_callback)
        result = analytics._get_gossip_states()
        assert result == {"node1": {"data": "test"}}
        gossip_callback.assert_called_once()


class TestDistributedTrainingSummary:
    """Tests for distributed training summary."""

    def test_empty_training_configs(self):
        """Test with no active training."""
        analytics = ClusterAnalytics()
        summary = analytics.get_distributed_training_summary()
        assert summary["active_configs"] == []
        assert summary["total_training_jobs"] == 0
        assert summary["configs_by_node_count"] == {}

    def test_with_training_configs(self):
        """Test with active training configs."""
        training_configs = {
            "hex8_2p": ["node1", "node2"],
            "square8_4p": ["node3"],
        }
        analytics = ClusterAnalytics(
            get_training_configs=lambda: training_configs
        )
        summary = analytics.get_distributed_training_summary()
        assert set(summary["active_configs"]) == {"hex8_2p", "square8_4p"}
        assert summary["total_training_jobs"] == 3
        assert summary["configs_by_node_count"]["hex8_2p"] == 2
        assert summary["configs_by_node_count"]["square8_4p"] == 1


class TestLocalEloSummary:
    """Tests for local ELO summary."""

    def test_deferred_during_startup(self):
        """Test ELO query is deferred during startup grace period."""
        config = ClusterAnalyticsConfig(startup_grace_period=60.0)
        analytics = ClusterAnalytics(config=config)
        # Just created, should be in grace period
        summary = analytics.get_local_elo_summary()
        assert summary.deferred is True
        assert summary.top_models == []

    def test_uses_cache(self):
        """Test that cached results are returned within TTL."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0, elo_cache_ttl=600.0)
        analytics = ClusterAnalytics(config=config)

        # Pre-populate cache
        cached_summary = EloSummary(
            top_models=[{"model": "cached", "elo": 1600}],
            total_models=5,
        )
        analytics._elo_cache = cached_summary
        analytics._elo_cache_time = time.time()

        result = analytics.get_local_elo_summary()
        assert result.top_models == [{"model": "cached", "elo": 1600}]

    def test_handles_database_error(self):
        """Test graceful handling of database errors."""
        import sys

        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        analytics = ClusterAnalytics(config=config)
        analytics._elo_cache = None  # Force DB query

        # Mock app.tournament to raise ImportError on get_elo_database
        orig_module = sys.modules.pop("app.tournament", None)
        try:
            class BrokenModule:
                def __getattr__(self, name):
                    raise ImportError("Mocked ImportError")

            sys.modules["app.tournament"] = BrokenModule()
            summary = analytics.get_local_elo_summary()

            assert summary.top_models == []
            assert summary.total_models == 0
        finally:
            sys.modules.pop("app.tournament", None)
            if orig_module is not None:
                sys.modules["app.tournament"] = orig_module


class TestClusterEloSummary:
    """Tests for cluster-wide ELO summary."""

    def test_empty_gossip(self):
        """Test with no gossip states."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        analytics = ClusterAnalytics(config=config)
        # Pre-populate cache to avoid DB query
        analytics._elo_cache = EloSummary()
        analytics._elo_cache_time = time.time()

        summary = analytics.get_cluster_elo_summary()
        assert summary["top_models"] == []
        assert summary["total_unique_models"] == 0

    def test_aggregates_from_gossip(self):
        """Test aggregation of ELO from gossip states."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        now = time.time()
        gossip_states = {
            "node1": {
                "timestamp": now,
                "elo_summary": {
                    "top_models": [{"model": "model_a", "elo": 1500}]
                }
            },
            "node2": {
                "timestamp": now,
                "elo_summary": {
                    "top_models": [{"model": "model_b", "elo": 1600}]
                }
            },
        }
        analytics = ClusterAnalytics(
            config=config,
            get_gossip_states=lambda: gossip_states,
        )
        analytics._elo_cache = EloSummary()
        analytics._elo_cache_time = time.time()

        summary = analytics.get_cluster_elo_summary()
        assert summary["total_unique_models"] == 2
        # model_b should be first (higher ELO)
        assert summary["top_models"][0]["model"] == "model_b"
        assert summary["top_models"][0]["elo"] == 1600

    def test_keeps_highest_elo_per_model(self):
        """Test that highest ELO is kept for duplicate models."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        now = time.time()
        gossip_states = {
            "node1": {
                "timestamp": now,
                "elo_summary": {
                    "top_models": [{"model": "model_a", "elo": 1500}]
                }
            },
            "node2": {
                "timestamp": now,
                "elo_summary": {
                    "top_models": [{"model": "model_a", "elo": 1700}]
                }
            },
        }
        analytics = ClusterAnalytics(
            config=config,
            get_gossip_states=lambda: gossip_states,
        )
        analytics._elo_cache = EloSummary()
        analytics._elo_cache_time = time.time()

        summary = analytics.get_cluster_elo_summary()
        assert summary["total_unique_models"] == 1
        assert summary["top_models"][0]["elo"] == 1700

    def test_skips_stale_gossip(self):
        """Test that stale gossip states are skipped."""
        config = ClusterAnalyticsConfig(
            startup_grace_period=0.0,
            stale_gossip_threshold=300.0,
        )
        old_time = time.time() - 400  # 400s ago, stale
        gossip_states = {
            "stale_node": {
                "timestamp": old_time,
                "elo_summary": {
                    "top_models": [{"model": "stale_model", "elo": 2000}]
                }
            },
        }
        analytics = ClusterAnalytics(
            config=config,
            get_gossip_states=lambda: gossip_states,
        )
        analytics._elo_cache = EloSummary()
        analytics._elo_cache_time = time.time()

        summary = analytics.get_cluster_elo_summary()
        # Stale node should be skipped
        assert summary["total_unique_models"] == 0


class TestPeerReputationSummary:
    """Tests for peer reputation summary."""

    def test_empty_reputation(self):
        """Test with no reputation data."""
        analytics = ClusterAnalytics()
        summary = analytics.get_peer_reputation_summary()
        assert summary.reliable_peers == []
        assert summary.unreliable_peers == []

    def test_classifies_reliable_peers(self):
        """Test classification of reliable peers (score >= 70)."""
        reputation_data = {"peer1": {}, "peer2": {}, "peer3": {}}
        score_map = {"peer1": 90, "peer2": 80, "peer3": 50}
        analytics = ClusterAnalytics(
            get_peer_reputation=lambda: reputation_data,
            get_reputation_score=lambda p: score_map.get(p, 50),
        )
        summary = analytics.get_peer_reputation_summary()
        reliable = [p["peer"] for p in summary.reliable_peers]
        assert "peer1" in reliable
        assert "peer2" in reliable
        assert "peer3" not in reliable

    def test_classifies_unreliable_peers(self):
        """Test classification of unreliable peers (score < 30)."""
        reputation_data = {"peer1": {}, "peer2": {}, "peer3": {}}
        score_map = {"peer1": 20, "peer2": 25, "peer3": 50}
        analytics = ClusterAnalytics(
            get_peer_reputation=lambda: reputation_data,
            get_reputation_score=lambda p: score_map.get(p, 50),
        )
        summary = analytics.get_peer_reputation_summary()
        unreliable = [p["peer"] for p in summary.unreliable_peers]
        assert "peer1" in unreliable
        assert "peer2" in unreliable
        assert "peer3" not in unreliable


class TestClusterPeerReputation:
    """Tests for cluster-wide peer reputation."""

    def test_aggregates_reputation(self):
        """Test aggregation of reputation from gossip."""
        now = time.time()
        gossip_states = {
            "node1": {
                "timestamp": now,
                "peer_reputation": {
                    "reliable_peers": [{"peer": "peer_a", "score": 90}]
                }
            },
            "node2": {
                "timestamp": now,
                "peer_reputation": {
                    "reliable_peers": [{"peer": "peer_a", "score": 80}]
                }
            },
        }
        analytics = ClusterAnalytics(
            get_gossip_states=lambda: gossip_states,
        )
        result = analytics.get_cluster_peer_reputation()
        # Should have peer_a with average score
        assert result["peers_tracked"] >= 1
        peer_a = next(
            (p for p in result["most_reliable"] if p["peer"] == "peer_a"), None
        )
        assert peer_a is not None
        assert peer_a["avg_score"] == 85  # (90 + 80) / 2

    def test_skips_stale_gossip(self):
        """Test that stale gossip is skipped."""
        config = ClusterAnalyticsConfig(stale_gossip_threshold=300.0)
        old_time = time.time() - 400
        gossip_states = {
            "stale_node": {
                "timestamp": old_time,
                "peer_reputation": {
                    "reliable_peers": [{"peer": "stale_peer", "score": 100}]
                }
            },
        }
        analytics = ClusterAnalytics(
            config=config,
            get_gossip_states=lambda: gossip_states,
        )
        result = analytics.get_cluster_peer_reputation()
        # Stale node should be skipped
        stale_peer = next(
            (p for p in result["most_reliable"] if p["peer"] == "stale_peer"), None
        )
        assert stale_peer is None


class TestSyncIntervalSummary:
    """Tests for sync interval summary."""

    def test_returns_intervals(self):
        """Test that sync intervals are returned."""
        intervals = {"data": 120, "model": 300, "training_db": 600}
        analytics = ClusterAnalytics(
            get_sync_intervals=lambda: intervals,
            get_sync_streaks=lambda: ({"data": 5}, {"data": 0}),
            get_activity_factor=lambda: 1.5,
        )
        summary = analytics.get_sync_interval_summary()
        assert summary["data_interval"] == 120
        assert summary["model_interval"] == 300
        assert summary["training_db_interval"] == 600
        assert summary["activity_factor"] == 1.5
        assert summary["data_streak"]["success"] == 5
        assert summary["data_streak"]["failure"] == 0


class TestDedupSummary:
    """Tests for deduplication summary."""

    def test_returns_dedup_stats(self):
        """Test that deduplication stats are returned."""
        stats = {
            "files_skipped": 100,
            "games_skipped": 500,
            "bytes_saved": 1024 * 1024 * 50,  # 50 MB
            "known_file_hashes": 200,
            "known_game_ids": 1000,
        }
        analytics = ClusterAnalytics(get_dedup_stats=lambda: stats)
        summary = analytics.get_dedup_summary()
        assert summary["files_skipped"] == 100
        assert summary["games_skipped"] == 500
        assert summary["bytes_saved_mb"] == 50.0
        assert summary["known_file_hashes"] == 200
        assert summary["known_game_ids"] == 1000

    def test_handles_missing_stats(self):
        """Test handling of missing stats keys."""
        analytics = ClusterAnalytics(get_dedup_stats=lambda: {})
        summary = analytics.get_dedup_summary()
        assert summary["files_skipped"] == 0
        assert summary["games_skipped"] == 0
        assert summary["bytes_saved_mb"] == 0.0


class TestFullAnalyticsSummary:
    """Tests for full analytics summary."""

    def test_returns_all_sections(self):
        """Test that all analytics sections are included."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        analytics = ClusterAnalytics(config=config)
        analytics._elo_cache = EloSummary()
        analytics._elo_cache_time = time.time()

        summary = analytics.get_full_analytics_summary()
        assert "training" in summary
        assert "elo" in summary
        assert "reputation" in summary
        assert "sync_intervals" in summary
        assert "deduplication" in summary


class TestHealthCheck:
    """Tests for health check."""

    def test_healthy_status(self):
        """Test healthy status returned after startup."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        analytics = ClusterAnalytics(config=config)
        analytics._elo_cache = EloSummary(total_models=10)
        analytics._elo_cache_time = time.time()

        health = analytics.health_check()
        assert health["status"] == "healthy"
        assert "ClusterAnalytics" in health["message"]

    def test_degraded_during_startup(self):
        """Test degraded status during startup."""
        config = ClusterAnalyticsConfig(startup_grace_period=60.0)
        analytics = ClusterAnalytics(config=config)

        health = analytics.health_check()
        assert health["status"] == "degraded"
        assert "deferred" in health["message"].lower()

    def test_includes_details(self):
        """Test that health check includes details."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        analytics = ClusterAnalytics(config=config)
        analytics._elo_cache = EloSummary(total_models=10)
        analytics._elo_cache_time = time.time()

        health = analytics.health_check()
        assert "elo_models_tracked" in health["details"]
        assert "startup_age_seconds" in health["details"]


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_get_analytics_initially_none(self):
        """Test that get_cluster_analytics() returns None initially."""
        set_cluster_analytics(None)
        assert get_cluster_analytics() is None

    def test_set_and_get_analytics(self):
        """Test setting and getting analytics instance."""
        analytics = ClusterAnalytics()
        set_cluster_analytics(analytics)
        assert get_cluster_analytics() is analytics
        set_cluster_analytics(None)  # Clean up

    def test_create_cluster_analytics(self):
        """Test create_cluster_analytics() creates and registers instance."""
        set_cluster_analytics(None)
        analytics = create_cluster_analytics(
            config=ClusterAnalyticsConfig(elo_cache_ttl=300.0)
        )
        assert analytics is not None
        assert get_cluster_analytics() is analytics
        assert analytics._config.elo_cache_ttl == 300.0
        set_cluster_analytics(None)  # Clean up

    def test_create_with_callbacks(self):
        """Test create_cluster_analytics() with callbacks."""
        set_cluster_analytics(None)
        training_configs = {"hex8_2p": ["node1"]}
        analytics = create_cluster_analytics(
            get_training_configs=lambda: training_configs
        )
        summary = analytics.get_distributed_training_summary()
        assert summary["total_training_jobs"] == 1
        set_cluster_analytics(None)  # Clean up


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_elo_access(self):
        """Test concurrent access to ELO cache."""
        config = ClusterAnalyticsConfig(startup_grace_period=0.0)
        analytics = ClusterAnalytics(config=config)
        analytics._elo_cache = EloSummary(total_models=10)
        analytics._elo_cache_time = time.time()

        results = []
        errors = []

        def get_elo():
            try:
                for _ in range(10):
                    summary = analytics.get_local_elo_summary()
                    results.append(summary.total_models)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_elo) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == 10 for r in results)

    def test_concurrent_singleton_access(self):
        """Test concurrent access to singleton."""
        set_cluster_analytics(None)
        errors = []
        instances = []

        def access_singleton():
            try:
                analytics = create_cluster_analytics()
                instances.append(id(analytics))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_singleton) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        set_cluster_analytics(None)  # Clean up
        assert len(errors) == 0

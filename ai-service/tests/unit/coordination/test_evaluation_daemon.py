"""Tests for EvaluationDaemon - Phase 2 test coverage (December 2025).

Tests cover:
- Configuration dataclasses (EvaluationConfig, EvaluationStats)
- Initialization and lifecycle
- Event subscriptions
- Deduplication (cooldown, content hash, concurrent)
- Queue management and backpressure
- Health check and status reporting
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.evaluation_daemon import (
    EvaluationConfig,
    EvaluationDaemon,
    EvaluationStats,
    get_evaluation_daemon,
)


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values.

        December 30, 2025: Updated to match current defaults in EvaluationConfig.
        """
        config = EvaluationConfig()
        assert config.games_per_baseline == 50
        assert config.baselines == ["random", "heuristic"]
        assert config.early_stopping_enabled is True
        assert config.early_stopping_confidence == 0.95
        assert config.early_stopping_min_games == 10
        assert config.max_concurrent_evaluations == 24
        assert config.evaluation_timeout_seconds == 300.0  # 5 minutes
        assert config.dedup_cooldown_seconds == 300.0
        assert config.max_queue_depth == 100
        assert config.backpressure_threshold == 70
        assert config.backpressure_release_threshold == 35

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = EvaluationConfig(
            games_per_baseline=50,
            baselines=["random"],
            early_stopping_enabled=False,
            max_concurrent_evaluations=8,
        )
        assert config.games_per_baseline == 50
        assert config.baselines == ["random"]
        assert config.early_stopping_enabled is False
        assert config.max_concurrent_evaluations == 8


class TestEvaluationStats:
    """Tests for EvaluationStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = EvaluationStats()
        assert stats.evaluations_triggered == 0
        assert stats.evaluations_completed == 0
        assert stats.evaluations_failed == 0
        assert stats.games_played == 0
        assert stats.models_evaluated == 0
        assert stats.promotions_triggered == 0

    def test_backward_compat_aliases(self):
        """Test backward compatibility aliases."""
        stats = EvaluationStats()
        stats.games_played = 100
        stats.avg_evaluation_duration = 60.0

        assert stats.total_games_played == 100
        assert stats.average_evaluation_time == 60.0

    def test_is_healthy(self):
        """Test is_healthy from base class.

        Note: is_healthy() checks errors_count and consecutive_failures
        from the base DaemonStatsBase class, not evaluations_failed.
        """
        stats = EvaluationStats()
        stats.operations_attempted = 10
        stats.errors_count = 0
        stats.consecutive_failures = 0
        assert stats.is_healthy() is True

        # High error rate (>10%) should return False
        stats.errors_count = 5  # 50% error rate
        assert stats.is_healthy() is False

        # Reset and test consecutive failures
        stats.errors_count = 0
        stats.consecutive_failures = 5
        assert stats.is_healthy() is False


class TestEvaluationDaemonInit:
    """Tests for EvaluationDaemon initialization."""

    def test_default_initialization(self):
        """Test default daemon initialization."""
        daemon = EvaluationDaemon()

        assert daemon.config.games_per_baseline == 50  # Updated Dec 30, 2025
        assert daemon._running is False
        assert daemon._active_evaluations == set()
        assert daemon._recently_evaluated == {}
        assert daemon._seen_event_hashes == set()
        assert daemon._backpressure_active is False

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = EvaluationConfig(
            games_per_baseline=100,
            max_concurrent_evaluations=2,
        )
        daemon = EvaluationDaemon(config=config)

        assert daemon.config.games_per_baseline == 100
        assert daemon.config.max_concurrent_evaluations == 2


class TestEventSubscriptions:
    """Tests for event subscription wiring."""

    def test_subscriptions_defined(self):
        """Test that event subscriptions are properly defined."""
        daemon = EvaluationDaemon()
        subscriptions = daemon._get_subscriptions()

        # Should subscribe to TRAINING_COMPLETED
        from app.coordination.event_router import DataEventType
        assert DataEventType.TRAINING_COMPLETED in subscriptions

    def test_handler_is_callable(self):
        """Test that subscription handler is callable."""
        daemon = EvaluationDaemon()
        subscriptions = daemon._get_subscriptions()

        for event_type, handler in subscriptions.items():
            assert callable(handler), f"Handler for {event_type} is not callable"


class TestDeduplication:
    """Tests for deduplication logic."""

    def test_compute_event_hash(self):
        """Test event hash computation."""
        daemon = EvaluationDaemon()

        hash1 = daemon._compute_event_hash(
            "models/hex8_2p.pth", "hex8", 2
        )
        hash2 = daemon._compute_event_hash(
            "models/hex8_2p.pth", "hex8", 2
        )
        hash3 = daemon._compute_event_hash(
            "models/square8_4p.pth", "square8", 4
        )

        # Same input should produce same hash
        assert hash1 == hash2
        # Different input should produce different hash
        assert hash1 != hash3
        # Hash should be 16 chars (truncated SHA256)
        assert len(hash1) == 16

    def test_is_duplicate_event_first_time(self):
        """Test that first occurrence is not duplicate."""
        daemon = EvaluationDaemon()

        hash1 = daemon._compute_event_hash("models/test.pth", "hex8", 2)
        assert daemon._is_duplicate_event(hash1) is False

    def test_is_duplicate_event_second_time(self):
        """Test that second occurrence is duplicate."""
        daemon = EvaluationDaemon()

        hash1 = daemon._compute_event_hash("models/test.pth", "hex8", 2)
        daemon._is_duplicate_event(hash1)  # First time
        assert daemon._is_duplicate_event(hash1) is True  # Second time

    def test_is_in_cooldown_false(self):
        """Test model not in cooldown."""
        daemon = EvaluationDaemon()

        assert daemon._is_in_cooldown("models/test.pth") is False

    def test_is_in_cooldown_true(self):
        """Test model in cooldown after evaluation."""
        daemon = EvaluationDaemon()

        # Simulate recent evaluation
        daemon._recently_evaluated["models/test.pth"] = time.time()

        assert daemon._is_in_cooldown("models/test.pth") is True

    def test_cooldown_expires(self):
        """Test that cooldown expires after configured time."""
        config = EvaluationConfig(dedup_cooldown_seconds=1.0)
        daemon = EvaluationDaemon(config=config)

        # Simulate old evaluation
        daemon._recently_evaluated["models/test.pth"] = time.time() - 2.0

        assert daemon._is_in_cooldown("models/test.pth") is False


class TestQueueManagement:
    """Tests for evaluation queue management."""

    def test_queue_starts_empty(self):
        """Test that queue starts empty."""
        daemon = EvaluationDaemon()
        assert daemon._evaluation_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_queue_accepts_items(self):
        """Test that items can be added to queue."""
        daemon = EvaluationDaemon()

        await daemon._evaluation_queue.put({"model": "test"})
        assert daemon._evaluation_queue.qsize() == 1


class TestBackpressure:
    """Tests for backpressure handling (Phase 4)."""

    def test_initial_state_no_backpressure(self):
        """Test initial state has no backpressure."""
        daemon = EvaluationDaemon()

        assert daemon._backpressure_active is False
        assert daemon._backpressure_stats["backpressure_activations"] == 0
        assert daemon._backpressure_stats["backpressure_releases"] == 0

    def test_backpressure_config(self):
        """Test backpressure configuration."""
        config = EvaluationConfig(
            backpressure_threshold=30,
            backpressure_release_threshold=10,
        )
        daemon = EvaluationDaemon(config=config)

        assert daemon.config.backpressure_threshold == 30
        assert daemon.config.backpressure_release_threshold == 10


class TestStatusAndHealth:
    """Tests for status and health check reporting."""

    def test_get_status(self):
        """Test get_status returns proper structure."""
        daemon = EvaluationDaemon()
        status = daemon.get_status()

        # Should have key fields
        assert "queue_size" in status
        assert "active_evaluations" in status
        assert "stats" in status
        assert "dedup_stats" in status
        assert "config" in status

        # Stats should be dict
        assert isinstance(status["stats"], dict)
        assert "evaluations_triggered" in status["stats"]
        assert "evaluations_completed" in status["stats"]

    def test_is_running_false_initially(self):
        """Test is_running returns False initially."""
        daemon = EvaluationDaemon()
        assert daemon.is_running() is False


class TestLifecycle:
    """Tests for daemon lifecycle (start/stop)."""

    @pytest.mark.asyncio
    async def test_start_creates_worker(self):
        """Test that start() creates worker task."""
        daemon = EvaluationDaemon()

        # Mock the parent class start() to avoid event subscription
        with patch.object(EvaluationDaemon.__bases__[0], 'start', new_callable=AsyncMock, return_value=True):
            success = await daemon.start()

        assert success is True
        assert daemon._worker_task is not None

        # Cleanup
        daemon._running = False
        if daemon._worker_task and not daemon._worker_task.done():
            daemon._worker_task.cancel()
            try:
                await daemon._worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stop_cancels_worker(self):
        """Test that stop() cancels worker task."""
        daemon = EvaluationDaemon()

        # Create a mock worker task
        async def mock_worker():
            while True:
                await asyncio.sleep(1)

        daemon._worker_task = asyncio.create_task(mock_worker())
        daemon._running = True

        # Mock parent stop to avoid event cleanup
        with patch.object(EvaluationDaemon.__bases__[0], 'stop', new_callable=AsyncMock):
            await daemon.stop()

        assert daemon._worker_task.cancelled() or daemon._worker_task.done()


class TestDedupStats:
    """Tests for deduplication statistics tracking."""

    def test_dedup_stats_initial(self):
        """Test initial dedup stats."""
        daemon = EvaluationDaemon()

        assert daemon._dedup_stats["cooldown_skips"] == 0
        assert daemon._dedup_stats["content_hash_skips"] == 0
        assert daemon._dedup_stats["concurrent_skips"] == 0

    def test_dedup_stats_in_status(self):
        """Test dedup stats included in status."""
        daemon = EvaluationDaemon()
        status = daemon.get_status()

        assert "dedup_stats" in status
        assert "cooldown_skips" in status["dedup_stats"]


class TestActiveEvaluations:
    """Tests for active evaluation tracking."""

    def test_active_evaluations_empty_initially(self):
        """Test active evaluations set is empty initially."""
        daemon = EvaluationDaemon()
        assert daemon._active_evaluations == set()

    def test_active_evaluations_tracking(self):
        """Test active evaluations can be tracked."""
        daemon = EvaluationDaemon()

        daemon._active_evaluations.add("hex8_2p")
        assert "hex8_2p" in daemon._active_evaluations
        assert len(daemon._active_evaluations) == 1

        daemon._active_evaluations.add("square8_4p")
        assert len(daemon._active_evaluations) == 2

        daemon._active_evaluations.discard("hex8_2p")
        assert "hex8_2p" not in daemon._active_evaluations
        assert len(daemon._active_evaluations) == 1


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_evaluation_daemon(self):
        """Test get_evaluation_daemon returns a daemon."""
        daemon = get_evaluation_daemon()
        assert daemon is not None
        assert isinstance(daemon, EvaluationDaemon)

    def test_get_evaluation_daemon_same_instance(self):
        """Test get_evaluation_daemon returns same instance."""
        daemon1 = get_evaluation_daemon()
        daemon2 = get_evaluation_daemon()

        # Should be same instance (singleton)
        assert daemon1 is daemon2


class TestRunCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_run_cycle_is_noop(self):
        """Test _run_cycle is a no-op (event-driven daemon)."""
        daemon = EvaluationDaemon()

        # Should not raise and should complete immediately
        await daemon._run_cycle()


# Run with: pytest tests/unit/coordination/test_evaluation_daemon.py -v

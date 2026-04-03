"""Tests for BacklogEvaluationDaemon.

Sprint 15 (January 3, 2026): Tests for OWC Model Evaluation Automation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.backlog_evaluation_daemon import (
    PRIORITY_BASE,
    PRIORITY_BEST,
    PRIORITY_CANONICAL,
    PRIORITY_LOCAL,
    PRIORITY_RECENT,
    PRIORITY_UNDERSERVED,
    BacklogEvalConfig,
    BacklogEvalStats,
    BacklogEvaluationDaemon,
    get_backlog_evaluation_daemon,
    reset_backlog_evaluation_daemon,
)


def _force_reset_singleton():
    """Force reset singleton without calling stop() (for tests)."""
    BacklogEvaluationDaemon._instance = None


# ============================================================================
# Test BacklogEvalConfig
# ============================================================================


class TestBacklogEvalConfig:
    """Tests for BacklogEvalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacklogEvalConfig()

        assert config.enabled is True
        assert config.scan_interval_seconds == 900
        assert config.batch_size == 3
        assert config.max_hourly == 10
        assert config.pause_queue_depth == 50
        assert config.canonical_priority_boost == PRIORITY_CANONICAL
        assert config.best_priority_boost == PRIORITY_BEST
        assert config.recent_priority_boost == PRIORITY_RECENT
        assert config.local_priority_boost == PRIORITY_LOCAL
        assert config.underserved_priority_boost == PRIORITY_UNDERSERVED
        assert config.stale_evaluation_days == 7
        assert config.underserved_game_threshold == 5000

    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            config = BacklogEvalConfig.from_env()

            assert config.enabled is True
            assert config.scan_interval_seconds == 900
            assert config.batch_size == 3
            assert config.max_hourly == 10
            assert config.pause_queue_depth == 50

    def test_from_env_custom_values(self):
        """Test from_env with custom environment variables."""
        env = {
            "RINGRIFT_BACKLOG_EVAL_ENABLED": "false",
            "RINGRIFT_BACKLOG_SCAN_INTERVAL": "600",
            "RINGRIFT_BACKLOG_BATCH_SIZE": "5",
            "RINGRIFT_BACKLOG_MAX_HOURLY": "20",
            "RINGRIFT_BACKLOG_PAUSE_DEPTH": "100",
        }
        with patch.dict("os.environ", env, clear=True):
            config = BacklogEvalConfig.from_env()

            assert config.enabled is False
            assert config.scan_interval_seconds == 600
            assert config.batch_size == 5
            assert config.max_hourly == 20
            assert config.pause_queue_depth == 100


# ============================================================================
# Test BacklogEvalStats
# ============================================================================


class TestBacklogEvalStats:
    """Tests for BacklogEvalStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = BacklogEvalStats()

        assert stats.discovery_cycles == 0
        assert stats.models_discovered == 0
        assert stats.models_registered == 0
        assert stats.models_queued == 0
        assert stats.models_skipped_backpressure == 0
        assert stats.models_skipped_rate_limit == 0
        assert stats.evaluations_started == 0
        assert stats.evaluations_completed == 0
        assert stats.evaluations_failed == 0
        assert stats.hourly_queued == 0
        assert stats.hourly_window_start > 0
        assert stats.last_discovery_time == 0.0
        assert stats.last_queue_time == 0.0

    def test_custom_values(self):
        """Test statistics with custom values."""
        stats = BacklogEvalStats(
            discovery_cycles=5,
            models_discovered=100,
            evaluations_completed=50,
        )

        assert stats.discovery_cycles == 5
        assert stats.models_discovered == 100
        assert stats.evaluations_completed == 50


# ============================================================================
# Test Priority Constants
# ============================================================================


class TestPriorityConstants:
    """Tests for priority scoring constants."""

    def test_priority_values(self):
        """Test priority constant values."""
        # Lower = higher priority
        assert PRIORITY_CANONICAL == -50
        assert PRIORITY_BEST == -30
        assert PRIORITY_RECENT == -10
        assert PRIORITY_LOCAL == -10
        assert PRIORITY_UNDERSERVED == -50
        assert PRIORITY_BASE == 100

    def test_canonical_is_highest_priority(self):
        """Canonical models should have highest priority."""
        assert PRIORITY_CANONICAL < PRIORITY_BEST
        assert PRIORITY_CANONICAL < PRIORITY_RECENT
        assert PRIORITY_CANONICAL < PRIORITY_LOCAL


# ============================================================================
# Test BacklogEvaluationDaemon
# ============================================================================


class TestBacklogEvaluationDaemon:
    """Tests for BacklogEvaluationDaemon class."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    def test_initialization(self):
        """Test daemon initialization."""
        daemon = BacklogEvaluationDaemon()

        assert daemon.name == "backlog_evaluation"
        assert daemon.config.enabled is True
        assert daemon.stats.discovery_cycles == 0

    def test_initialization_with_config(self):
        """Test daemon initialization with custom config."""
        config = BacklogEvalConfig(
            enabled=False,
            scan_interval_seconds=300,
            batch_size=5,
        )
        daemon = BacklogEvaluationDaemon(config)

        assert daemon.config.enabled is False
        assert daemon.config.scan_interval_seconds == 300
        assert daemon.config.batch_size == 5

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        daemon1 = BacklogEvaluationDaemon.get_instance()
        daemon2 = BacklogEvaluationDaemon.get_instance()

        assert daemon1 is daemon2

    def test_singleton_reset(self):
        """Test singleton reset."""
        daemon1 = BacklogEvaluationDaemon.get_instance()
        _force_reset_singleton()  # Use force reset for tests (avoids async issues)
        daemon2 = BacklogEvaluationDaemon.get_instance()

        assert daemon1 is not daemon2

    def test_config_property(self):
        """Test config property."""
        daemon = BacklogEvaluationDaemon()

        assert isinstance(daemon.config, BacklogEvalConfig)

    def test_stats_property(self):
        """Test stats property."""
        daemon = BacklogEvaluationDaemon()

        assert isinstance(daemon.stats, BacklogEvalStats)


# ============================================================================
# Test Event Subscriptions
# ============================================================================


class TestEventSubscriptions:
    """Tests for event subscription handlers."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    def test_get_event_subscriptions(self):
        """Test event subscriptions are defined."""
        daemon = BacklogEvaluationDaemon()
        subs = daemon._get_event_subscriptions()

        assert "evaluation_backpressure" in subs
        assert "evaluation_backpressure_released" in subs
        assert "evaluation_completed" in subs
        assert "evaluation_failed" in subs

    @pytest.mark.asyncio
    async def test_on_backpressure(self):
        """Test backpressure handler activates backpressure."""
        daemon = BacklogEvaluationDaemon()
        assert daemon._backpressure_active is False

        event = {"queue_depth": 75, "event_id": "test1"}
        await daemon._on_backpressure(event)

        assert daemon._backpressure_active is True

    @pytest.mark.asyncio
    async def test_on_backpressure_released(self):
        """Test backpressure release handler deactivates backpressure."""
        daemon = BacklogEvaluationDaemon()
        daemon._backpressure_active = True

        event = {"queue_depth": 30, "event_id": "test2"}
        await daemon._on_backpressure_released(event)

        assert daemon._backpressure_active is False

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_backlog_source(self):
        """Test evaluation completed handler for backlog models."""
        daemon = BacklogEvaluationDaemon()
        assert daemon.stats.evaluations_completed == 0

        event = {
            "source": "backlog_owc",
            "model_path": "/models/canonical_hex8_2p.pth",
            "event_id": "test3",
        }
        await daemon._on_evaluation_completed(event)

        assert daemon.stats.evaluations_completed == 1

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_non_backlog_source(self):
        """Test evaluation completed handler ignores non-backlog sources."""
        daemon = BacklogEvaluationDaemon()
        assert daemon.stats.evaluations_completed == 0

        event = {
            "source": "training",
            "model_path": "/models/canonical_hex8_2p.pth",
            "event_id": "test4",
        }
        await daemon._on_evaluation_completed(event)

        assert daemon.stats.evaluations_completed == 0

    @pytest.mark.asyncio
    async def test_on_evaluation_failed_backlog_source(self):
        """Test evaluation failed handler for backlog models."""
        daemon = BacklogEvaluationDaemon()
        assert daemon.stats.evaluations_failed == 0

        event = {
            "source": "backlog_local",
            "model_path": "/models/canonical_hex8_2p.pth",
            "event_id": "test5",
        }
        await daemon._on_evaluation_failed(event)

        assert daemon.stats.evaluations_failed == 1


# ============================================================================
# Test Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    def test_check_hourly_rate_limit_below_limit(self):
        """Test rate limit check when below limit."""
        config = BacklogEvalConfig(max_hourly=10)
        daemon = BacklogEvaluationDaemon(config)
        daemon._stats.hourly_queued = 5

        assert daemon._check_hourly_rate_limit() is True

    def test_check_hourly_rate_limit_at_limit(self):
        """Test rate limit check when at limit."""
        config = BacklogEvalConfig(max_hourly=10)
        daemon = BacklogEvaluationDaemon(config)
        daemon._stats.hourly_queued = 10

        assert daemon._check_hourly_rate_limit() is False

    def test_check_hourly_rate_limit_above_limit(self):
        """Test rate limit check when above limit."""
        config = BacklogEvalConfig(max_hourly=10)
        daemon = BacklogEvaluationDaemon(config)
        daemon._stats.hourly_queued = 15

        assert daemon._check_hourly_rate_limit() is False

    def test_check_hourly_rate_limit_resets_after_hour(self):
        """Test rate limit resets after an hour."""
        config = BacklogEvalConfig(max_hourly=10)
        daemon = BacklogEvaluationDaemon(config)
        daemon._stats.hourly_queued = 15
        daemon._stats.hourly_window_start = time.time() - 3700  # Over an hour ago

        # Check should reset counter and return True
        assert daemon._check_hourly_rate_limit() is True
        assert daemon._stats.hourly_queued == 0


# ============================================================================
# Test Priority Scoring
# ============================================================================


class TestPriorityScoring:
    """Tests for model priority scoring."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    def _create_mock_model(
        self,
        name: str = "model.pth",
        is_canonical: bool = False,
        is_best: bool = False,
        modified_at: float | None = None,
        source: str = "owc",
        config_key: str | None = "hex8_2p",
    ):
        """Create a mock DiscoveredModel."""
        model = MagicMock()
        model.file_name = name
        model.is_canonical = is_canonical
        model.is_best = is_best
        model.modified_at = modified_at
        model.source = source
        model.config_key = config_key
        return model

    def test_prioritize_empty_list(self):
        """Test prioritizing empty list."""
        daemon = BacklogEvaluationDaemon()
        result = daemon._prioritize_models([])

        assert result == []

    def test_prioritize_canonical_first(self):
        """Test canonical models get highest priority."""
        daemon = BacklogEvaluationDaemon()
        daemon._underserved_configs = set()

        regular = self._create_mock_model("regular.pth")
        canonical = self._create_mock_model("canonical_hex8_2p.pth", is_canonical=True)

        result = daemon._prioritize_models([regular, canonical])

        # Canonical should be first (lower score)
        assert result[0] is canonical
        assert result[1] is regular

    def test_prioritize_best_before_regular(self):
        """Test 'best' models get priority over regular."""
        daemon = BacklogEvaluationDaemon()
        daemon._underserved_configs = set()

        regular = self._create_mock_model("regular.pth")
        best = self._create_mock_model("best_hex8_2p.pth", is_best=True)

        result = daemon._prioritize_models([regular, best])

        assert result[0] is best
        assert result[1] is regular

    def test_prioritize_recent_models(self):
        """Test recent models get priority boost."""
        daemon = BacklogEvaluationDaemon()
        daemon._underserved_configs = set()

        old = self._create_mock_model("old.pth", modified_at=time.time() - 86400 * 30)
        recent = self._create_mock_model("recent.pth", modified_at=time.time() - 86400)

        result = daemon._prioritize_models([old, recent])

        assert result[0] is recent
        assert result[1] is old

    def test_prioritize_local_models(self):
        """Test local models get priority boost."""
        daemon = BacklogEvaluationDaemon()
        daemon._underserved_configs = set()
        daemon._underserved_cache_time = time.time()  # Skip refresh

        remote = self._create_mock_model("remote.pth", source="owc")
        local = self._create_mock_model("local.pth", source="local")

        result = daemon._prioritize_models([remote, local])

        assert result[0] is local
        assert result[1] is remote

    def test_prioritize_underserved_configs(self):
        """Test underserved configs get priority boost."""
        daemon = BacklogEvaluationDaemon()
        daemon._underserved_configs = {"hex8_4p"}
        daemon._underserved_cache_time = time.time()  # Skip refresh

        served = self._create_mock_model("served.pth", config_key="hex8_2p")
        underserved = self._create_mock_model("underserved.pth", config_key="hex8_4p")

        result = daemon._prioritize_models([served, underserved])

        assert result[0] is underserved
        assert result[1] is served

    def test_prioritize_combined_factors(self):
        """Test combined priority factors."""
        daemon = BacklogEvaluationDaemon()
        daemon._underserved_configs = {"hex8_4p"}
        daemon._underserved_cache_time = time.time()  # Skip refresh

        # Regular model
        regular = self._create_mock_model("regular.pth", config_key="hex8_2p")

        # Canonical + underserved = highest priority
        canonical_underserved = self._create_mock_model(
            "canonical_hex8_4p.pth",
            is_canonical=True,
            config_key="hex8_4p",
        )

        result = daemon._prioritize_models([regular, canonical_underserved])

        assert result[0] is canonical_underserved


# ============================================================================
# Test Health Check
# ============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    def test_health_check_disabled(self):
        """Test health check when daemon is disabled."""
        config = BacklogEvalConfig(enabled=False)
        daemon = BacklogEvaluationDaemon(config)

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "disabled"
        assert "disabled" in result.message.lower()

    def test_health_check_healthy(self):
        """Test health check when daemon is healthy."""
        daemon = BacklogEvaluationDaemon()
        daemon._stats.last_discovery_time = time.time()
        daemon._stats.models_discovered = 100

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "healthy"
        assert "100" in result.message

    def test_health_check_backpressure(self):
        """Test health check when backpressure is active."""
        daemon = BacklogEvaluationDaemon()
        daemon._stats.last_discovery_time = time.time()
        daemon._backpressure_active = True

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "backpressure"
        assert "backpressure" in result.message.lower()

    def test_health_check_degraded(self):
        """Test health check when OWC not yet scanned."""
        daemon = BacklogEvaluationDaemon()
        daemon._stats.last_discovery_time = 0

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "degraded"

    def test_health_check_stale(self):
        """Test health check when discovery is stale."""
        daemon = BacklogEvaluationDaemon()
        daemon._stats.last_discovery_time = time.time() - 10000  # Very stale

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "warning"

    def test_health_check_details(self):
        """Test health check includes detailed stats."""
        daemon = BacklogEvaluationDaemon()
        daemon._stats.discovery_cycles = 5
        daemon._stats.models_queued = 10
        daemon._stats.evaluations_completed = 8
        daemon._stats.last_discovery_time = time.time()

        result = daemon.health_check()

        assert "discovery_cycles" in result.details
        assert result.details["discovery_cycles"] == 5
        assert result.details["models_queued"] == 10
        assert result.details["evaluations_completed"] == 8


# ============================================================================
# Test Run Cycle
# ============================================================================


class TestRunCycle:
    """Tests for main run cycle."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self):
        """Test run cycle when daemon is disabled."""
        config = BacklogEvalConfig(enabled=False)
        daemon = BacklogEvaluationDaemon(config)

        await daemon._run_cycle()

        # Should not increment any counters
        assert daemon.stats.discovery_cycles == 0

    @pytest.mark.asyncio
    async def test_run_cycle_backpressure(self):
        """Test run cycle skips when backpressure active."""
        daemon = BacklogEvaluationDaemon()
        daemon._backpressure_active = True

        await daemon._run_cycle()

        assert daemon.stats.models_skipped_backpressure == 1
        assert daemon.stats.discovery_cycles == 0

    @pytest.mark.asyncio
    async def test_run_cycle_rate_limited(self):
        """Test run cycle skips when rate limited."""
        config = BacklogEvalConfig(max_hourly=0)  # Always rate limited
        daemon = BacklogEvaluationDaemon(config)

        await daemon._run_cycle()

        assert daemon.stats.models_skipped_rate_limit == 1
        assert daemon.stats.discovery_cycles == 0


# ============================================================================
# Test Module Helpers
# ============================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    def test_get_backlog_evaluation_daemon(self):
        """Test get_backlog_evaluation_daemon returns singleton."""
        daemon1 = get_backlog_evaluation_daemon()
        daemon2 = get_backlog_evaluation_daemon()

        assert daemon1 is daemon2

    def test_get_backlog_evaluation_daemon_with_config(self):
        """Test get_backlog_evaluation_daemon with custom config."""
        config = BacklogEvalConfig(batch_size=5)
        daemon = get_backlog_evaluation_daemon(config)

        assert daemon.config.batch_size == 5

    def test_reset_backlog_evaluation_daemon(self):
        """Test reset clears singleton."""
        daemon1 = get_backlog_evaluation_daemon()
        _force_reset_singleton()  # Use force reset for tests (avoids async issues)
        daemon2 = get_backlog_evaluation_daemon()

        assert daemon1 is not daemon2


# ============================================================================
# Test Queue For Evaluation
# ============================================================================


class TestQueueForEvaluation:
    """Tests for _queue_for_evaluation method."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    @pytest.mark.asyncio
    async def test_queue_rejects_model_without_config_key(self):
        """Test queue rejects models without config_key."""
        daemon = BacklogEvaluationDaemon()

        model = MagicMock()
        model.config_key = None
        model.file_name = "test.pth"

        result = await daemon._queue_for_evaluation(model)

        assert result is False
        assert daemon.stats.evaluations_started == 0

    @pytest.mark.asyncio
    @patch("app.coordination.backlog_evaluation_daemon.BacklogEvaluationDaemon._emit_model_queued")
    @patch(
        "app.coordination.backlog_evaluation_daemon.BacklogEvaluationDaemon._emit_synthetic_training_completed"
    )
    async def test_queue_success(self, mock_emit_tc, mock_emit_queued):
        """Test successful model queuing."""
        daemon = BacklogEvaluationDaemon()

        # Mock the tracker
        mock_tracker = MagicMock()
        daemon._tracker = mock_tracker

        model = MagicMock()
        model.config_key = "hex8_2p"
        model.sha256 = "abc123def456"
        model.board_type = "hex8"
        model.num_players = 2
        model.file_name = "canonical_hex8_2p.pth"

        result = await daemon._queue_for_evaluation(model)

        assert result is True
        assert daemon.stats.evaluations_started == 1
        mock_emit_queued.assert_called_once_with(model)
        mock_emit_tc.assert_called_once_with(model)


# ============================================================================
# Test Event Emission
# ============================================================================


class TestEventEmission:
    """Tests for event emission methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        _force_reset_singleton()

    def teardown_method(self):
        """Reset singleton after each test."""
        _force_reset_singleton()

    @patch("app.coordination.event_emission_helpers.safe_emit_event")
    def test_emit_discovery_completed(self, mock_emit):
        """Test discovery completed event emission."""
        daemon = BacklogEvaluationDaemon()
        daemon._stats.discovery_cycles = 5
        daemon._stats.evaluations_completed = 10
        daemon._stats.evaluations_failed = 2

        daemon._emit_discovery_completed(100, 5)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        payload = call_args[0][1]
        assert payload["total_models"] == 100
        assert payload["queued"] == 5
        assert payload["discovery_cycles"] == 5
        assert payload["evaluations_completed"] == 10
        assert payload["evaluations_failed"] == 2

    @patch("app.coordination.event_emission_helpers.safe_emit_event")
    def test_emit_model_queued(self, mock_emit):
        """Test model queued event emission."""
        daemon = BacklogEvaluationDaemon()

        model = MagicMock()
        model.path = "/models/canonical_hex8_2p.pth"
        model.file_name = "canonical_hex8_2p.pth"
        model.config_key = "hex8_2p"
        model.board_type = "hex8"
        model.num_players = 2

        daemon._emit_model_queued(model)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        payload = call_args[0][1]
        assert payload["model_path"] == "/models/canonical_hex8_2p.pth"
        assert payload["config_key"] == "hex8_2p"
        assert payload["source"] == "owc"

    @patch("app.coordination.event_emission_helpers.safe_emit_event")
    def test_emit_synthetic_training_completed(self, mock_emit):
        """Test synthetic TRAINING_COMPLETED event emission."""
        daemon = BacklogEvaluationDaemon()

        model = MagicMock()
        model.path = "/models/canonical_hex8_2p.pth"
        model.config_key = "hex8_2p"
        model.board_type = "hex8"
        model.num_players = 2
        model.architecture_version = "v5-heavy"
        model.sha256 = "abc123def456"

        daemon._emit_synthetic_training_completed(model)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        payload = call_args[0][1]
        assert payload["config_key"] == "hex8_2p"
        assert payload["source"] == "backlog_owc"
        assert payload["architecture"] == "v5-heavy"
        assert "job_id" in payload

"""Tests for NPZCombinationDaemon.

Tests the automated NPZ file combination daemon that triggers
quality-weighted combination after NPZ exports complete.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.npz_combination_daemon import (
    CombinationStats,
    NPZCombinationConfig,
    NPZCombinationDaemon,
    get_npz_combination_daemon,
)


class TestNPZCombinationConfig:
    """Tests for NPZCombinationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NPZCombinationConfig()

        assert config.freshness_weight == 1.5
        assert config.freshness_half_life_hours == 24.0
        assert config.min_quality_score == 0.2
        assert config.deduplicate is True
        assert config.dedup_threshold == 0.98
        assert config.output_dir == Path("data/training")
        assert config.output_suffix == "_combined"
        assert config.min_input_files == 1
        assert config.combine_on_single_file is False
        assert config.min_interval_seconds == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NPZCombinationConfig(
            freshness_weight=2.0,
            freshness_half_life_hours=12.0,
            min_quality_score=0.5,
            deduplicate=False,
            min_interval_seconds=120.0,
        )

        assert config.freshness_weight == 2.0
        assert config.freshness_half_life_hours == 12.0
        assert config.min_quality_score == 0.5
        assert config.deduplicate is False
        assert config.min_interval_seconds == 120.0

    def test_to_combiner_config(self):
        """Test conversion to NPZCombinerConfig."""
        config = NPZCombinationConfig(
            freshness_weight=1.8,
            freshness_half_life_hours=48.0,
            min_quality_score=0.3,
            deduplicate=True,
        )

        combiner_config = config.to_combiner_config()

        assert combiner_config.freshness_weight == 1.8
        assert combiner_config.freshness_half_life_hours == 48.0
        assert combiner_config.min_quality_score == 0.3
        assert combiner_config.deduplicate is True

    def test_output_dir_is_path(self):
        """Test that output_dir is a Path object."""
        config = NPZCombinationConfig()
        assert isinstance(config.output_dir, Path)

        config_custom = NPZCombinationConfig(output_dir=Path("/custom/path"))
        assert config_custom.output_dir == Path("/custom/path")


class TestCombinationStats:
    """Tests for CombinationStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = CombinationStats()

        assert stats.combinations_triggered == 0
        assert stats.combinations_succeeded == 0
        assert stats.combinations_failed == 0
        assert stats.combinations_skipped == 0
        assert stats.total_samples_combined == 0
        assert stats.samples_deduplicated == 0
        assert stats.last_combination_time == 0.0
        assert stats.last_combination_config == ""
        assert stats.last_combination_by_config == {}

    def test_stat_mutation(self):
        """Test that stats can be mutated."""
        stats = CombinationStats()

        stats.combinations_triggered += 1
        stats.combinations_succeeded += 1
        stats.total_samples_combined = 10000
        stats.last_combination_config = "hex8_2p"
        stats.last_combination_by_config["hex8_2p"] = time.time()

        assert stats.combinations_triggered == 1
        assert stats.combinations_succeeded == 1
        assert stats.total_samples_combined == 10000
        assert stats.last_combination_config == "hex8_2p"
        assert "hex8_2p" in stats.last_combination_by_config


class TestNPZCombinationDaemonInit:
    """Tests for NPZCombinationDaemon initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        NPZCombinationDaemon.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NPZCombinationDaemon.reset_instance()

    def test_initialization_default_config(self):
        """Test daemon initialization with default config."""
        daemon = NPZCombinationDaemon()

        assert daemon.name == "NPZCombinationDaemon"
        assert daemon.config is not None
        assert isinstance(daemon.config, NPZCombinationConfig)
        assert daemon.combination_stats is not None
        assert isinstance(daemon.combination_stats, CombinationStats)

    def test_initialization_custom_config(self):
        """Test daemon initialization with custom config."""
        config = NPZCombinationConfig(freshness_weight=2.0)
        daemon = NPZCombinationDaemon(config=config)

        assert daemon.config.freshness_weight == 2.0

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that get_instance returns same instance."""
        daemon1 = await NPZCombinationDaemon.get_instance()
        daemon2 = await NPZCombinationDaemon.get_instance()

        assert daemon1 is daemon2

    def test_reset_instance(self):
        """Test that reset_instance clears singleton."""
        daemon1 = NPZCombinationDaemon()
        NPZCombinationDaemon._instance = daemon1

        NPZCombinationDaemon.reset_instance()

        assert NPZCombinationDaemon._instance is None

    def test_event_subscriptions(self):
        """Test event subscription mapping."""
        daemon = NPZCombinationDaemon()

        subs = daemon._get_event_subscriptions()

        assert "npz_export_complete" in subs
        assert "npz_combination_complete" in subs
        assert callable(subs["npz_export_complete"])


class TestNPZCombinationDaemonEvents:
    """Tests for NPZCombinationDaemon event handling."""

    def setup_method(self):
        """Reset singleton before each test."""
        NPZCombinationDaemon.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NPZCombinationDaemon.reset_instance()

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_missing_config_key(self):
        """Test handling of event without config_key."""
        daemon = NPZCombinationDaemon()

        # Event without config_key should be ignored
        await daemon._on_npz_export_complete({})

        assert daemon.combination_stats.combinations_triggered == 0

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_throttling(self):
        """Test that rapid events are throttled."""
        daemon = NPZCombinationDaemon()
        daemon.config.min_interval_seconds = 60.0

        # Set last combination time to now
        daemon.combination_stats.last_combination_by_config["hex8_2p"] = time.time()

        # Event should be throttled
        await daemon._on_npz_export_complete({"config_key": "hex8_2p"})

        assert daemon.combination_stats.combinations_skipped == 1
        assert daemon.combination_stats.combinations_triggered == 0

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_triggers_combination(self):
        """Test that event triggers combination after throttle window."""
        daemon = NPZCombinationDaemon()
        daemon.config.min_interval_seconds = 0  # No throttling

        # Mock the combination method
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_samples = 1000

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock, return_value=mock_result
        ), patch.object(daemon, "_emit_combination_complete"), patch.object(
            daemon, "_is_duplicate_event", return_value=False
        ):
            await daemon._on_npz_export_complete({"config_key": "hex8_2p"})

        assert daemon.combination_stats.combinations_triggered == 1
        assert daemon.combination_stats.combinations_succeeded == 1
        assert daemon.combination_stats.total_samples_combined == 1000

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_handles_failure(self):
        """Test that failed combination is tracked."""
        daemon = NPZCombinationDaemon()
        daemon.config.min_interval_seconds = 0

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Test error"

        with patch.object(
            daemon, "_combine_for_config", new_callable=AsyncMock, return_value=mock_result
        ), patch.object(daemon, "_emit_combination_failed"), patch.object(
            daemon, "_is_duplicate_event", return_value=False
        ):
            await daemon._on_npz_export_complete({"config_key": "hex8_2p"})

        assert daemon.combination_stats.combinations_triggered == 1
        assert daemon.combination_stats.combinations_failed == 1

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_handles_exception(self):
        """Test that exceptions during combination are caught."""
        daemon = NPZCombinationDaemon()
        daemon.config.min_interval_seconds = 0

        with patch.object(
            daemon,
            "_combine_for_config",
            new_callable=AsyncMock,
            side_effect=Exception("Test exception"),
        ), patch.object(daemon, "_emit_combination_failed"), patch.object(
            daemon, "_is_duplicate_event", return_value=False
        ):
            await daemon._on_npz_export_complete({"config_key": "hex8_2p"})

        assert daemon.combination_stats.combinations_triggered == 1
        assert daemon.combination_stats.combinations_failed == 1


class TestNPZCombinationDaemonCombination:
    """Tests for NPZCombinationDaemon combination logic."""

    def setup_method(self):
        """Reset singleton before each test."""
        NPZCombinationDaemon.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NPZCombinationDaemon.reset_instance()

    @pytest.mark.asyncio
    async def test_combine_for_config_invalid_config_key(self):
        """Test that invalid config_key returns error result."""
        daemon = NPZCombinationDaemon()

        result = await daemon._combine_for_config("invalid_config")

        assert result is not None
        assert result.success is False
        assert "Invalid" in result.error or "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_combine_for_config_updates_tracking(self):
        """Test that successful combination updates tracking."""
        daemon = NPZCombinationDaemon()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_samples = 5000

        with patch(
            "app.coordination.npz_combination_daemon.discover_and_combine_for_config",
            return_value=mock_result,
        ):
            result = await daemon._combine_for_config("hex8_2p")

        assert result.success is True
        assert daemon.combination_stats.last_combination_config == "hex8_2p"
        assert daemon.combination_stats.last_combination_time > 0
        assert "hex8_2p" in daemon.combination_stats.last_combination_by_config


class TestNPZCombinationDaemonHealthCheck:
    """Tests for NPZCombinationDaemon health check."""

    def setup_method(self):
        """Reset singleton before each test."""
        NPZCombinationDaemon.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NPZCombinationDaemon.reset_instance()

    def test_health_check_initial_state(self):
        """Test health check returns result on fresh daemon."""
        daemon = NPZCombinationDaemon()

        result = daemon.health_check()

        assert result is not None
        # HealthCheckResult uses 'healthy' not 'is_healthy'
        assert hasattr(result, "healthy")
        assert hasattr(result, "details")

    def test_health_check_includes_stats(self):
        """Test health check includes combination stats."""
        daemon = NPZCombinationDaemon()
        daemon.combination_stats.combinations_succeeded = 5
        daemon.combination_stats.combinations_failed = 1

        result = daemon.health_check()

        # Stats should be in details
        details = result.details
        assert isinstance(details, dict)


class TestGetNPZCombinationDaemon:
    """Tests for get_npz_combination_daemon factory function."""

    def setup_method(self):
        """Reset singleton before each test."""
        NPZCombinationDaemon.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NPZCombinationDaemon.reset_instance()

    @pytest.mark.asyncio
    async def test_returns_daemon_instance(self):
        """Test that factory returns daemon instance."""
        daemon = await get_npz_combination_daemon()

        assert daemon is not None
        assert isinstance(daemon, NPZCombinationDaemon)

    @pytest.mark.asyncio
    async def test_returns_same_instance(self):
        """Test that factory returns singleton."""
        daemon1 = await get_npz_combination_daemon()
        daemon2 = await get_npz_combination_daemon()

        assert daemon1 is daemon2


class TestNPZCombinationDaemonLifecycle:
    """Tests for NPZCombinationDaemon lifecycle methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        NPZCombinationDaemon.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NPZCombinationDaemon.reset_instance()

    @pytest.mark.asyncio
    async def test_run_cycle_updates_stats(self):
        """Test that _run_cycle updates activity stats."""
        daemon = NPZCombinationDaemon()
        initial_cycles = daemon.stats.cycles_completed

        await daemon._run_cycle()

        assert daemon.stats.cycles_completed == initial_cycles + 1
        assert daemon.stats.last_activity > 0

    def test_daemon_not_running_initially(self):
        """Test daemon is not running initially."""
        daemon = NPZCombinationDaemon()

        # Daemon should not be running on initialization
        assert not daemon._running

    def test_daemon_inherits_handler_base(self):
        """Test that daemon inherits from HandlerBase."""
        daemon = NPZCombinationDaemon()

        # Should have HandlerBase methods
        assert hasattr(daemon, "health_check")
        assert hasattr(daemon, "_run_cycle")
        assert hasattr(daemon, "_get_event_subscriptions")
        assert hasattr(daemon, "stats")

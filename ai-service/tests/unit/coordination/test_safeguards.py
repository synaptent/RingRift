"""Tests for coordination safeguards module.

Tests SafeguardConfig, SpawnRateTracker, ResourceMonitor, and Safeguards.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from app.coordination.safeguards import (
    SafeguardConfig,
    SpawnRateTracker,
    ResourceMonitor,
    Safeguards,
    check_before_spawn,
)


class TestSafeguardConfig:
    """Tests for SafeguardConfig dataclass."""

    def test_default_values(self):
        """Test that default configuration values are sensible."""
        config = SafeguardConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 300.0
        assert config.half_open_max_calls == 3

        # Resource thresholds
        assert config.disk_critical_percent == 70.0
        assert config.memory_critical_percent == 80.0
        assert config.cpu_critical_percent == 80.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SafeguardConfig(
            failure_threshold=10,
            recovery_timeout=600.0,
            disk_critical_percent=90.0,
        )

        assert config.failure_threshold == 10
        assert config.recovery_timeout == 600.0
        assert config.disk_critical_percent == 90.0


class TestSpawnRateTracker:
    """Tests for SpawnRateTracker class."""

    def test_record_spawn(self):
        """Test recording spawns."""
        config = SafeguardConfig()
        tracker = SpawnRateTracker(config)

        tracker.record_spawn()
        tracker.record_spawn()

        rate = tracker.get_rate()
        assert rate == 2

    def test_spawn_rate_window(self):
        """Test that spawn rate is calculated within window."""
        config = SafeguardConfig(spawn_window_seconds=1.0)
        tracker = SpawnRateTracker(config)

        tracker.record_spawn()
        tracker.record_spawn()

        # Within window
        assert tracker.get_rate() == 2

        # Wait for window to expire
        time.sleep(1.1)

        # Should be zero after window expires
        assert tracker.get_rate() == 0

    def test_get_delay(self):
        """Test delay calculation."""
        config = SafeguardConfig()
        tracker = SpawnRateTracker(config)

        # No delay initially
        delay = tracker.get_delay()
        assert delay >= 0

    def test_at_limit(self):
        """Test at_limit check."""
        config = SafeguardConfig(max_spawns_per_window=2)
        tracker = SpawnRateTracker(config)

        assert tracker.at_limit() is False

        tracker.record_spawn()
        tracker.record_spawn()

        assert tracker.at_limit() is True


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def test_get_resources(self):
        """Test getting resource status."""
        config = SafeguardConfig()
        monitor = ResourceMonitor(config)

        resources = monitor.get_resources()

        # Should have all required keys
        assert "disk_percent" in resources
        assert "memory_percent" in resources
        assert "cpu_percent" in resources
        # load_avg or load_average depending on platform
        assert "load_avg" in resources or "load_average" in resources or "cpu_count" in resources

    def test_is_critical(self):
        """Test resource critical detection returns tuple."""
        config = SafeguardConfig()
        monitor = ResourceMonitor(config)

        result = monitor.is_critical()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_is_warning(self):
        """Test resource warning detection returns tuple."""
        config = SafeguardConfig()
        monitor = ResourceMonitor(config)

        result = monitor.is_warning()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


class TestSafeguards:
    """Tests for Safeguards class."""

    @pytest.fixture
    def safeguards(self, tmp_path):
        """Create safeguards instance with test config."""
        config = SafeguardConfig(
            emergency_file=str(tmp_path / "EMERGENCY_HALT"),
            max_spawns_per_window=10,
            spawn_window_seconds=60.0,
            auto_emergency_on_oom=False,  # Disable auto-halt for testing
        )
        # Reset singleton for testing
        Safeguards._instance = None
        return Safeguards(config)

    def test_allow_spawn_basic(self, safeguards):
        """Test basic spawn allowance."""
        allowed = safeguards.allow_spawn("selfplay", "node-1")
        # Should be allowed unless resources are critical
        assert isinstance(allowed, bool)

    def test_record_spawn(self, safeguards):
        """Test recording spawns."""
        safeguards.record_spawn("selfplay", "node-1")
        safeguards.record_spawn("selfplay", "node-1")

        stats = safeguards.get_stats()
        # Check that global_spawn_rate is tracked
        assert "global_spawn_rate" in stats
        assert stats["global_spawn_rate"] >= 0

    def test_record_failure(self, safeguards):
        """Test recording failures."""
        safeguards.record_failure("training", "node-2", "Test error")

        stats = safeguards.get_stats()
        assert "circuit_breakers" in stats

    def test_emergency_halt(self, safeguards, tmp_path):
        """Test emergency halt functionality."""
        # Initially not halted
        assert safeguards.is_emergency_active() is False

        # Trigger emergency halt
        safeguards.activate_emergency()

        assert safeguards.is_emergency_active() is True

        # Clear halt
        safeguards.deactivate_emergency()
        assert safeguards.is_emergency_active() is False

    def test_get_block_reason(self, safeguards):
        """Test getting block reason."""
        reason = safeguards.get_block_reason()
        # Should return a string (empty or reason)
        assert isinstance(reason, str)

    def test_get_stats(self, safeguards):
        """Test getting safeguard stats."""
        stats = safeguards.get_stats()

        assert "global_spawn_rate" in stats
        assert "emergency_active" in stats
        assert "circuit_breakers" in stats

    def test_get_delay(self, safeguards):
        """Test getting delay value."""
        delay = safeguards.get_delay()
        assert isinstance(delay, (int, float))
        assert delay >= 0


class TestCheckBeforeSpawn:
    """Tests for check_before_spawn helper function."""

    def test_returns_tuple(self, tmp_path):
        """Test that check_before_spawn returns (bool, str) tuple."""
        # Reset singleton with test config
        Safeguards._instance = None
        config = SafeguardConfig(
            emergency_file=str(tmp_path / "EMERGENCY_HALT"),
        )
        Safeguards(config)

        result = check_before_spawn("selfplay", "node-1")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

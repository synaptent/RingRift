"""Tests for unified backpressure monitoring.

Tests the BackpressureSignal and BackpressureMonitor for cluster coordination.
"""

import asyncio
import pytest
import time

from app.coordination.backpressure import (
    BackpressureConfig,
    BackpressureMonitor,
    BackpressureSignal,
    get_backpressure_monitor,
    reset_backpressure_monitor,
)


class TestBackpressureSignal:
    """Tests for BackpressureSignal dataclass."""

    def test_default_signal_is_healthy(self):
        """Default signal should indicate healthy system."""
        signal = BackpressureSignal()
        assert signal.is_healthy
        assert signal.spawn_rate_multiplier == 1.0
        assert not signal.should_pause

    def test_overall_pressure_calculation(self):
        """Overall pressure should be weighted average."""
        signal = BackpressureSignal(
            queue_pressure=1.0,  # 25%
            training_pressure=0.0,  # 20%
            disk_pressure=0.0,  # 15%
            sync_pressure=0.0,  # 15%
            memory_pressure=0.0,  # 25%
        )
        assert abs(signal.overall_pressure - 0.25) < 0.01

    def test_full_pressure_causes_pause(self):
        """Full pressure should cause spawn pause."""
        signal = BackpressureSignal(
            queue_pressure=1.0,
            training_pressure=1.0,
            disk_pressure=1.0,
            sync_pressure=1.0,
            memory_pressure=1.0,
        )
        assert signal.overall_pressure == 1.0
        assert signal.spawn_rate_multiplier == 0.0
        assert signal.should_pause

    def test_medium_pressure_throttles(self):
        """Medium pressure should reduce spawn rate."""
        signal = BackpressureSignal(
            queue_pressure=0.6,
            training_pressure=0.6,
            disk_pressure=0.6,
            sync_pressure=0.6,
            memory_pressure=0.6,
        )
        assert 0.3 < signal.overall_pressure < 0.9
        assert 0.0 < signal.spawn_rate_multiplier < 1.0

    def test_spawn_rate_multiplier_thresholds(self):
        """Test spawn rate multiplier threshold behavior."""
        # Low pressure - full speed
        low = BackpressureSignal(queue_pressure=0.2)
        assert low.spawn_rate_multiplier == 1.0

        # High pressure - stopped (all pressures at 1.0 gives overall 1.0 > 0.7)
        high = BackpressureSignal(
            queue_pressure=1.0,
            training_pressure=1.0,
            disk_pressure=1.0,
            sync_pressure=1.0,
            memory_pressure=1.0,
        )
        assert high.spawn_rate_multiplier == 0.0

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all relevant fields."""
        signal = BackpressureSignal(
            queue_pressure=0.5,
            training_pressure=0.3,
        )
        d = signal.to_dict()
        assert "queue_pressure" in d
        assert "training_pressure" in d
        assert "disk_pressure" in d
        assert "sync_pressure" in d
        assert "memory_pressure" in d
        assert "overall_pressure" in d
        assert "spawn_rate_multiplier" in d
        assert "should_pause" in d
        assert "is_healthy" in d
        assert "timestamp" in d


class TestBackpressureConfig:
    """Tests for BackpressureConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = BackpressureConfig()
        assert config.queue_low_threshold < config.queue_high_threshold
        assert config.training_low_threshold < config.training_high_threshold
        assert config.disk_low_threshold < config.disk_high_threshold
        assert config.cache_ttl_seconds > 0


class TestBackpressureMonitor:
    """Tests for BackpressureMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor for testing."""
        reset_backpressure_monitor()
        return BackpressureMonitor()

    @pytest.mark.asyncio
    async def test_get_signal_returns_signal(self, monitor):
        """get_signal should return a BackpressureSignal."""
        signal = await monitor.get_signal()
        assert isinstance(signal, BackpressureSignal)

    @pytest.mark.asyncio
    async def test_signal_is_cached(self, monitor):
        """Signal should be cached for TTL duration."""
        signal1 = await monitor.get_signal()
        signal2 = await monitor.get_signal()
        # Should be the same cached signal
        assert signal1.timestamp == signal2.timestamp

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, monitor):
        """force_refresh should bypass cache."""
        signal1 = await monitor.get_signal()
        await asyncio.sleep(0.01)
        signal2 = await monitor.get_signal(force_refresh=True)
        # Timestamps should be different
        assert signal2.timestamp >= signal1.timestamp

    def test_get_cached_signal_returns_none_initially(self, monitor):
        """get_cached_signal should return None before first get_signal."""
        assert monitor.get_cached_signal() is None

    @pytest.mark.asyncio
    async def test_get_cached_signal_after_fetch(self, monitor):
        """get_cached_signal should return signal after fetch."""
        await monitor.get_signal()
        cached = monitor.get_cached_signal()
        assert cached is not None
        assert isinstance(cached, BackpressureSignal)

    def test_normalize_function(self, monitor):
        """Test the _normalize function."""
        # Below low threshold
        assert monitor._normalize(5, 10, 100) == 0.0
        # Above high threshold
        assert monitor._normalize(150, 10, 100) == 1.0
        # At midpoint
        assert abs(monitor._normalize(55, 10, 100) - 0.5) < 0.01


class TestBackpressureMonitorSingleton:
    """Tests for singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_backpressure_monitor()

    def test_get_backpressure_monitor_returns_same_instance(self):
        """get_backpressure_monitor should return same instance."""
        m1 = get_backpressure_monitor()
        m2 = get_backpressure_monitor()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        """reset_backpressure_monitor should allow new instance."""
        m1 = get_backpressure_monitor()
        reset_backpressure_monitor()
        m2 = get_backpressure_monitor()
        assert m1 is not m2


class TestBackpressureMonitorHealthCheck:
    """Tests for health_check() method."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor for testing."""
        reset_backpressure_monitor()
        return BackpressureMonitor()

    def test_health_check_returns_result(self, monitor):
        """health_check should return a HealthCheckResult."""
        result = monitor.health_check()
        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")
        assert hasattr(result, "details")

    def test_health_check_healthy_before_first_signal(self, monitor):
        """Monitor should be healthy even before first signal fetch."""
        result = monitor.health_check()
        assert result.healthy is True
        assert result.details["has_cached_signal"] is False

    @pytest.mark.asyncio
    async def test_health_check_after_signal_fetch(self, monitor):
        """Health check should report cached signal after fetch."""
        await monitor.get_signal()
        result = monitor.health_check()
        assert result.healthy is True
        assert result.details["has_cached_signal"] is True
        assert result.details["cache_age_seconds"] >= 0

    def test_health_check_includes_config(self, monitor):
        """Health check should include config details."""
        result = monitor.health_check()
        assert "config" in result.details
        config = result.details["config"]
        assert "cache_ttl" in config
        assert "queue_low" in config
        assert "queue_high" in config


class TestNormalizeEdgeCases:
    """Tests for _normalize edge cases."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor for testing."""
        reset_backpressure_monitor()
        return BackpressureMonitor()

    def test_normalize_equal_thresholds(self, monitor):
        """Equal thresholds should return 0.0 (avoid division by zero)."""
        # When high <= low, should return 0.0
        result = monitor._normalize(50, 50, 50)
        assert result == 0.0

    def test_normalize_inverted_thresholds(self, monitor):
        """Inverted thresholds (high < low) should return 0.0."""
        result = monitor._normalize(50, 100, 10)
        assert result == 0.0

    def test_normalize_at_low_threshold(self, monitor):
        """Value at low threshold should return 0.0."""
        result = monitor._normalize(10, 10, 100)
        assert result == 0.0

    def test_normalize_at_high_threshold(self, monitor):
        """Value at high threshold should return 1.0."""
        result = monitor._normalize(100, 10, 100)
        assert result == 1.0

    def test_normalize_negative_value(self, monitor):
        """Negative values should clamp to 0.0."""
        result = monitor._normalize(-10, 0, 100)
        assert result == 0.0


class TestBackpressureConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_backpressure_monitor()

    @pytest.mark.asyncio
    async def test_get_spawn_rate_multiplier(self):
        """get_spawn_rate_multiplier should return float 0-1."""
        from app.coordination.backpressure import get_spawn_rate_multiplier

        multiplier = await get_spawn_rate_multiplier()
        assert isinstance(multiplier, float)
        assert 0.0 <= multiplier <= 1.0

    @pytest.mark.asyncio
    async def test_should_pause_spawning(self):
        """should_pause_spawning should return bool."""
        from app.coordination.backpressure import should_pause_spawning

        should_pause = await should_pause_spawning()
        assert isinstance(should_pause, bool)

    @pytest.mark.asyncio
    async def test_convenience_functions_use_singleton(self):
        """Convenience functions should use the singleton monitor."""
        from app.coordination.backpressure import (
            get_spawn_rate_multiplier,
            should_pause_spawning,
        )

        # Call convenience functions
        await get_spawn_rate_multiplier()
        await should_pause_spawning()

        # Singleton should now have a cached signal
        monitor = get_backpressure_monitor()
        assert monitor.get_cached_signal() is not None


class TestBackpressureSignalEdgeCases:
    """Additional edge case tests for BackpressureSignal."""

    def test_spawn_rate_at_exact_threshold_low(self):
        """Spawn rate at exactly 0.2 pressure should be 1.0."""
        signal = BackpressureSignal(
            queue_pressure=0.2,
            training_pressure=0.2,
            disk_pressure=0.2,
            sync_pressure=0.2,
            memory_pressure=0.2,
        )
        assert abs(signal.overall_pressure - 0.2) < 0.01
        assert signal.spawn_rate_multiplier == 1.0

    def test_spawn_rate_just_above_threshold(self):
        """Spawn rate just above 0.2 should start decreasing."""
        signal = BackpressureSignal(
            queue_pressure=0.25,
            training_pressure=0.25,
            disk_pressure=0.25,
            sync_pressure=0.25,
            memory_pressure=0.25,
        )
        assert signal.overall_pressure > 0.2
        assert signal.spawn_rate_multiplier < 1.0

    def test_source_details_field(self):
        """source_details should be accessible and default to empty dict."""
        signal = BackpressureSignal()
        assert signal.source_details == {}

        signal_with_details = BackpressureSignal(
            source_details={"queue_depth": 50}
        )
        assert signal_with_details.source_details["queue_depth"] == 50

    def test_timestamp_auto_generated(self):
        """Timestamp should be auto-generated."""
        before = time.time()
        signal = BackpressureSignal()
        after = time.time()
        assert before <= signal.timestamp <= after


class TestBackpressureConfigEdgeCases:
    """Additional tests for BackpressureConfig."""

    def test_custom_config_values(self):
        """Custom config values should be accepted."""
        config = BackpressureConfig(
            queue_low_threshold=5,
            queue_high_threshold=50,
            cache_ttl_seconds=5.0,
        )
        assert config.queue_low_threshold == 5
        assert config.queue_high_threshold == 50
        assert config.cache_ttl_seconds == 5.0

    def test_memory_thresholds(self):
        """Memory pressure thresholds should have valid defaults."""
        config = BackpressureConfig()
        assert config.memory_low_threshold < config.memory_high_threshold
        assert 0.0 <= config.memory_low_threshold <= 1.0
        assert 0.0 <= config.memory_high_threshold <= 1.0

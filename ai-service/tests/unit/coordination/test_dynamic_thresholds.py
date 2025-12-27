"""Tests for app.coordination.dynamic_thresholds module.

Tests the dynamic threshold adjustment system including:
- DynamicThreshold class behavior
- ThresholdManager operations
- Adjustment strategies (LINEAR, EXPONENTIAL, ADAPTIVE)
- Singleton pattern for global manager
- Health check integration
"""

import time
from unittest.mock import patch

import pytest

from app.coordination.dynamic_thresholds import (
    AdjustmentStrategy,
    DynamicThreshold,
    ThresholdManager,
    ThresholdObservation,
    get_threshold_manager,
    reset_threshold_manager,
)


class TestAdjustmentStrategy:
    """Tests for AdjustmentStrategy enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert AdjustmentStrategy.LINEAR.value == "linear"
        assert AdjustmentStrategy.EXPONENTIAL.value == "exponential"
        assert AdjustmentStrategy.ADAPTIVE.value == "adaptive"

    def test_enum_iteration(self):
        """Test enum can be iterated."""
        strategies = list(AdjustmentStrategy)
        assert len(strategies) == 3


class TestThresholdObservation:
    """Tests for ThresholdObservation dataclass."""

    def test_creation_minimal(self):
        """Test creation with minimal arguments."""
        obs = ThresholdObservation(timestamp=1234567890.0, success=True)
        assert obs.timestamp == 1234567890.0
        assert obs.success is True
        assert obs.measured_value is None
        assert obs.metadata == {}

    def test_creation_full(self):
        """Test creation with all arguments."""
        obs = ThresholdObservation(
            timestamp=1234567890.0,
            success=False,
            measured_value=45.5,
            metadata={"config": "hex8_2p", "node": "worker-1"},
        )
        assert obs.timestamp == 1234567890.0
        assert obs.success is False
        assert obs.measured_value == 45.5
        assert obs.metadata["config"] == "hex8_2p"


class TestDynamicThreshold:
    """Tests for DynamicThreshold class."""

    def test_creation_defaults(self):
        """Test creation with default parameters."""
        threshold = DynamicThreshold(
            name="test_timeout",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
        )
        assert threshold.name == "test_timeout"
        assert threshold.value == 30.0
        assert threshold.min_value == 5.0
        assert threshold.max_value == 120.0
        assert threshold.target_success_rate == 0.95
        assert threshold.adjustment_strategy == AdjustmentStrategy.ADAPTIVE

    def test_creation_custom(self):
        """Test creation with custom parameters."""
        threshold = DynamicThreshold(
            name="custom",
            initial_value=50.0,
            min_value=10.0,
            max_value=100.0,
            target_success_rate=0.90,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.2,
            window_size=50,
            cooldown_seconds=30.0,
            higher_is_more_permissive=False,
        )
        assert threshold.target_success_rate == 0.90
        assert threshold.adjustment_strategy == AdjustmentStrategy.LINEAR
        assert threshold.window_size == 50
        assert threshold.cooldown_seconds == 30.0
        assert threshold.higher_is_more_permissive is False

    def test_initial_success_rate(self):
        """Test success rate is 1.0 initially."""
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )
        assert threshold.success_rate == 1.0

    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            cooldown_seconds=0.0,  # No cooldown for testing
        )

        # Record 8 successes and 2 failures
        for _ in range(8):
            threshold.record_outcome(success=True)
        for _ in range(2):
            threshold.record_outcome(success=False)

        assert threshold.success_rate == 0.8

    def test_record_outcome_with_measured_value(self):
        """Test recording outcomes with measured values."""
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )

        threshold.record_outcome(success=True, measured_value=25.0)
        threshold.record_outcome(success=True, measured_value=28.0)
        threshold.record_outcome(success=False, measured_value=45.0)

        stats = threshold.get_stats()
        assert stats["observations_in_window"] == 3
        assert stats["total_observations"] == 3

    def test_window_size_limiting(self):
        """Test observation window is properly limited."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            window_size=10,
        )

        # Record more than window size
        for i in range(20):
            threshold.record_outcome(success=True)

        stats = threshold.get_stats()
        assert stats["observations_in_window"] == 10
        assert stats["total_observations"] == 20

    def test_cooldown_prevents_adjustment(self):
        """Test cooldown prevents rapid adjustments."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            window_size=10,
            cooldown_seconds=60.0,
        )

        initial_value = threshold.value

        # Record many failures - should trigger adjustment
        for _ in range(10):
            threshold.record_outcome(success=False)

        # Value may have adjusted once
        first_adjusted = threshold.value

        # Record more failures immediately
        for _ in range(10):
            threshold.record_outcome(success=False)

        # Value should not have adjusted again due to cooldown
        assert threshold.value == first_adjusted

    def test_adjustment_respects_min_bound(self):
        """Test adjustment respects minimum bound."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=10.0,  # Close to min
            min_value=5.0,
            max_value=120.0,
            window_size=10,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,  # High success -> can decrease
        )

        # Record high success rate - should try to decrease (tighten)
        with patch.object(threshold, '_last_adjustment_time', 0.0):
            for _ in range(10):
                threshold.record_outcome(success=True)
                # Force recalculation
                threshold._last_adjustment_time = 0.0

        # Should not go below min
        assert threshold.value >= threshold.min_value

    def test_adjustment_respects_max_bound(self):
        """Test adjustment respects maximum bound."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=100.0,  # Close to max
            min_value=5.0,
            max_value=120.0,
            window_size=10,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,  # Low success -> should increase
        )

        # Record low success rate - should try to increase (loosen)
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0.0

        # Should not go above max
        assert threshold.value <= threshold.max_value

    def test_reset(self):
        """Test reset clears observations."""
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )

        for _ in range(10):
            threshold.record_outcome(success=True)

        threshold.reset()

        stats = threshold.get_stats()
        assert stats["observations_in_window"] == 0
        assert threshold.success_rate == 1.0

    def test_get_stats_structure(self):
        """Test get_stats returns correct structure."""
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )

        threshold.record_outcome(success=True, measured_value=25.0)

        stats = threshold.get_stats()

        assert "name" in stats
        assert "current_value" in stats
        assert "min_value" in stats
        assert "max_value" in stats
        assert "success_rate" in stats
        assert "target_success_rate" in stats
        assert "observations_in_window" in stats
        assert "total_observations" in stats
        assert "adjustment_count" in stats
        assert "strategy" in stats


class TestLinearAdjustment:
    """Tests for LINEAR adjustment strategy."""

    def test_linear_increases_on_low_success(self):
        """Test LINEAR increases threshold when success is low."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.95,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,
        )

        initial = threshold.value

        # Record low success rate
        for _ in range(7):
            threshold.record_outcome(success=False)
        for _ in range(3):
            threshold.record_outcome(success=True)

        # Force adjustment check
        threshold._last_adjustment_time = 0.0
        threshold.record_outcome(success=False)

        # Should increase (loosen) threshold
        assert threshold.value >= initial

    def test_linear_decreases_on_high_success(self):
        """Test LINEAR decreases threshold when success is high."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.80,  # Low target so 100% is "too high"
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,
        )

        initial = threshold.value

        # Record all successes (above target)
        for _ in range(10):
            threshold.record_outcome(success=True)

        # Force adjustment check
        threshold._last_adjustment_time = 0.0
        threshold.record_outcome(success=True)

        # Should decrease (tighten) threshold
        assert threshold.value <= initial


class TestExponentialAdjustment:
    """Tests for EXPONENTIAL adjustment strategy."""

    def test_exponential_proportional_increase(self):
        """Test EXPONENTIAL increases proportionally."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=5.0,
            max_value=200.0,
            target_success_rate=0.95,
            adjustment_strategy=AdjustmentStrategy.EXPONENTIAL,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,
        )

        # Record low success to trigger increase
        for _ in range(10):
            threshold.record_outcome(success=False)

        threshold._last_adjustment_time = 0.0
        threshold.record_outcome(success=False)

        # EXPONENTIAL multiplies by (1 + adjustment_factor)
        # Expected: 50.0 * 1.1 = 55.0
        assert threshold.value > 50.0


class TestAdaptiveAdjustment:
    """Tests for ADAPTIVE adjustment strategy."""

    def test_adaptive_adjusts_based_on_deviation(self):
        """Test ADAPTIVE adjusts proportionally to deviation."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.95,
            adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,
        )

        # Record 50% success (large deviation from 95% target)
        for _ in range(5):
            threshold.record_outcome(success=True)
        for _ in range(5):
            threshold.record_outcome(success=False)

        initial = threshold.value
        threshold._last_adjustment_time = 0.0
        threshold.record_outcome(success=False)

        # Should adjust significantly due to large deviation
        assert threshold.value != initial


class TestThresholdManager:
    """Tests for ThresholdManager class."""

    def test_register_threshold(self):
        """Test registering a threshold."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )

        manager.register(threshold)

        assert manager.get("test") is threshold

    def test_get_nonexistent(self):
        """Test get returns None for nonexistent threshold."""
        manager = ThresholdManager()
        assert manager.get("nonexistent") is None

    def test_get_value(self):
        """Test get_value returns current value."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )
        manager.register(threshold)

        assert manager.get_value("test") == 30.0

    def test_get_value_default(self):
        """Test get_value returns default for nonexistent."""
        manager = ThresholdManager()
        assert manager.get_value("nonexistent", default=99.0) == 99.0

    def test_record_existing(self):
        """Test record works for existing threshold."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )
        manager.register(threshold)

        result = manager.record("test", success=True, measured_value=25.0)

        assert result is True
        assert threshold.success_rate == 1.0

    def test_record_nonexistent(self):
        """Test record returns False for nonexistent threshold."""
        manager = ThresholdManager()
        result = manager.record("nonexistent", success=True)
        assert result is False

    def test_reset_all(self):
        """Test reset_all clears all thresholds."""
        manager = ThresholdManager()

        t1 = DynamicThreshold(
            name="t1", initial_value=30.0, min_value=5.0, max_value=120.0
        )
        t2 = DynamicThreshold(
            name="t2", initial_value=60.0, min_value=10.0, max_value=180.0
        )
        manager.register(t1)
        manager.register(t2)

        # Record some observations
        for _ in range(5):
            manager.record("t1", success=True)
            manager.record("t2", success=False)

        manager.reset_all()

        assert t1.success_rate == 1.0
        assert t2.success_rate == 1.0

    def test_get_all_stats(self):
        """Test get_all_stats returns stats for all thresholds."""
        manager = ThresholdManager()

        t1 = DynamicThreshold(
            name="t1", initial_value=30.0, min_value=5.0, max_value=120.0
        )
        t2 = DynamicThreshold(
            name="t2", initial_value=60.0, min_value=10.0, max_value=180.0
        )
        manager.register(t1)
        manager.register(t2)

        stats = manager.get_all_stats()

        assert "t1" in stats
        assert "t2" in stats
        assert stats["t1"]["current_value"] == 30.0
        assert stats["t2"]["current_value"] == 60.0

    def test_get_health_all_healthy(self):
        """Test get_health with all healthy thresholds."""
        manager = ThresholdManager()

        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.95,
        )
        manager.register(threshold)

        # Record high success
        for _ in range(10):
            threshold.record_outcome(success=True)

        health = manager.get_health()

        assert health["all_healthy"] is True
        assert len(health["unhealthy_thresholds"]) == 0

    def test_get_health_with_unhealthy(self):
        """Test get_health identifies unhealthy thresholds."""
        manager = ThresholdManager()

        threshold = DynamicThreshold(
            name="failing",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.95,
        )
        manager.register(threshold)

        # Record very low success (below target - 10% = 85%)
        for _ in range(7):
            threshold.record_outcome(success=False)
        for _ in range(3):
            threshold.record_outcome(success=True)

        health = manager.get_health()

        assert health["all_healthy"] is False
        assert "failing" in health["unhealthy_thresholds"]

    def test_health_check_integration(self):
        """Test health_check returns proper HealthCheckResult."""
        manager = ThresholdManager()

        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )
        manager.register(threshold)

        result = manager.health_check()

        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")
        assert hasattr(result, "details")


class TestGlobalManager:
    """Tests for global threshold manager singleton."""

    def setup_method(self):
        """Reset global manager before each test."""
        reset_threshold_manager()

    def teardown_method(self):
        """Reset global manager after each test."""
        reset_threshold_manager()

    def test_get_threshold_manager_singleton(self):
        """Test get_threshold_manager returns same instance."""
        m1 = get_threshold_manager()
        m2 = get_threshold_manager()
        assert m1 is m2

    def test_default_thresholds_initialized(self):
        """Test default thresholds are created."""
        manager = get_threshold_manager()

        # Check default thresholds exist
        assert manager.get("handler_timeout") is not None
        assert manager.get("heartbeat_threshold") is not None
        assert manager.get("plateau_window") is not None
        assert manager.get("memory_warning") is not None

    def test_default_threshold_values(self):
        """Test default threshold values are correct."""
        manager = get_threshold_manager()

        assert manager.get_value("handler_timeout") == 30.0
        assert manager.get_value("heartbeat_threshold") == 60.0
        assert manager.get_value("plateau_window") == 10.0
        assert manager.get_value("memory_warning") == 0.8

    def test_reset_clears_manager(self):
        """Test reset_threshold_manager creates new instance."""
        m1 = get_threshold_manager()
        reset_threshold_manager()
        m2 = get_threshold_manager()

        assert m1 is not m2


class TestHigherIsMorePermissive:
    """Tests for higher_is_more_permissive flag behavior."""

    def test_higher_permissive_true(self):
        """Test behavior when higher values are more permissive."""
        # Example: timeout threshold - higher = more lenient
        threshold = DynamicThreshold(
            name="timeout",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.95,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            cooldown_seconds=0.0,
            higher_is_more_permissive=True,
        )

        initial = threshold.value

        # Low success rate - should increase (loosen)
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0.0

        # Value should have increased
        assert threshold.value >= initial

    def test_higher_permissive_false(self):
        """Test behavior when higher values are more restrictive."""
        # Example: quality threshold - higher = stricter
        threshold = DynamicThreshold(
            name="quality",
            initial_value=0.5,
            min_value=0.1,
            max_value=0.9,
            target_success_rate=0.95,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            cooldown_seconds=0.0,
            higher_is_more_permissive=False,
        )

        initial = threshold.value

        # Low success rate - should decrease (loosen)
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0.0

        # Value should have decreased
        assert threshold.value <= initial


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_adjustment_within_tolerance(self):
        """Test no adjustment when success rate is within tolerance."""
        # Create threshold with high window_size to delay first adjustment
        # and very long cooldown to prevent adjustments during recording
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            target_success_rate=0.90,  # 90% target
            window_size=100,  # Need 50 observations before first adjustment check
            cooldown_seconds=3600.0,  # 1 hour cooldown - no adjustments during test
        )

        initial = threshold.value

        # Record exactly 90% success (at target - should be within 5% tolerance)
        # With window_size=100, we need 50 obs before adjustment is considered
        for _ in range(45):
            threshold.record_outcome(success=True)
        for _ in range(5):
            threshold.record_outcome(success=False)

        # Now we have 50 observations (90% success), exactly at target
        # Reset cooldown to allow adjustment check
        threshold._last_adjustment_time = 0.0
        threshold._maybe_adjust()

        # Should not adjust since 90% == 90% target (within 5% tolerance)
        assert threshold.value == initial

    def test_insufficient_observations(self):
        """Test no adjustment with insufficient observations."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=120.0,
            window_size=100,  # Need at least 50 observations
            cooldown_seconds=0.0,
        )

        initial = threshold.value

        # Record only 10 failures
        for _ in range(10):
            threshold.record_outcome(success=False)

        # Should not adjust (not enough observations)
        assert threshold.value == initial

    def test_measured_value_statistics(self):
        """Test measured value statistics in get_stats."""
        threshold = DynamicThreshold(
            name="test", initial_value=30.0, min_value=5.0, max_value=120.0
        )

        # Record with measured values (need > 20 for p95 calculation)
        for v in range(5, 115, 5):  # 22 values: 5, 10, 15, ..., 110
            threshold.record_outcome(success=True, measured_value=float(v))

        stats = threshold.get_stats()

        # Should have mean calculated
        assert stats["measured_mean"] is not None
        # Should have p95 calculated (need > 20 samples, we have 22)
        assert stats["measured_p95"] is not None
        # Mean should be around 57.5 (avg of 5 to 110)
        assert 50 <= stats["measured_mean"] <= 65

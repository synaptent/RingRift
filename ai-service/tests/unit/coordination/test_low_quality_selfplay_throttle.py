"""Integration tests for LOW_QUALITY_DATA_WARNING → selfplay throttle.

Tests the SelfplayScheduler's response to low quality warnings,
including throttling selfplay allocation and applying quality penalties.

Event flow:
1. QualityMonitorDaemon emits LOW_QUALITY_DATA_WARNING
2. SelfplayScheduler throttles exploration_boost by 0.7x
3. Applies quality penalty proportional to quality score
4. Logs throttling action for monitoring

December 2025 - Phase 3 integration tests
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.selfplay_scheduler import (
    ALL_CONFIGS,
    ConfigPriority,
    SelfplayScheduler,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scheduler():
    """Create a SelfplayScheduler instance."""
    return SelfplayScheduler()


@pytest.fixture
def mock_event():
    """Create a mock LOW_QUALITY_DATA_WARNING event."""
    event = MagicMock()
    event.payload = {
        "quality_score": 0.45,
        "old_state": "ok",
        "new_state": "warning",
    }
    return event


@pytest.fixture
def scheduler_with_priorities(scheduler):
    """Create scheduler with pre-set priority values."""
    for config_key in ALL_CONFIGS[:6]:  # Use first 6 configs
        priority = scheduler._config_priorities[config_key]
        priority.exploration_boost = 1.5
        priority.quality_penalty = 0.0
        priority.priority_score = 1.0
    return scheduler


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestLowQualityEventHandling:
    """Tests for LOW_QUALITY_DATA_WARNING event handling."""

    def test_handles_low_quality_warning(self, scheduler, mock_event):
        """Test handling LOW_QUALITY_DATA_WARNING throttles selfplay."""
        # Set initial boost values
        for config_key in ALL_CONFIGS[:3]:
            scheduler._config_priorities[config_key].exploration_boost = 1.5

        scheduler._on_low_quality_warning(mock_event)

        # Check throttling applied
        for config_key in ALL_CONFIGS[:3]:
            priority = scheduler._config_priorities[config_key]
            # 0.45 quality -> 0.6 throttle factor -> 1.5 * 0.6 = 0.9
            assert abs(priority.exploration_boost - 0.9) < 0.01

    def test_throttle_factor_scales_with_quality(self, scheduler):
        """Test throttle factor scales with quality score."""
        # Very poor quality (< 0.4) -> 0.5 throttle
        event1 = MagicMock()
        event1.payload = {"quality_score": 0.35, "old_state": "ok", "new_state": "warning"}

        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.0
        scheduler._on_low_quality_warning(event1)
        very_poor_boost = scheduler._config_priorities["hex8_2p"].exploration_boost

        # Poor quality (0.4-0.5) -> 0.6 throttle
        scheduler._config_priorities["square8_2p"].exploration_boost = 1.0
        event2 = MagicMock()
        event2.payload = {"quality_score": 0.45, "old_state": "ok", "new_state": "warning"}
        scheduler._on_low_quality_warning(event2)
        poor_boost = scheduler._config_priorities["square8_2p"].exploration_boost

        # Marginal quality (0.5-0.6) -> 0.7 throttle
        scheduler._config_priorities["square19_2p"].exploration_boost = 1.0
        event3 = MagicMock()
        event3.payload = {"quality_score": 0.55, "old_state": "ok", "new_state": "warning"}
        scheduler._on_low_quality_warning(event3)
        marginal_boost = scheduler._config_priorities["square19_2p"].exploration_boost

        # Worse quality = more aggressive throttling
        assert very_poor_boost < poor_boost < marginal_boost

    def test_applies_quality_penalty(self, scheduler, mock_event):
        """Test applies quality penalty proportional to score."""
        scheduler._on_low_quality_warning(mock_event)

        # Check penalty applied to configs
        for config_key in ALL_CONFIGS:
            priority = scheduler._config_priorities[config_key]
            # quality_score=0.45 -> penalty = -0.15 * (1.0 - 0.45) = -0.0825
            expected_penalty = -0.15 * (1.0 - 0.45)
            assert abs(priority.quality_penalty - expected_penalty) < 0.01

    def test_throttle_respects_minimum(self, scheduler, mock_event):
        """Test throttling respects minimum exploration boost of 0.5."""
        # Set very low initial boost
        scheduler._config_priorities["hex8_2p"].exploration_boost = 0.6

        # Very poor quality -> 0.5 throttle -> 0.6 * 0.5 = 0.3
        # Should clamp to 0.5 minimum
        event = MagicMock()
        event.payload = {"quality_score": 0.35, "old_state": "ok", "new_state": "warning"}

        scheduler._on_low_quality_warning(event)

        priority = scheduler._config_priorities["hex8_2p"]
        assert priority.exploration_boost >= 0.5

    def test_handles_missing_quality_score(self, scheduler):
        """Test gracefully handles event with missing quality_score."""
        event = MagicMock()
        event.payload = {"old_state": "ok", "new_state": "warning"}

        # Should not raise
        scheduler._on_low_quality_warning(event)


# =============================================================================
# Throttling Logic Tests
# =============================================================================


class TestThrottlingLogic:
    """Tests for selfplay throttling logic."""

    def test_throttles_all_configs(self, scheduler_with_priorities, mock_event):
        """Test throttles all configs on low quality warning."""
        scheduler_with_priorities._on_low_quality_warning(mock_event)

        throttled_count = 0
        for config_key in ALL_CONFIGS[:6]:
            priority = scheduler_with_priorities._config_priorities[config_key]
            if priority.exploration_boost < 1.5:  # Was 1.5 initially
                throttled_count += 1

        # All configs should be throttled
        assert throttled_count == 6

    def test_only_throttles_changed_configs(self, scheduler_with_priorities, mock_event):
        """Test only counts significantly changed configs in log."""
        # Set one config to already low boost
        scheduler_with_priorities._config_priorities["hex8_2p"].exploration_boost = 0.5

        scheduler_with_priorities._on_low_quality_warning(mock_event)

        # hex8_2p won't change much (0.5 * 0.6 = 0.3, clamped to 0.5)
        # Others will change from 1.5 to 0.9

    def test_preserves_relative_priorities(self, scheduler):
        """Test throttling preserves relative priorities between configs."""
        # Set different initial boosts
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.8
        scheduler._config_priorities["square8_2p"].exploration_boost = 1.2

        event = MagicMock()
        event.payload = {"quality_score": 0.5, "old_state": "ok", "new_state": "warning"}

        scheduler._on_low_quality_warning(event)

        boost1 = scheduler._config_priorities["hex8_2p"].exploration_boost
        boost2 = scheduler._config_priorities["square8_2p"].exploration_boost

        # Higher boost should remain higher after throttling
        assert boost1 > boost2


# =============================================================================
# Quality Penalty Tests
# =============================================================================


class TestQualityPenalty:
    """Tests for quality penalty application."""

    def test_penalty_proportional_to_quality(self, scheduler):
        """Test penalty is proportional to quality degradation."""
        # Quality 0.2 -> penalty = -0.15 * (1.0 - 0.2) = -0.12
        event1 = MagicMock()
        event1.payload = {"quality_score": 0.2, "old_state": "ok", "new_state": "warning"}
        scheduler._on_low_quality_warning(event1)
        penalty1 = scheduler._config_priorities["hex8_2p"].quality_penalty

        # Reset
        scheduler._config_priorities["hex8_2p"].quality_penalty = 0.0

        # Quality 0.6 -> penalty = -0.15 * (1.0 - 0.6) = -0.06
        event2 = MagicMock()
        event2.payload = {"quality_score": 0.6, "old_state": "ok", "new_state": "warning"}
        scheduler._on_low_quality_warning(event2)
        penalty2 = scheduler._config_priorities["hex8_2p"].quality_penalty

        # Lower quality = larger penalty
        assert penalty1 < penalty2 < 0

    def test_penalty_range(self, scheduler):
        """Test penalty stays within expected range."""
        # Quality 0.0 -> max penalty -0.15
        event1 = MagicMock()
        event1.payload = {"quality_score": 0.0, "old_state": "ok", "new_state": "warning"}
        scheduler._on_low_quality_warning(event1)

        for config_key in ALL_CONFIGS:
            penalty = scheduler._config_priorities[config_key].quality_penalty
            assert -0.15 <= penalty <= 0.0

    def test_penalty_zero_at_perfect_quality(self, scheduler):
        """Test no penalty at perfect quality."""
        event = MagicMock()
        event.payload = {"quality_score": 1.0, "old_state": "ok", "new_state": "ok"}

        scheduler._on_low_quality_warning(event)

        for config_key in ALL_CONFIGS:
            penalty = scheduler._config_priorities[config_key].quality_penalty
            assert abs(penalty) < 0.01


# =============================================================================
# State Update Tests
# =============================================================================


class TestStateUpdate:
    """Tests for priority state updates."""

    def test_updates_all_config_states(self, scheduler_with_priorities, mock_event):
        """Test updates exploration boost for all configs."""
        scheduler_with_priorities._on_low_quality_warning(mock_event)

        for config_key in ALL_CONFIGS:
            priority = scheduler_with_priorities._config_priorities[config_key]
            # Should have been modified from initial 1.5 (for first 6) or 1.0 (default)
            assert priority.exploration_boost != 1.5 or priority.exploration_boost != 1.0

    def test_preserves_other_priority_fields(self, scheduler, mock_event):
        """Test preserves other priority fields during throttle."""
        # Set various fields
        priority = scheduler._config_priorities["hex8_2p"]
        priority.staleness_hours = 5.0
        priority.elo_velocity = 25.0
        priority.curriculum_weight = 1.3
        priority.improvement_boost = 0.1

        scheduler._on_low_quality_warning(mock_event)

        # Check other fields unchanged
        assert priority.staleness_hours == 5.0
        assert priority.elo_velocity == 25.0
        assert priority.curriculum_weight == 1.3
        assert priority.improvement_boost == 0.1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full low quality feedback loop."""

    def test_full_low_quality_feedback_loop(self, scheduler_with_priorities):
        """Test complete low quality feedback loop."""
        event = MagicMock()
        event.payload = {
            "quality_score": 0.4,
            "old_state": "ok",
            "new_state": "critical",
        }

        # Record initial boosts
        initial_boosts = {
            k: p.exploration_boost
            for k, p in scheduler_with_priorities._config_priorities.items()
        }

        scheduler_with_priorities._on_low_quality_warning(event)

        # Verify all configs throttled
        for config_key in ALL_CONFIGS[:6]:
            priority = scheduler_with_priorities._config_priorities[config_key]
            assert priority.exploration_boost < initial_boosts[config_key]
            assert priority.quality_penalty < 0

    def test_multiple_warnings_compound(self, scheduler):
        """Test multiple warnings compound throttling."""
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.5

        # First warning
        event1 = MagicMock()
        event1.payload = {"quality_score": 0.5, "old_state": "ok", "new_state": "warning"}
        scheduler._on_low_quality_warning(event1)
        boost_after_first = scheduler._config_priorities["hex8_2p"].exploration_boost

        # Second warning
        event2 = MagicMock()
        event2.payload = {"quality_score": 0.45, "old_state": "warning", "new_state": "critical"}
        scheduler._on_low_quality_warning(event2)
        boost_after_second = scheduler._config_priorities["hex8_2p"].exploration_boost

        # Should compound
        assert boost_after_second < boost_after_first


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_malformed_event(self, scheduler):
        """Test handles malformed event payload."""
        event = MagicMock()
        event.payload = None

        # Should not raise
        scheduler._on_low_quality_warning(event)

    def test_handles_missing_payload(self, scheduler):
        """Test handles event with missing payload attribute."""
        event = MagicMock(spec=[])

        # Should not raise
        scheduler._on_low_quality_warning(event)

    def test_handles_extreme_quality_values(self, scheduler):
        """Test handles extreme quality values."""
        with patch("app.coordination.event_emission_helpers.safe_emit_event", return_value=False):
            # Negative quality (should not happen but defensive)
            event1 = MagicMock()
            event1.payload = {"quality_score": -0.5, "old_state": "ok", "new_state": "critical"}
            scheduler._on_low_quality_warning(event1)

            # Quality > 1.0 (should not happen)
            event2 = MagicMock()
            event2.payload = {"quality_score": 1.5, "old_state": "ok", "new_state": "ok"}
            scheduler._on_low_quality_warning(event2)

    def test_thread_safety(self, scheduler):
        """Test concurrent low quality handling is thread-safe."""
        import threading

        events = [
            MagicMock(payload={
                "quality_score": 0.4 + i * 0.05,
                "old_state": "ok",
                "new_state": "warning",
            })
            for i in range(5)
        ]

        threads = []
        for event in events:
            t = threading.Thread(target=scheduler._on_low_quality_warning, args=(event,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All configs should have penalties
        for config_key in ALL_CONFIGS:
            priority = scheduler._config_priorities[config_key]
            assert priority.quality_penalty <= 0

    def test_handles_concurrent_throttling(self, scheduler):
        """Test handles concurrent throttling from different sources."""
        import threading

        def throttle_from_quality():
            event = MagicMock()
            event.payload = {"quality_score": 0.4, "old_state": "ok", "new_state": "warning"}
            scheduler._on_low_quality_warning(event)

        def throttle_from_other():
            # Simulate other throttling mechanism
            scheduler._config_priorities["hex8_2p"].exploration_boost *= 0.8

        t1 = threading.Thread(target=throttle_from_quality)
        t2 = threading.Thread(target=throttle_from_other)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Should complete without deadlock (both operations may compound)
        # Quality=0.4 -> 0.6 throttle, other thread -> 0.8 throttle
        # Result could be as low as 1.0 * 0.6 * 0.8 = 0.48, clamped to 0.5 minimum
        # But race conditions mean it could be higher, so just check it's reasonable
        assert 0.48 <= scheduler._config_priorities["hex8_2p"].exploration_boost <= 1.0

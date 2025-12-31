"""Tests for curriculum_integration.py module.

December 2025: Added as part of test coverage initiative.
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.curriculum_integration import (
    MomentumToCurriculumBridge,
    PFSPWeaknessWatcher,
    QualityPenaltyToCurriculumWatcher,
    QualityToTemperatureWatcher,
    wire_all_feedback_loops,
    unwire_all_feedback_loops,
    get_integration_status,
    get_exploration_boost,
    get_mastered_opponents,
    force_momentum_sync,
    get_quality_penalty_weights,
    reset_quality_penalty,
    get_promotion_failure_counts,
    reset_promotion_failure_count,
)


# =============================================================================
# MomentumToCurriculumBridge Tests
# =============================================================================


class TestMomentumToCurriculumBridge:
    """Tests for MomentumToCurriculumBridge class."""

    def test_init_defaults(self):
        """Bridge initializes with default values."""
        bridge = MomentumToCurriculumBridge()
        assert bridge.poll_interval_seconds == 10.0
        assert bridge.momentum_weight_boost == 0.3
        assert bridge._running is False
        assert bridge._event_subscribed is False

    def test_init_custom(self):
        """Bridge accepts custom parameters."""
        bridge = MomentumToCurriculumBridge(
            poll_interval_seconds=30.0,
            momentum_weight_boost=0.5,
        )
        assert bridge.poll_interval_seconds == 30.0
        assert bridge.momentum_weight_boost == 0.5

    def test_stop_when_not_running(self):
        """Stop is safe when not running."""
        bridge = MomentumToCurriculumBridge()
        bridge.stop()  # Should not raise

    def test_start_sets_running_flag(self):
        """Start sets the running flag."""
        bridge = MomentumToCurriculumBridge()

        # Mock event subscription to avoid actual event bus
        with patch.object(bridge, '_subscribe_to_events', return_value=True):
            bridge.start()
            assert bridge._running is True
            bridge.stop()

    def test_force_sync_returns_dict(self):
        """force_sync returns a dict (may contain weights from accelerator)."""
        bridge = MomentumToCurriculumBridge()
        result = bridge.force_sync()
        assert isinstance(result, dict)
        # Values should be floats if present
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, float)


# =============================================================================
# PFSPWeaknessWatcher Tests
# =============================================================================


class TestPFSPWeaknessWatcher:
    """Tests for PFSPWeaknessWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        watcher = PFSPWeaknessWatcher()
        assert watcher.MASTERY_THRESHOLD == 0.85
        assert watcher.MIN_GAMES_FOR_MASTERY == 20
        assert watcher.CHECK_INTERVAL == 120.0
        assert watcher._running is False
        assert len(watcher._mastered_matchups) == 0

    def test_get_mastered_matchups_empty(self):
        """get_mastered_matchups returns empty list initially."""
        watcher = PFSPWeaknessWatcher()
        assert watcher.get_mastered_matchups() == []

    def test_stop_when_not_running(self):
        """Stop is safe when not running."""
        watcher = PFSPWeaknessWatcher()
        watcher.stop()  # Should not raise

    def test_extract_config_standard(self):
        """_extract_config parses standard model IDs."""
        watcher = PFSPWeaknessWatcher()

        assert watcher._extract_config("hex8_2p_v123") == "hex8_2p"
        assert watcher._extract_config("square8_4p_v1") == "square8_4p"
        assert watcher._extract_config("canonical_hex8_2p") == "hex8_2p"

    def test_extract_config_fallback(self):
        """_extract_config falls back to first two parts."""
        watcher = PFSPWeaknessWatcher()

        assert watcher._extract_config("model_unknown") == "model_unknown"
        assert watcher._extract_config("simple") == "simple"


# =============================================================================
# QualityPenaltyToCurriculumWatcher Tests
# =============================================================================


class TestQualityPenaltyToCurriculumWatcher:
    """Tests for QualityPenaltyToCurriculumWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        watcher = QualityPenaltyToCurriculumWatcher()
        assert watcher.WEIGHT_REDUCTION_PER_PENALTY == 0.15
        assert watcher._subscribed is False
        # Dec 2025: Uses base class _state dict via get_penalty_weights()
        assert len(watcher.get_penalty_weights()) == 0

    def test_get_penalty_weights_empty(self):
        """get_penalty_weights returns empty dict initially."""
        watcher = QualityPenaltyToCurriculumWatcher()
        assert watcher.get_penalty_weights() == {}

    def test_reset_penalty(self):
        """reset_penalty removes weight for config."""
        watcher = QualityPenaltyToCurriculumWatcher()
        # Dec 2025: Use _compute_weight_multiplier to populate state
        watcher._compute_weight_multiplier("hex8_2p", {"new_penalty": 1.0})
        assert "hex8_2p" in watcher.get_penalty_weights()

        watcher.reset_penalty("hex8_2p")
        assert "hex8_2p" not in watcher.get_penalty_weights()

    def test_reset_penalty_nonexistent(self):
        """reset_penalty is safe for nonexistent config."""
        watcher = QualityPenaltyToCurriculumWatcher()
        watcher.reset_penalty("nonexistent")  # Should not raise


# =============================================================================
# QualityToTemperatureWatcher Tests
# =============================================================================


class TestQualityToTemperatureWatcher:
    """Tests for QualityToTemperatureWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        watcher = QualityToTemperatureWatcher()
        assert watcher.EXPLORATION_BOOST_FACTOR == 1.3
        assert watcher._subscribed is False
        assert len(watcher._quality_boosts) == 0

    def test_low_quality_threshold_property(self):
        """LOW_QUALITY_THRESHOLD is a property with fallback."""
        watcher = QualityToTemperatureWatcher()
        threshold = watcher.LOW_QUALITY_THRESHOLD
        # Should be a float between 0 and 1
        assert isinstance(threshold, float)
        assert 0 < threshold < 1

    def test_get_exploration_boost_default(self):
        """get_exploration_boost returns 1.0 for unknown config."""
        watcher = QualityToTemperatureWatcher()
        assert watcher.get_exploration_boost("unknown_config") == 1.0

    def test_get_exploration_boost_tracked(self):
        """get_exploration_boost returns tracked value."""
        watcher = QualityToTemperatureWatcher()
        watcher._quality_boosts["hex8_2p"] = 1.5

        assert watcher.get_exploration_boost("hex8_2p") == 1.5

    def test_get_all_boosts_empty(self):
        """get_all_boosts returns empty dict initially."""
        watcher = QualityToTemperatureWatcher()
        assert watcher.get_all_boosts() == {}

    def test_get_all_boosts_copy(self):
        """get_all_boosts returns a copy."""
        watcher = QualityToTemperatureWatcher()
        watcher._quality_boosts["hex8_2p"] = 1.3

        boosts = watcher.get_all_boosts()
        boosts["hex8_2p"] = 2.0  # Modify the copy

        # Original should be unchanged
        assert watcher._quality_boosts["hex8_2p"] == 1.3


# =============================================================================
# Wiring Function Tests
# =============================================================================


class TestWiringFunctions:
    """Tests for wire/unwire functions."""

    def test_wire_already_active(self):
        """wire_all_feedback_loops returns early if already active."""
        # First call
        with patch('app.coordination.curriculum_integration._integration_active', True):
            result = wire_all_feedback_loops()
            assert result["status"] == "already_active"

    def test_unwire_clears_state(self):
        """unwire_all_feedback_loops clears integration state."""
        # Ensure clean state after unwire
        unwire_all_feedback_loops()

        status = get_integration_status()
        assert status["active"] is False
        assert status["watchers"] == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_exploration_boost_no_watcher(self):
        """get_exploration_boost returns 1.0 when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_exploration_boost("any_config") == 1.0

    def test_get_mastered_opponents_empty(self):
        """get_mastered_opponents returns empty list when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_mastered_opponents() == []

    def test_force_momentum_sync_empty(self):
        """force_momentum_sync returns empty dict when bridge not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert force_momentum_sync() == {}

    def test_get_quality_penalty_weights_empty(self):
        """get_quality_penalty_weights returns empty dict when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_quality_penalty_weights() == {}

    def test_reset_quality_penalty_no_watcher(self):
        """reset_quality_penalty is safe when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        reset_quality_penalty("any_config")  # Should not raise

    def test_get_promotion_failure_counts_empty(self):
        """get_promotion_failure_counts returns empty dict when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_promotion_failure_counts() == {}

    def test_reset_promotion_failure_count_no_watcher(self):
        """reset_promotion_failure_count is safe when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        reset_promotion_failure_count("any_config")  # Should not raise


# =============================================================================
# Integration Status Tests
# =============================================================================


class TestIntegrationStatus:
    """Tests for get_integration_status function."""

    def test_status_inactive(self):
        """Status shows inactive when not wired."""
        unwire_all_feedback_loops()  # Ensure clean state

        status = get_integration_status()
        assert status["active"] is False
        assert isinstance(status["watchers"], list)

    def test_status_structure(self):
        """Status has expected structure."""
        status = get_integration_status()

        assert "active" in status
        assert "watchers" in status
        assert isinstance(status["active"], bool)
        assert isinstance(status["watchers"], list)


# =============================================================================
# Event Handler Tests (December 29, 2025)
# =============================================================================


class TestCurriculumAdvancementHandler:
    """Tests for _on_curriculum_advancement_needed handler.

    December 29, 2025: Tests for the handler that closes the curriculum feedback loop.
    When a config stagnates (3+ evaluations with minimal Elo improvement),
    TrainingTriggerDaemon emits CURRICULUM_ADVANCEMENT_NEEDED.
    """

    def test_handler_with_valid_event(self):
        """Handler processes valid event with config_key."""
        bridge = MomentumToCurriculumBridge()

        # Create mock event with proper structure
        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        # Mock dependencies to avoid side effects
        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(mock_event)
            # Should not raise

    def test_handler_with_dict_event(self):
        """Handler accepts dict directly (without .payload attribute)."""
        bridge = MomentumToCurriculumBridge()

        # Event can be a dict directly
        event = {
            "config_key": "square8_4p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(event)
            # Should not raise

    def test_handler_skips_empty_config_key(self):
        """Handler returns early when config_key is missing."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {"reason": "elo_plateau"}

        # Should not call _sync_weights for empty config
        with patch.object(bridge, '_sync_weights') as mock_sync:
            bridge._on_curriculum_advancement_needed(mock_event)
            mock_sync.assert_not_called()

    def test_handler_updates_last_sync_time(self):
        """Handler updates _last_sync_time on successful processing."""
        bridge = MomentumToCurriculumBridge()
        initial_time = bridge._last_sync_time

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hexagonal_3p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(mock_event)
            assert bridge._last_sync_time > initial_time

    def test_handler_calls_sync_weights(self):
        """Handler calls _sync_weights to propagate curriculum changes."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "square19_2p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        with patch.object(bridge, '_sync_weights') as mock_sync:
            bridge._on_curriculum_advancement_needed(mock_event)
            mock_sync.assert_called_once()

    def test_handler_with_config_alias(self):
        """Handler accepts 'config' as alias for 'config_key'."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config": "hex8_4p",  # Uses 'config' instead of 'config_key'
            "reason": "stagnation",
        }

        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(mock_event)
            # Should not raise

    def test_handler_graceful_failure_on_curriculum_import_error(self):
        """Handler handles ImportError when curriculum_feedback not available."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "reason": "elo_plateau",
        }

        # Patch the curriculum import to raise ImportError
        # The import happens inside the handler, so we patch the module path
        with patch.object(bridge, '_sync_weights'):
            with patch.dict('sys.modules', {'app.training.curriculum_feedback': None}):
                # Should handle gracefully without raising
                bridge._on_curriculum_advancement_needed(mock_event)

    def test_handler_graceful_failure_on_attribute_error(self):
        """Handler handles AttributeError from malformed event."""
        bridge = MomentumToCurriculumBridge()

        # Create event that will cause AttributeError
        mock_event = None  # This will cause .payload access to fail

        # Should not raise, just log warning
        bridge._on_curriculum_advancement_needed(mock_event)

    def test_curriculum_advancement_needed_event_exists(self):
        """CURRICULUM_ADVANCEMENT_NEEDED event type exists in DataEventType."""
        from app.distributed.data_events import DataEventType

        assert hasattr(DataEventType, 'CURRICULUM_ADVANCEMENT_NEEDED')
        assert DataEventType.CURRICULUM_ADVANCEMENT_NEEDED.value == "curriculum_advancement_needed"

    def test_curriculum_advanced_event_exists(self):
        """CURRICULUM_ADVANCED event type exists in DataEventType."""
        from app.distributed.data_events import DataEventType

        assert hasattr(DataEventType, 'CURRICULUM_ADVANCED')
        assert DataEventType.CURRICULUM_ADVANCED.value == "curriculum_advanced"


# =============================================================================
# PromotionFailedToCurriculumWatcher Tests
# =============================================================================


class TestPromotionFailedToCurriculumWatcher:
    """Tests for PromotionFailedToCurriculumWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher

        watcher = PromotionFailedToCurriculumWatcher()
        assert watcher.WEIGHT_INCREASE_PER_FAILURE == 0.20
        assert watcher._subscribed is False
        # Dec 2025: Uses base class _state dict via get_failure_counts()
        assert len(watcher.get_failure_counts()) == 0

    def test_get_failure_counts_empty(self):
        """get_failure_counts returns empty dict initially."""
        from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher

        watcher = PromotionFailedToCurriculumWatcher()
        assert watcher.get_failure_counts() == {}

    def test_reset_failure_count_nonexistent(self):
        """reset_failure_count is safe for nonexistent config."""
        from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher

        watcher = PromotionFailedToCurriculumWatcher()
        watcher.reset_failure_count("nonexistent_config")  # Should not raise

    def test_subscribe_returns_bool(self):
        """subscribe returns a boolean."""
        from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher

        watcher = PromotionFailedToCurriculumWatcher()
        # Will likely return False since event router may not be fully initialized
        result = watcher.subscribe()
        assert isinstance(result, bool)

    def test_health_check_returns_result(self):
        """health_check returns a HealthCheckResult."""
        from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher
        from app.coordination.contracts import HealthCheckResult

        watcher = PromotionFailedToCurriculumWatcher()
        health = watcher.health_check()

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")
        assert hasattr(health, "status")


# =============================================================================
# PromotionCompletedToCurriculumWatcher Tests
# =============================================================================


class TestPromotionCompletedToCurriculumWatcher:
    """Tests for PromotionCompletedToCurriculumWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        from app.coordination.curriculum_integration import PromotionCompletedToCurriculumWatcher

        watcher = PromotionCompletedToCurriculumWatcher()
        assert watcher.WEIGHT_REDUCTION_PER_REGRESSION == 0.15
        assert watcher.CONSECUTIVE_FAILURE_THRESHOLD == 3
        assert watcher.WEIGHT_BOOST_ON_SUCCESS == 0.10
        assert watcher._subscribed is False
        assert len(watcher._success_streak) == 0

    def test_get_success_streaks_empty(self):
        """get_success_streaks returns empty dict initially."""
        from app.coordination.curriculum_integration import PromotionCompletedToCurriculumWatcher

        watcher = PromotionCompletedToCurriculumWatcher()
        assert watcher.get_success_streaks() == {}

    def test_subscribe_returns_bool(self):
        """subscribe returns a boolean."""
        from app.coordination.curriculum_integration import PromotionCompletedToCurriculumWatcher

        watcher = PromotionCompletedToCurriculumWatcher()
        # Will likely return False since event router may not be fully initialized
        result = watcher.subscribe()
        assert isinstance(result, bool)

    def test_health_check_returns_result(self):
        """health_check returns a HealthCheckResult."""
        from app.coordination.curriculum_integration import PromotionCompletedToCurriculumWatcher
        from app.coordination.contracts import HealthCheckResult

        watcher = PromotionCompletedToCurriculumWatcher()
        health = watcher.health_check()

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")


# =============================================================================
# RegressionCriticalToCurriculumWatcher Tests
# =============================================================================


class TestRegressionCriticalToCurriculumWatcher:
    """Tests for RegressionCriticalToCurriculumWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        from app.coordination.curriculum_integration import RegressionCriticalToCurriculumWatcher

        watcher = RegressionCriticalToCurriculumWatcher()
        assert watcher.WEIGHT_INCREASE_MODERATE == 0.25
        assert watcher.WEIGHT_INCREASE_SEVERE == 0.50
        assert watcher._subscribed is False
        # Dec 2025: Uses base class _state dict via get_regression_counts()
        assert len(watcher.get_regression_counts()) == 0

    def test_get_regression_counts_empty(self):
        """get_regression_counts returns empty dict initially."""
        from app.coordination.curriculum_integration import RegressionCriticalToCurriculumWatcher

        watcher = RegressionCriticalToCurriculumWatcher()
        assert watcher.get_regression_counts() == {}

    def test_reset_regression_count_nonexistent(self):
        """reset_regression_count is safe for nonexistent config."""
        from app.coordination.curriculum_integration import RegressionCriticalToCurriculumWatcher

        watcher = RegressionCriticalToCurriculumWatcher()
        watcher.reset_regression_count("nonexistent_config")  # Should not raise

    def test_subscribe_returns_bool(self):
        """subscribe returns a boolean."""
        from app.coordination.curriculum_integration import RegressionCriticalToCurriculumWatcher

        watcher = RegressionCriticalToCurriculumWatcher()
        # Will likely return False since event router may not be fully initialized
        result = watcher.subscribe()
        assert isinstance(result, bool)

    def test_health_check_returns_result(self):
        """health_check returns a HealthCheckResult."""
        from app.coordination.curriculum_integration import RegressionCriticalToCurriculumWatcher
        from app.coordination.contracts import HealthCheckResult

        watcher = RegressionCriticalToCurriculumWatcher()
        health = watcher.health_check()

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")


# =============================================================================
# Additional Convenience Function Tests
# =============================================================================


class TestAdditionalConvenienceFunctions:
    """Tests for additional convenience functions."""

    def test_get_promotion_success_streaks_empty(self):
        """get_promotion_success_streaks returns empty when no watcher."""
        from app.coordination.curriculum_integration import get_promotion_success_streaks

        result = get_promotion_success_streaks()
        assert isinstance(result, dict)

    def test_get_regression_critical_counts_empty(self):
        """get_regression_critical_counts returns empty when no watcher."""
        from app.coordination.curriculum_integration import get_regression_critical_counts

        result = get_regression_critical_counts()
        assert isinstance(result, dict)

    def test_reset_regression_critical_count_no_watcher(self):
        """reset_regression_critical_count is safe when no watcher."""
        from app.coordination.curriculum_integration import reset_regression_critical_count

        reset_regression_critical_count("hex8_2p")  # Should not raise


# =============================================================================
# MomentumToCurriculumBridge Event Handler Tests
# =============================================================================


class TestMomentumBridgeEventHandlers:
    """Tests for MomentumToCurriculumBridge event handlers."""

    def test_on_evaluation_completed(self):
        """_on_evaluation_completed handles valid event."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "win_rate": 0.75,
            "elo_change": 50.0,
        }

        # Should not raise
        bridge._on_evaluation_completed(mock_event)

    def test_on_selfplay_rate_changed(self):
        """_on_selfplay_rate_changed handles valid event."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "old_rate": 0.1,
            "new_rate": 0.2,
        }

        # Should not raise
        bridge._on_selfplay_rate_changed(mock_event)

    def test_on_elo_significant_change(self):
        """_on_elo_significant_change handles valid event."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "old_elo": 1200,
            "new_elo": 1350,
        }

        # Should not raise
        bridge._on_elo_significant_change(mock_event)

    def test_on_selfplay_allocation_updated(self):
        """_on_selfplay_allocation_updated handles valid event."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "allocations": {"hex8_2p": 100, "square8_2p": 50},
        }

        # Should not raise
        bridge._on_selfplay_allocation_updated(mock_event)

    def test_on_model_promoted(self):
        """_on_model_promoted handles valid event."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "model_path": "/path/to/model.pth",
            "elo_gain": 75.0,
        }

        # Should not raise
        bridge._on_model_promoted(mock_event)


# =============================================================================
# Health Check Tests for All Watchers
# =============================================================================


class TestWatcherHealthChecks:
    """Tests for health_check methods across all watchers."""

    def test_momentum_bridge_health_check(self):
        """MomentumToCurriculumBridge health_check."""
        bridge = MomentumToCurriculumBridge()
        health = bridge.health_check()

        from app.coordination.contracts import HealthCheckResult

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")
        assert hasattr(health, "details")
        assert isinstance(health.details, dict)

    def test_pfsp_watcher_health_check(self):
        """PFSPWeaknessWatcher health_check."""
        watcher = PFSPWeaknessWatcher()
        health = watcher.health_check()

        from app.coordination.contracts import HealthCheckResult

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")

    def test_quality_penalty_watcher_health_check(self):
        """QualityPenaltyToCurriculumWatcher health_check."""
        watcher = QualityPenaltyToCurriculumWatcher()
        health = watcher.health_check()

        from app.coordination.contracts import HealthCheckResult

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")

    def test_quality_temperature_watcher_health_check(self):
        """QualityToTemperatureWatcher health_check."""
        watcher = QualityToTemperatureWatcher()
        health = watcher.health_check()

        from app.coordination.contracts import HealthCheckResult

        assert isinstance(health, HealthCheckResult)
        assert hasattr(health, "healthy")

"""Extended tests for RollbackManager.

Tests untested components:
- QualityRollbackWatcher
- AutoRollbackHandler event bus methods
- Wiring functions (wire_regression_to_rollback, wire_quality_to_rollback)
- Singleton getters
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.training.rollback_manager import (
    AutoRollbackHandler,
    QualityRollbackWatcher,
    RollbackEvent,
    RollbackManager,
    RollbackThresholds,
    get_auto_rollback_handler,
    get_quality_rollback_watcher,
    wire_quality_to_rollback,
    wire_regression_to_rollback,
)


class TestRollbackEventExtended:
    """Extended tests for RollbackEvent."""

    def test_event_with_error(self):
        """Test event with failure status."""
        event = RollbackEvent(
            model_id="test_model",
            from_version=2,
            to_version=1,
            reason="Test failure",
            triggered_by="manual",
            timestamp="2024-01-01T12:00:00",
            success=False,
            error_message="Rollback failed: model not found",
        )

        d = event.to_dict()
        assert d["success"] is False
        assert d["error_message"] == "Rollback failed: model not found"

        restored = RollbackEvent.from_dict(d)
        assert not restored.success
        assert restored.error_message == "Rollback failed: model not found"

    def test_event_default_values(self):
        """Test event with minimal required fields."""
        event = RollbackEvent(
            model_id="test",
            from_version=1,
            to_version=0,
            reason="Test",
            triggered_by="auto",
            timestamp="2024-01-01",
        )

        assert event.from_metrics == {}
        assert event.to_metrics == {}
        assert event.success is True
        assert event.error_message is None


class TestRollbackThresholdsExtended:
    """Extended tests for RollbackThresholds."""

    def test_all_threshold_fields(self):
        """Test all threshold fields can be customized."""
        thresholds = RollbackThresholds(
            elo_drop_threshold=100.0,
            elo_drop_window_hours=48.0,
            win_rate_drop_threshold=0.20,
            error_rate_threshold=0.10,
            min_games_for_evaluation=100,
        )

        assert thresholds.elo_drop_threshold == 100.0
        assert thresholds.elo_drop_window_hours == 48.0
        assert thresholds.win_rate_drop_threshold == 0.20
        assert thresholds.error_rate_threshold == 0.10
        assert thresholds.min_games_for_evaluation == 100


class TestRollbackManagerHistory:
    """Test rollback history persistence."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.history_path = Path(self.temp_dir) / "rollback_history.json"
        self.registry = MagicMock()

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_history_loads_on_init(self):
        """Test history is loaded from disk on initialization."""
        # Create history file
        history_data = [
            {
                "model_id": "test",
                "from_version": 2,
                "to_version": 1,
                "reason": "Test",
                "triggered_by": "manual",
                "timestamp": "2024-01-01",
                "from_metrics": {},
                "to_metrics": {},
                "success": True,
                "error_message": None,
            }
        ]
        with open(self.history_path, "w") as f:
            json.dump(history_data, f)

        manager = RollbackManager(
            self.registry,
            history_path=self.history_path,
        )

        history = manager.get_rollback_history()
        assert len(history) == 1
        assert history[0].model_id == "test"

    def test_history_handles_missing_file(self):
        """Test graceful handling of missing history file."""
        manager = RollbackManager(
            self.registry,
            history_path=Path(self.temp_dir) / "nonexistent.json",
        )

        history = manager.get_rollback_history()
        assert len(history) == 0

    def test_history_handles_corrupt_file(self):
        """Test graceful handling of corrupt history file."""
        corrupt_path = Path(self.temp_dir) / "corrupt.json"
        corrupt_path.write_text("not valid json {{{")

        manager = RollbackManager(
            self.registry,
            history_path=corrupt_path,
        )

        history = manager.get_rollback_history()
        assert len(history) == 0


class TestRollbackManagerMetrics:
    """Test rollback manager Prometheus metrics integration."""

    def test_metrics_counter_exists(self):
        """Test that metrics are tracked."""
        registry = MagicMock()
        manager = RollbackManager(registry)

        # Metrics should be available
        stats = manager.get_rollback_stats()
        assert "total_rollbacks" in stats
        assert "by_trigger" in stats


class TestAutoRollbackHandlerExtended:
    """Extended tests for AutoRollbackHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = MagicMock()
        self.rollback_manager = RollbackManager(
            self.registry,
            history_path=Path(self.temp_dir) / "history.json",
        )

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_handler_initialization_defaults(self):
        """Test default initialization values."""
        handler = AutoRollbackHandler(self.rollback_manager)

        assert handler.auto_rollback_enabled is True
        assert handler.require_approval_for_severe is True
        assert handler._pending_rollbacks == {}

    def test_handler_initialization_custom(self):
        """Test custom initialization values."""
        handler = AutoRollbackHandler(
            self.rollback_manager,
            auto_rollback_enabled=False,
            require_approval_for_severe=False,
        )

        assert handler.auto_rollback_enabled is False
        assert handler.require_approval_for_severe is False

    def test_pending_rollbacks_tracking(self):
        """Test that SEVERE regressions are added to pending rollbacks."""
        from app.training.regression_detector import RegressionSeverity

        handler = AutoRollbackHandler(
            self.rollback_manager,
            require_approval_for_severe=True,
        )

        # Create a mock SEVERE regression event
        mock_event = MagicMock()
        mock_event.model_id = "test_model"
        mock_event.reason = "Elo dropped 45 points"
        mock_event.timestamp = time.time()
        mock_event.to_dict.return_value = {"model_id": "test_model", "reason": "Test"}
        mock_event.severity = RegressionSeverity.SEVERE

        handler.on_regression(mock_event)

        assert "test_model" in handler._pending_rollbacks
        assert handler._pending_rollbacks["test_model"]["reason"] == "Elo dropped 45 points"

    def test_disabled_handler_logs_only(self):
        """Test that disabled handler doesn't execute rollbacks."""
        from app.training.regression_detector import RegressionSeverity

        handler = AutoRollbackHandler(
            self.rollback_manager,
            auto_rollback_enabled=False,
        )

        mock_event = MagicMock()
        mock_event.model_id = "test_model"
        mock_event.reason = "Test"
        mock_event.timestamp = time.time()
        mock_event.severity = RegressionSeverity.CRITICAL

        # Should not execute rollback
        handler.on_regression(mock_event)

        # No pending rollbacks since handler is disabled
        assert "test_model" not in handler._pending_rollbacks

    def test_exploration_boost_factors(self):
        """Test exploration boost factor calculation."""
        handler = AutoRollbackHandler(self.rollback_manager)

        assert handler._get_exploration_boost_factor("critical") == 2.0
        assert handler._get_exploration_boost_factor("major") == 1.75
        assert handler._get_exploration_boost_factor("moderate") == 1.5
        assert handler._get_exploration_boost_factor("minor") == 1.25
        assert handler._get_exploration_boost_factor("unknown") == 1.5

    @patch("app.coordination.event_router.get_event_bus")
    def test_subscribe_to_events_success(self, mock_get_bus):
        """Test successful event subscription."""
        mock_bus = MagicMock()
        mock_get_bus.return_value = mock_bus

        handler = AutoRollbackHandler(self.rollback_manager)
        result = handler.subscribe_to_regression_events()

        assert result is True
        assert handler._event_subscribed is True
        assert mock_bus.subscribe.call_count >= 3

    @patch("app.coordination.event_router.get_event_bus")
    def test_unsubscribe_from_events(self, mock_get_bus):
        """Test event unsubscription."""
        mock_bus = MagicMock()
        mock_get_bus.return_value = mock_bus

        handler = AutoRollbackHandler(self.rollback_manager)
        handler._event_subscribed = True

        handler.unsubscribe_from_regression_events()

        assert mock_bus.unsubscribe.call_count >= 3
        assert handler._event_subscribed is False

    def test_unsubscribe_when_not_subscribed(self):
        """Test unsubscribe is no-op when not subscribed."""
        handler = AutoRollbackHandler(self.rollback_manager)
        handler._event_subscribed = False

        # Should not raise
        handler.unsubscribe_from_regression_events()


class TestQualityRollbackWatcher:
    """Test QualityRollbackWatcher class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = MagicMock()
        self.rollback_manager = RollbackManager(
            self.registry,
            history_path=Path(self.temp_dir) / "history.json",
        )

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_watcher_initialization_defaults(self):
        """Test default initialization values."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        assert watcher.critical_threshold == 0.3
        assert watcher.sustained_minutes == 30
        assert watcher._subscribed is False
        assert watcher._rollbacks_triggered == 0
        assert watcher._low_quality_start == {}

    def test_watcher_initialization_custom(self):
        """Test custom initialization values."""
        watcher = QualityRollbackWatcher(
            self.rollback_manager,
            critical_threshold=0.2,
            sustained_minutes=60.0,
        )

        assert watcher.critical_threshold == 0.2
        assert watcher.sustained_minutes == 60.0

    @patch("app.coordination.event_router.get_event_bus")
    def test_subscribe_success(self, mock_get_bus):
        """Test successful event subscription."""
        mock_bus = MagicMock()
        mock_get_bus.return_value = mock_bus

        watcher = QualityRollbackWatcher(self.rollback_manager)
        result = watcher.subscribe_to_quality_events()

        assert result is True
        assert watcher._subscribed is True
        assert mock_bus.subscribe.call_count == 2

    @patch("app.coordination.event_router.get_event_bus")
    def test_subscribe_fails_no_bus(self, mock_get_bus):
        """Test subscription fails when bus is None."""
        mock_get_bus.return_value = None

        watcher = QualityRollbackWatcher(self.rollback_manager)
        result = watcher.subscribe_to_quality_events()

        assert result is False
        assert watcher._subscribed is False

    def test_on_low_quality_starts_tracking(self):
        """Test that low quality event starts tracking."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        event = {"quality_score": 0.2, "config_key": "hex8_2p", "timestamp": time.time()}
        watcher._on_low_quality(event)

        assert "hex8_2p" in watcher._low_quality_start

    def test_on_low_quality_dict_event(self):
        """Test handling dict-type event."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        event = {"quality_score": 0.25, "config_key": "square8_2p", "timestamp": time.time()}
        watcher._on_low_quality(event)

        assert "square8_2p" in watcher._low_quality_start

    def test_on_low_quality_payload_event(self):
        """Test handling event with payload attribute."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        mock_event = MagicMock()
        mock_event.payload = {"quality_score": 0.15, "config_key": "hex8_4p", "timestamp": time.time()}

        watcher._on_low_quality(mock_event)

        assert "hex8_4p" in watcher._low_quality_start

    def test_on_low_quality_above_threshold_no_tracking(self):
        """Test that quality above threshold doesn't start tracking."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        event = {"quality_score": 0.5, "config_key": "hex8_2p", "timestamp": time.time()}
        watcher._on_low_quality(event)

        assert "hex8_2p" not in watcher._low_quality_start

    def test_on_quality_recovered_clears_tracking(self):
        """Test that quality recovery clears tracking."""
        watcher = QualityRollbackWatcher(self.rollback_manager)
        watcher._low_quality_start["hex8_2p"] = time.time() - 100

        event = {"config_key": "hex8_2p"}
        watcher._on_quality_recovered(event)

        assert "hex8_2p" not in watcher._low_quality_start

    def test_on_quality_recovered_handles_missing_config(self):
        """Test recovery handles config not being tracked."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        event = {"config_key": "nonexistent"}
        # Should not raise
        watcher._on_quality_recovered(event)

    def test_sustained_low_quality_triggers_rollback(self):
        """Test that sustained low quality triggers rollback."""
        watcher = QualityRollbackWatcher(
            self.rollback_manager,
            sustained_minutes=0.01,  # 0.6 seconds for testing
        )

        # Mock rollback execution
        with patch.object(watcher, '_trigger_quality_rollback') as mock_trigger:
            # First event starts tracking
            start_time = time.time()
            event1 = {"quality_score": 0.1, "config_key": "hex8_2p", "timestamp": start_time}
            watcher._on_low_quality(event1)

            # Second event after sustained period
            event2 = {"quality_score": 0.1, "config_key": "hex8_2p", "timestamp": start_time + 2}
            watcher._on_low_quality(event2)

            mock_trigger.assert_called_once()

    def test_get_stats(self):
        """Test statistics reporting."""
        watcher = QualityRollbackWatcher(self.rollback_manager)
        watcher._subscribed = True
        watcher._rollbacks_triggered = 3
        watcher._low_quality_start["hex8_2p"] = time.time()
        watcher._low_quality_start["square8_2p"] = time.time()

        stats = watcher.get_stats()

        assert stats["subscribed"] is True
        assert stats["rollbacks_triggered"] == 3
        assert len(stats["configs_being_monitored"]) == 2
        assert stats["critical_threshold"] == 0.3
        assert stats["sustained_minutes"] == 30

    def test_trigger_quality_rollback_success(self):
        """Test successful quality-triggered rollback."""
        watcher = QualityRollbackWatcher(self.rollback_manager)
        watcher._low_quality_start["hex8_2p"] = time.time() - 3600

        with patch.object(self.rollback_manager, 'rollback_model') as mock_rollback:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.from_version = 2
            mock_result.to_version = 1
            mock_rollback.return_value = mock_result

            with patch.object(watcher, '_emit_rollback_event'):
                watcher._trigger_quality_rollback("hex8_2p", 0.1, 60.0)

        assert watcher._rollbacks_triggered == 1
        assert "hex8_2p" not in watcher._low_quality_start
        mock_rollback.assert_called_once()

    def test_trigger_quality_rollback_failure(self):
        """Test failed quality-triggered rollback."""
        watcher = QualityRollbackWatcher(self.rollback_manager)
        watcher._low_quality_start["hex8_2p"] = time.time() - 3600

        with patch.object(self.rollback_manager, 'rollback_model') as mock_rollback:
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error_message = "Model not found"
            mock_rollback.return_value = mock_result

            with patch.object(watcher, '_emit_rollback_event'):
                watcher._trigger_quality_rollback("hex8_2p", 0.1, 60.0)

        # Rollback counter not incremented on failure
        assert watcher._rollbacks_triggered == 0


class TestWiringFunctions:
    """Test wiring functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = MagicMock()
        # Reset global singletons
        import app.training.rollback_manager as rm
        rm._auto_handler = None
        rm._quality_rollback_watcher = None

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Reset globals
        import app.training.rollback_manager as rm
        rm._auto_handler = None
        rm._quality_rollback_watcher = None

    @patch("app.training.regression_detector.get_regression_detector")
    @patch("app.coordination.event_router.get_event_bus")
    def test_wire_regression_to_rollback(self, mock_get_bus, mock_get_detector):
        """Test wire_regression_to_rollback creates handler."""
        mock_detector = MagicMock()
        mock_get_detector.return_value = mock_detector
        mock_bus = MagicMock()
        mock_get_bus.return_value = mock_bus

        handler = wire_regression_to_rollback(
            self.registry,
            auto_rollback_enabled=True,
            require_approval_for_severe=False,
            subscribe_to_events=True,
        )

        assert handler is not None
        assert isinstance(handler, AutoRollbackHandler)
        mock_detector.add_listener.assert_called_once()

    def test_get_auto_rollback_handler_before_wiring(self):
        """Test getter returns None before wiring."""
        assert get_auto_rollback_handler() is None

    @patch("app.training.regression_detector.get_regression_detector")
    def test_get_auto_rollback_handler_after_wiring(self, mock_get_detector):
        """Test getter returns handler after wiring."""
        mock_get_detector.return_value = MagicMock()

        wire_regression_to_rollback(self.registry)
        handler = get_auto_rollback_handler()

        assert handler is not None

    @patch("app.coordination.event_router.get_event_bus")
    def test_wire_quality_to_rollback(self, mock_get_bus):
        """Test wire_quality_to_rollback creates watcher."""
        mock_bus = MagicMock()
        mock_get_bus.return_value = mock_bus

        watcher = wire_quality_to_rollback(
            self.registry,
            critical_threshold=0.25,
            sustained_minutes=45.0,
        )

        assert watcher is not None
        assert isinstance(watcher, QualityRollbackWatcher)
        assert watcher.critical_threshold == 0.25
        assert watcher.sustained_minutes == 45.0

    def test_get_quality_rollback_watcher_before_wiring(self):
        """Test getter returns None before wiring."""
        assert get_quality_rollback_watcher() is None

    @patch("app.coordination.event_router.get_event_bus")
    def test_get_quality_rollback_watcher_after_wiring(self, mock_get_bus):
        """Test getter returns watcher after wiring."""
        mock_get_bus.return_value = MagicMock()

        wire_quality_to_rollback(self.registry)
        watcher = get_quality_rollback_watcher()

        assert watcher is not None


class TestAutoRollbackHandlerEmitEvents:
    """Test event emission from AutoRollbackHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = MagicMock()
        self.rollback_manager = RollbackManager(
            self.registry,
            history_path=Path(self.temp_dir) / "history.json",
        )

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("app.coordination.event_router.emit_selfplay_target_updated", new_callable=AsyncMock)
    def test_emit_regression_recovery_selfplay(self, mock_emit):
        """Test regression recovery selfplay event emission."""
        handler = AutoRollbackHandler(self.rollback_manager)

        # Test the emission method
        payload = {"config_key": "hex8_2p", "current_target": 1000}
        handler._emit_regression_recovery_selfplay("hex8_2p_v5", "critical", payload)

        # The method creates an async task, so we can't directly verify
        # Just ensure no exceptions are raised


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = MagicMock()
        self.rollback_manager = RollbackManager(
            self.registry,
            history_path=Path(self.temp_dir) / "history.json",
        )

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_on_low_quality_empty_event(self):
        """Test handling of empty/malformed events."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        # Empty object
        watcher._on_low_quality(object())

        # Should use defaults and not crash
        assert "all" not in watcher._low_quality_start

    def test_on_quality_recovered_empty_event(self):
        """Test recovery handling of empty events."""
        watcher = QualityRollbackWatcher(self.rollback_manager)

        # Should not crash
        watcher._on_quality_recovered(object())

    def test_quality_watcher_class_constants(self):
        """Test class-level threshold constants."""
        assert QualityRollbackWatcher.CRITICAL_QUALITY_THRESHOLD == 0.3
        assert QualityRollbackWatcher.WARNING_QUALITY_THRESHOLD == 0.5
        assert QualityRollbackWatcher.SUSTAINED_LOW_QUALITY_MINUTES == 30

    def test_on_regression_event_missing_model_id(self):
        """Test handling event with missing model_id."""
        handler = AutoRollbackHandler(self.rollback_manager)

        mock_event = MagicMock()
        mock_event.payload = {}

        # Should return early without processing
        handler._on_regression_event(mock_event)

        assert len(handler._pending_rollbacks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

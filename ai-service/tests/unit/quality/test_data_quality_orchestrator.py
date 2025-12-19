"""Tests for DataQualityOrchestrator (data quality monitoring).

Tests cover:
- QualityLevel enum
- ConfigQualityState and QualityStats dataclasses
- DataQualityOrchestrator event handling and state management
- Module functions (wire_quality_events, get_quality_orchestrator, reset_quality_orchestrator)
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# =============================================================================
# Test QualityLevel Enum
# =============================================================================

class TestQualityLevel:
    """Tests for QualityLevel enum."""

    def test_excellent_value(self):
        """Test EXCELLENT level value."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.EXCELLENT.value == "excellent"

    def test_good_value(self):
        """Test GOOD level value."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.GOOD.value == "good"

    def test_adequate_value(self):
        """Test ADEQUATE level value."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.ADEQUATE.value == "adequate"

    def test_poor_value(self):
        """Test POOR level value."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.POOR.value == "poor"

    def test_critical_value(self):
        """Test CRITICAL level value."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.CRITICAL.value == "critical"

    def test_from_score_excellent(self):
        """Test excellent threshold (>= 0.9)."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.from_score(0.95) == QualityLevel.EXCELLENT
        assert QualityLevel.from_score(0.9) == QualityLevel.EXCELLENT

    def test_from_score_good(self):
        """Test good threshold (>= 0.7)."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.from_score(0.85) == QualityLevel.GOOD
        assert QualityLevel.from_score(0.7) == QualityLevel.GOOD

    def test_from_score_adequate(self):
        """Test adequate threshold (>= 0.5)."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.from_score(0.6) == QualityLevel.ADEQUATE
        assert QualityLevel.from_score(0.5) == QualityLevel.ADEQUATE

    def test_from_score_poor(self):
        """Test poor threshold (>= 0.3)."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.from_score(0.4) == QualityLevel.POOR
        assert QualityLevel.from_score(0.3) == QualityLevel.POOR

    def test_from_score_critical(self):
        """Test critical threshold (< 0.3)."""
        from app.quality.data_quality_orchestrator import QualityLevel
        assert QualityLevel.from_score(0.2) == QualityLevel.CRITICAL
        assert QualityLevel.from_score(0.0) == QualityLevel.CRITICAL


# =============================================================================
# Test ConfigQualityState Dataclass
# =============================================================================

class TestConfigQualityState:
    """Tests for ConfigQualityState dataclass."""

    def test_create_with_required_fields(self):
        """Test creating state with required fields."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        state = ConfigQualityState(config_key="square8_2p")
        assert state.config_key == "square8_2p"

    def test_default_values(self):
        """Test default values are set correctly."""
        from app.quality.data_quality_orchestrator import (
            ConfigQualityState,
            QualityLevel,
        )

        state = ConfigQualityState(config_key="test")

        assert state.avg_quality_score == 0.0
        assert state.high_quality_count == 0
        assert state.low_quality_count == 0
        assert state.total_games == 0
        assert state.last_update_time == 0.0
        assert state.last_alert_time == 0.0
        assert state.quality_level == QualityLevel.ADEQUATE
        assert state.is_ready_for_training is False
        assert state.has_active_warning is False
        assert state.quality_trend == 0.0
        assert state.samples_since_last_training == 0

    def test_all_fields(self):
        """Test creating state with all fields."""
        from app.quality.data_quality_orchestrator import (
            ConfigQualityState,
            QualityLevel,
        )

        state = ConfigQualityState(
            config_key="hex_2p",
            avg_quality_score=0.85,
            high_quality_count=1000,
            low_quality_count=50,
            total_games=1500,
            last_update_time=1000.0,
            last_alert_time=500.0,
            quality_level=QualityLevel.GOOD,
            is_ready_for_training=True,
            has_active_warning=False,
            quality_trend=0.02,
            samples_since_last_training=200,
        )

        assert state.config_key == "hex_2p"
        assert state.avg_quality_score == 0.85
        assert state.high_quality_count == 1000
        assert state.quality_level == QualityLevel.GOOD
        assert state.is_ready_for_training is True


# =============================================================================
# Test QualityStats Dataclass
# =============================================================================

class TestQualityStats:
    """Tests for QualityStats dataclass."""

    def test_default_values(self):
        """Test default values for aggregate stats."""
        from app.quality.data_quality_orchestrator import QualityStats

        stats = QualityStats()

        assert stats.configs_tracked == 0
        assert stats.total_quality_updates == 0
        assert stats.total_alerts == 0
        assert stats.total_warnings == 0
        assert stats.configs_ready_for_training == 0
        assert stats.configs_with_warnings == 0
        assert stats.avg_quality_across_configs == 0.0
        assert stats.last_activity_time == 0.0

    def test_all_fields(self):
        """Test creating stats with all fields."""
        from app.quality.data_quality_orchestrator import QualityStats

        stats = QualityStats(
            configs_tracked=5,
            total_quality_updates=10000,
            total_alerts=3,
            total_warnings=10,
            configs_ready_for_training=3,
            configs_with_warnings=1,
            avg_quality_across_configs=0.75,
            last_activity_time=time.time(),
        )

        assert stats.configs_tracked == 5
        assert stats.total_quality_updates == 10000
        assert stats.total_alerts == 3
        assert stats.total_warnings == 10


# =============================================================================
# Test DataQualityOrchestrator
# =============================================================================

class TestDataQualityOrchestrator:
    """Tests for DataQualityOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.quality.data_quality_orchestrator import (
            DataQualityOrchestrator,
            reset_quality_orchestrator,
        )

        reset_quality_orchestrator()
        orch = DataQualityOrchestrator()
        yield orch
        reset_quality_orchestrator()

    def test_initialization_defaults(self, orchestrator):
        """Test default initialization."""
        assert orchestrator.stale_threshold_seconds == 3600.0
        assert orchestrator.max_history_per_config == 100
        assert orchestrator.high_quality_threshold == 0.7
        assert orchestrator.low_quality_threshold == 0.3
        assert not orchestrator._subscribed

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        from app.quality.data_quality_orchestrator import DataQualityOrchestrator

        orch = DataQualityOrchestrator(
            stale_threshold_seconds=1800.0,
            max_history_per_config=50,
            high_quality_threshold=0.8,
            low_quality_threshold=0.4,
        )

        assert orch.stale_threshold_seconds == 1800.0
        assert orch.max_history_per_config == 50
        assert orch.high_quality_threshold == 0.8
        assert orch.low_quality_threshold == 0.4

    def test_get_or_create_config_new(self, orchestrator):
        """Test getting/creating a new config."""
        config = orchestrator._get_or_create_config("sq19_2p")
        assert config.config_key == "sq19_2p"
        assert "sq19_2p" in orchestrator._configs

    def test_get_or_create_config_existing(self, orchestrator):
        """Test getting an existing config."""
        config1 = orchestrator._get_or_create_config("sq8_2p")
        config1.avg_quality_score = 0.8
        config2 = orchestrator._get_or_create_config("sq8_2p")
        assert config2.avg_quality_score == 0.8
        assert config1 is config2

    def test_on_quality_score_updated(self, orchestrator):
        """Test handling quality score updated event."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "config": "hex_3p",
            "quality_score": 0.85,
        })

        orchestrator._on_quality_score_updated(event)

        config = orchestrator._configs["hex_3p"]
        assert config.total_games == 1
        assert config.avg_quality_score == 0.85
        assert config.high_quality_count == 1
        assert orchestrator._total_updates == 1

    def test_on_quality_score_updated_running_average(self, orchestrator):
        """Test running average calculation."""
        @dataclass
        class MockEvent:
            payload: dict

        # First update: 0.8
        orchestrator._on_quality_score_updated(MockEvent(payload={
            "config": "test",
            "quality_score": 0.8,
        }))

        # Second update: 0.6
        orchestrator._on_quality_score_updated(MockEvent(payload={
            "config": "test",
            "quality_score": 0.6,
        }))

        config = orchestrator._configs["test"]
        assert config.total_games == 2
        assert config.avg_quality_score == 0.7  # (0.8 + 0.6) / 2
        assert config.quality_trend < 0  # Downward trend

    def test_on_quality_score_updated_low_quality(self, orchestrator):
        """Test handling low quality score."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "config": "test",
            "quality_score": 0.2,
        })

        orchestrator._on_quality_score_updated(event)

        config = orchestrator._configs["test"]
        assert config.low_quality_count == 1

    def test_on_quality_score_updated_no_config(self, orchestrator):
        """Test handling event with no config key."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={"quality_score": 0.8})

        orchestrator._on_quality_score_updated(event)
        assert len(orchestrator._configs) == 0

    def test_on_distribution_changed(self, orchestrator):
        """Test handling distribution changed event."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "config": "sq8_2p",
            "avg_quality": 0.75,
            "high_quality_ratio": 0.6,
            "low_quality_ratio": 0.05,
            "total_games": 1000,
        })

        orchestrator._on_distribution_changed(event)

        config = orchestrator._configs["sq8_2p"]
        assert config.avg_quality_score == 0.75
        assert config.total_games == 1000
        assert config.high_quality_count == 600
        assert config.low_quality_count == 50

        # Check history
        history = orchestrator._config_history["sq8_2p"]
        assert len(history) == 1
        assert history[0]["event_type"] == "distribution_changed"

    def test_on_high_quality_available(self, orchestrator):
        """Test handling high quality data available event."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "config": "hex_2p",
            "high_quality_count": 5000,
            "avg_quality": 0.82,
        })

        orchestrator._on_high_quality_available(event)

        config = orchestrator._configs["hex_2p"]
        assert config.is_ready_for_training is True
        assert config.high_quality_count == 5000
        assert config.avg_quality_score == 0.82

    def test_on_low_quality_warning(self, orchestrator):
        """Test handling low quality warning event."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "config": "sq19_2p",
            "low_quality_count": 500,
            "low_quality_ratio": 0.25,
            "avg_quality": 0.55,
        })

        orchestrator._on_low_quality_warning(event)

        config = orchestrator._configs["sq19_2p"]
        assert config.has_active_warning is True
        assert config.low_quality_count == 500
        assert orchestrator._total_warnings == 1

    def test_on_quality_alert(self, orchestrator):
        """Test handling quality alert event."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "config": "test",
            "alert_type": "critical_degradation",
            "message": "Quality dropped below threshold",
        })

        orchestrator._on_quality_alert(event)

        config = orchestrator._configs["test"]
        assert config.has_active_warning is True
        assert orchestrator._total_alerts == 1

    def test_add_to_history(self, orchestrator):
        """Test adding entries to config history."""
        orchestrator._add_to_history("cfg1", "test_event", {"value": 123})
        orchestrator._add_to_history("cfg1", "test_event", {"value": 456})

        assert len(orchestrator._config_history["cfg1"]) == 2
        assert orchestrator._config_history["cfg1"][0]["value"] == 123

    def test_history_trimming(self):
        """Test that history is trimmed to max_history_per_config."""
        from app.quality.data_quality_orchestrator import DataQualityOrchestrator

        orch = DataQualityOrchestrator(max_history_per_config=5)

        for i in range(10):
            orch._add_to_history("test", "event", {"value": i})

        assert len(orch._config_history["test"]) == 5
        values = [e["value"] for e in orch._config_history["test"]]
        assert values == [5, 6, 7, 8, 9]

    def test_get_config_quality(self, orchestrator):
        """Test getting quality state for a specific config."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        orchestrator._configs["sq8_2p"] = ConfigQualityState(
            config_key="sq8_2p",
            avg_quality_score=0.8,
        )

        state = orchestrator.get_config_quality("sq8_2p")
        assert state is not None
        assert state.avg_quality_score == 0.8

        assert orchestrator.get_config_quality("nonexistent") is None

    def test_get_configs_ready_for_training(self, orchestrator):
        """Test getting configs ready for training."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        orchestrator._configs["ready1"] = ConfigQualityState(
            config_key="ready1",
            is_ready_for_training=True,
        )
        orchestrator._configs["ready2"] = ConfigQualityState(
            config_key="ready2",
            is_ready_for_training=True,
        )
        orchestrator._configs["not_ready"] = ConfigQualityState(
            config_key="not_ready",
            is_ready_for_training=False,
        )

        ready = orchestrator.get_configs_ready_for_training()
        assert len(ready) == 2
        assert all(c.is_ready_for_training for c in ready)

    def test_get_configs_with_warnings(self, orchestrator):
        """Test getting configs with active warnings."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        orchestrator._configs["warned"] = ConfigQualityState(
            config_key="warned",
            has_active_warning=True,
        )
        orchestrator._configs["ok"] = ConfigQualityState(
            config_key="ok",
            has_active_warning=False,
        )

        warned = orchestrator.get_configs_with_warnings()
        assert len(warned) == 1
        assert warned[0].config_key == "warned"

    def test_get_config_history(self, orchestrator):
        """Test getting config history."""
        orchestrator._add_to_history("cfg1", "event", {"a": 1})
        orchestrator._add_to_history("cfg2", "event", {"b": 2})

        # Get all
        history = orchestrator.get_config_history()
        assert "cfg1" in history
        assert "cfg2" in history

        # Get specific
        history = orchestrator.get_config_history("cfg1")
        assert "cfg1" in history
        assert "cfg2" not in history

        # Get nonexistent
        history = orchestrator.get_config_history("nonexistent")
        assert history["nonexistent"] == []

    def test_get_stats(self, orchestrator):
        """Test getting aggregate statistics."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        now = time.time()

        orchestrator._configs["cfg1"] = ConfigQualityState(
            config_key="cfg1",
            avg_quality_score=0.8,
            is_ready_for_training=True,
            last_update_time=now - 100,
        )
        orchestrator._configs["cfg2"] = ConfigQualityState(
            config_key="cfg2",
            avg_quality_score=0.6,
            has_active_warning=True,
            last_alert_time=now - 50,
        )
        orchestrator._total_updates = 1000
        orchestrator._total_alerts = 2
        orchestrator._total_warnings = 5

        stats = orchestrator.get_stats()

        assert stats.configs_tracked == 2
        assert stats.total_quality_updates == 1000
        assert stats.total_alerts == 2
        assert stats.total_warnings == 5
        assert stats.configs_ready_for_training == 1
        assert stats.configs_with_warnings == 1
        assert stats.avg_quality_across_configs == 0.7

    def test_get_status(self, orchestrator):
        """Test getting orchestrator status."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        orchestrator._configs["test"] = ConfigQualityState(
            config_key="test",
            avg_quality_score=0.75,
        )

        status = orchestrator.get_status()

        assert status["subscribed"] is False
        assert status["configs_tracked"] == 1
        assert "test" in status["config_keys"]

    def test_mark_training_started(self, orchestrator):
        """Test marking training started for a config."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        orchestrator._configs["test"] = ConfigQualityState(
            config_key="test",
            is_ready_for_training=True,
            samples_since_last_training=500,
        )

        orchestrator.mark_training_started("test")

        config = orchestrator._configs["test"]
        assert config.is_ready_for_training is False
        assert config.samples_since_last_training == 0

    def test_clear_warning(self, orchestrator):
        """Test clearing warning for a config."""
        from app.quality.data_quality_orchestrator import ConfigQualityState

        orchestrator._configs["test"] = ConfigQualityState(
            config_key="test",
            has_active_warning=True,
        )

        orchestrator.clear_warning("test")

        assert orchestrator._configs["test"].has_active_warning is False

    def test_subscribe_to_events_success(self, orchestrator):
        """Test successful event subscription."""
        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            result = orchestrator.subscribe_to_events()

        assert result is True
        assert orchestrator._subscribed is True
        assert mock_bus.subscribe.call_count == 5

    def test_subscribe_to_events_already_subscribed(self, orchestrator):
        """Test subscription when already subscribed."""
        orchestrator._subscribed = True
        result = orchestrator.subscribe_to_events()
        assert result is True

    def test_subscribe_to_events_failure(self, orchestrator):
        """Test subscription failure."""
        with patch("app.distributed.data_events.get_event_bus", side_effect=Exception("Error")):
            result = orchestrator.subscribe_to_events()

        assert result is False
        assert orchestrator._subscribed is False

    def test_unsubscribe(self, orchestrator):
        """Test event unsubscription."""
        mock_bus = MagicMock()
        orchestrator._subscribed = True

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orchestrator.unsubscribe()

        assert orchestrator._subscribed is False
        assert mock_bus.unsubscribe.call_count == 5

    def test_unsubscribe_not_subscribed(self, orchestrator):
        """Test unsubscribe when not subscribed."""
        orchestrator.unsubscribe()  # Should not raise
        assert orchestrator._subscribed is False


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from app.quality.data_quality_orchestrator import reset_quality_orchestrator

        reset_quality_orchestrator()
        yield
        reset_quality_orchestrator()

    def test_wire_quality_events(self):
        """Test wiring quality events."""
        from app.quality.data_quality_orchestrator import (
            wire_quality_events,
            get_quality_orchestrator,
        )

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch = wire_quality_events()

        assert orch is not None
        assert get_quality_orchestrator() is orch
        assert orch._subscribed is True

    def test_wire_quality_events_custom_params(self):
        """Test wiring with custom parameters."""
        from app.quality.data_quality_orchestrator import wire_quality_events

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch = wire_quality_events(
                stale_threshold_seconds=1800.0,
                high_quality_threshold=0.8,
            )

        assert orch.stale_threshold_seconds == 1800.0
        assert orch.high_quality_threshold == 0.8

    def test_wire_quality_events_singleton(self):
        """Test that wire_quality_events returns same instance."""
        from app.quality.data_quality_orchestrator import wire_quality_events

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch1 = wire_quality_events()
            orch2 = wire_quality_events()

        assert orch1 is orch2

    def test_get_quality_orchestrator_none(self):
        """Test get_quality_orchestrator returns None when not wired."""
        from app.quality.data_quality_orchestrator import get_quality_orchestrator

        assert get_quality_orchestrator() is None

    def test_reset_quality_orchestrator(self):
        """Test resetting the orchestrator singleton."""
        from app.quality.data_quality_orchestrator import (
            wire_quality_events,
            get_quality_orchestrator,
            reset_quality_orchestrator,
        )

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            wire_quality_events()

        assert get_quality_orchestrator() is not None

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            reset_quality_orchestrator()

        assert get_quality_orchestrator() is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestQualityIntegration:
    """Integration tests for quality orchestrator."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton."""
        from app.quality.data_quality_orchestrator import reset_quality_orchestrator

        reset_quality_orchestrator()
        yield
        reset_quality_orchestrator()

    def test_full_quality_lifecycle(self):
        """Test full quality lifecycle from updates to training ready."""
        from app.quality.data_quality_orchestrator import DataQualityOrchestrator

        @dataclass
        class MockEvent:
            payload: dict

        orch = DataQualityOrchestrator()

        config_key = "sq8_2p"

        # Simulate multiple quality updates
        for i in range(100):
            score = 0.7 + (i * 0.002)  # Gradually improving
            orch._on_quality_score_updated(MockEvent(payload={
                "config": config_key,
                "quality_score": min(score, 0.95),
            }))

        config = orch.get_config_quality(config_key)
        assert config is not None
        assert config.total_games == 100
        assert config.avg_quality_score > 0.7
        assert config.high_quality_count > 0

        # Trigger high quality available
        orch._on_high_quality_available(MockEvent(payload={
            "config": config_key,
            "high_quality_count": 80,
            "avg_quality": 0.85,
        }))

        assert config.is_ready_for_training is True

        # Mark training started
        orch.mark_training_started(config_key)
        assert config.is_ready_for_training is False
        assert config.samples_since_last_training == 0

    def test_quality_degradation_flow(self):
        """Test quality degradation and warning flow."""
        from app.quality.data_quality_orchestrator import DataQualityOrchestrator

        @dataclass
        class MockEvent:
            payload: dict

        orch = DataQualityOrchestrator()

        config_key = "test_config"

        # Start with good quality
        orch._on_distribution_changed(MockEvent(payload={
            "config": config_key,
            "avg_quality": 0.8,
            "high_quality_ratio": 0.7,
            "low_quality_ratio": 0.05,
            "total_games": 1000,
        }))

        # Quality degrades
        orch._on_distribution_changed(MockEvent(payload={
            "config": config_key,
            "avg_quality": 0.5,
            "high_quality_ratio": 0.3,
            "low_quality_ratio": 0.3,
            "total_games": 1500,
        }))

        config = orch.get_config_quality(config_key)
        assert config.quality_trend < 0  # Negative trend

        # Low quality warning
        orch._on_low_quality_warning(MockEvent(payload={
            "config": config_key,
            "low_quality_count": 450,
            "low_quality_ratio": 0.3,
            "avg_quality": 0.5,
        }))

        assert config.has_active_warning is True
        assert len(orch.get_configs_with_warnings()) == 1

        # Clear warning
        orch.clear_warning(config_key)
        assert config.has_active_warning is False

    def test_multiple_configs_tracking(self):
        """Test tracking multiple configurations."""
        from app.quality.data_quality_orchestrator import DataQualityOrchestrator

        @dataclass
        class MockEvent:
            payload: dict

        orch = DataQualityOrchestrator()

        configs = ["sq8_2p", "sq19_2p", "hex_2p", "hex_3p"]

        for i, config in enumerate(configs):
            orch._on_quality_score_updated(MockEvent(payload={
                "config": config,
                "quality_score": 0.7 + (i * 0.05),
            }))

        stats = orch.get_stats()
        assert stats.configs_tracked == 4
        assert stats.total_quality_updates == 4

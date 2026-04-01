"""Tests for CurriculumSignalBridge base class.

December 30, 2025: Created as part of Priority 4 consolidation effort.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.curriculum_router import (
    CurriculumSignalBridge,
    CurriculumSignalConfig,
    WeightAdjustment,
    create_signal_bridge,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleBridge(CurriculumSignalBridge):
    """Simple test implementation of CurriculumSignalBridge."""

    WATCHER_NAME = "test_bridge"
    EVENT_TYPES = ["TEST_EVENT"]

    def __init__(self, multiplier: float = 1.2, config=None):
        super().__init__(config=config)
        self._test_multiplier = multiplier
        self._on_adjusted_calls = []

    def _compute_weight_multiplier(self, config_key, payload):
        # Return None if severity is "skip"
        if payload.get("severity") == "skip":
            return None
        return self._test_multiplier

    def _extract_event_details(self, payload):
        return {"severity": payload.get("severity", "unknown")}

    def _on_weight_adjusted(self, adjustment):
        self._on_adjusted_calls.append(adjustment)


@pytest.fixture
def simple_bridge():
    """Create a SimpleBridge instance."""
    return SimpleBridge()


@pytest.fixture
def mock_router():
    """Create a mock event router."""
    router = MagicMock()
    router.subscribe = MagicMock()
    router.unsubscribe = MagicMock()
    router.publish_sync = MagicMock()
    return router


@pytest.fixture
def mock_curriculum_feedback():
    """Create a mock curriculum feedback."""
    feedback = MagicMock()
    feedback._current_weights = {"hex8_2p": 1.0, "square8_4p": 0.5}
    feedback.weight_min = 0.1
    feedback.weight_max = 5.0
    return feedback


# =============================================================================
# CurriculumSignalConfig Tests
# =============================================================================


class TestCurriculumSignalConfig:
    """Tests for CurriculumSignalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CurriculumSignalConfig()
        assert config.max_weight_multiplier == 3.0
        assert config.min_weight_multiplier == 0.3
        assert config.emit_rebalance_events is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CurriculumSignalConfig(
            max_weight_multiplier=2.0,
            min_weight_multiplier=0.5,
            emit_rebalance_events=False,
        )
        assert config.max_weight_multiplier == 2.0
        assert config.min_weight_multiplier == 0.5
        assert config.emit_rebalance_events is False


# =============================================================================
# WeightAdjustment Tests
# =============================================================================


class TestWeightAdjustment:
    """Tests for WeightAdjustment dataclass."""

    def test_was_adjusted_true(self):
        """Test was_adjusted is True when weights differ."""
        adjustment = WeightAdjustment(
            config_key="hex8_2p",
            old_weight=1.0,
            new_weight=1.2,
            multiplier=1.2,
            trigger="test",
        )
        assert adjustment.was_adjusted is True

    def test_was_adjusted_false(self):
        """Test was_adjusted is False when weights are same."""
        adjustment = WeightAdjustment(
            config_key="hex8_2p",
            old_weight=1.0,
            new_weight=1.0,
            multiplier=1.0,
            trigger="test",
        )
        assert adjustment.was_adjusted is False

    def test_was_adjusted_small_difference(self):
        """Test was_adjusted handles small differences correctly."""
        # Difference of 0.0005 should be considered unchanged
        adjustment = WeightAdjustment(
            config_key="hex8_2p",
            old_weight=1.0,
            new_weight=1.0005,
            multiplier=1.0,
            trigger="test",
        )
        assert adjustment.was_adjusted is False

        # Difference of 0.002 should be considered changed
        adjustment2 = WeightAdjustment(
            config_key="hex8_2p",
            old_weight=1.0,
            new_weight=1.002,
            multiplier=1.0,
            trigger="test",
        )
        assert adjustment2.was_adjusted is True

    def test_details_default(self):
        """Test details defaults to empty dict."""
        adjustment = WeightAdjustment(
            config_key="hex8_2p",
            old_weight=1.0,
            new_weight=1.2,
            multiplier=1.2,
            trigger="test",
        )
        assert adjustment.details == {}


# =============================================================================
# CurriculumSignalBridge Initialization Tests
# =============================================================================


class TestBridgeInitialization:
    """Tests for CurriculumSignalBridge initialization."""

    def test_default_initialization(self, simple_bridge):
        """Test default initialization state."""
        assert simple_bridge._subscribed is False
        assert simple_bridge._event_count == 0
        assert simple_bridge._adjustment_count == 0
        assert simple_bridge._last_event_time == 0.0
        assert simple_bridge._state == {}

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = CurriculumSignalConfig(max_weight_multiplier=2.0)
        bridge = SimpleBridge(config=config)
        assert bridge._config.max_weight_multiplier == 2.0

    def test_watcher_name_set(self, simple_bridge):
        """Test watcher name is correctly set."""
        assert simple_bridge.WATCHER_NAME == "test_bridge"

    def test_event_types_set(self, simple_bridge):
        """Test event types are correctly set."""
        assert simple_bridge.EVENT_TYPES == ["TEST_EVENT"]


# =============================================================================
# Subscription Tests
# =============================================================================


class TestSubscription:
    """Tests for event subscription/unsubscription."""

    def test_subscribe_success(self, simple_bridge):
        """Test successful subscription."""
        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = simple_bridge.subscribe()

        assert result is True
        assert simple_bridge._subscribed is True
        mock_subscribe.assert_called_once_with("TEST_EVENT", simple_bridge._handle_event)

    def test_subscribe_already_subscribed(self, simple_bridge, mock_router):
        """Test subscribe when already subscribed returns True."""
        simple_bridge._subscribed = True
        result = simple_bridge.subscribe()
        assert result is True

    def test_subscribe_handles_router_error(self, simple_bridge):
        """Test subscribe fails gracefully when helper raises."""
        with patch(
            "app.coordination.event_router.subscribe",
            side_effect=RuntimeError("router unavailable"),
        ):
            result = simple_bridge.subscribe()

        assert result is False
        assert simple_bridge._subscribed is False

    def test_subscribe_no_event_types(self, mock_router):
        """Test subscribe fails when no event types configured."""

        class EmptyBridge(CurriculumSignalBridge):
            WATCHER_NAME = "empty"
            EVENT_TYPES = []

            def _compute_weight_multiplier(self, config_key, payload):
                return 1.0

        bridge = EmptyBridge()
        result = bridge.subscribe()
        assert result is False

    def test_unsubscribe(self, simple_bridge):
        """Test unsubscription."""
        simple_bridge._subscribed = True

        with patch("app.coordination.event_router.unsubscribe") as mock_unsubscribe:
            simple_bridge.unsubscribe()

        assert simple_bridge._subscribed is False
        mock_unsubscribe.assert_called_once_with("TEST_EVENT", simple_bridge._handle_event)

    def test_unsubscribe_when_not_subscribed(self, simple_bridge):
        """Test unsubscribe does nothing when not subscribed."""
        simple_bridge._subscribed = False
        simple_bridge.unsubscribe()  # Should not raise


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Tests for event handling."""

    def test_handle_event_with_payload_attribute(
        self, simple_bridge, mock_curriculum_feedback
    ):
        """Test handling event with payload attribute."""
        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "severity": "high"}

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            simple_bridge._handle_event(event)

        assert simple_bridge._event_count == 1
        assert simple_bridge._last_event_time > 0

    def test_handle_event_dict_payload(self, simple_bridge, mock_curriculum_feedback):
        """Test handling event as direct dict."""
        event = {"config_key": "hex8_2p", "severity": "medium"}

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            simple_bridge._handle_event(event)

        assert simple_bridge._event_count == 1

    def test_handle_event_missing_config_key(self, simple_bridge):
        """Test event without config_key is skipped."""
        event = {"severity": "high"}  # No config_key

        simple_bridge._handle_event(event)

        assert simple_bridge._event_count == 1
        assert simple_bridge._adjustment_count == 0

    def test_handle_event_multiplier_none(self, simple_bridge):
        """Test event with None multiplier is skipped."""
        event = {"config_key": "hex8_2p", "severity": "skip"}

        simple_bridge._handle_event(event)

        assert simple_bridge._event_count == 1
        assert simple_bridge._adjustment_count == 0

    def test_handle_event_calls_on_adjusted(
        self, simple_bridge, mock_curriculum_feedback
    ):
        """Test _on_weight_adjusted is called after adjustment."""
        event = {"config_key": "hex8_2p", "severity": "high"}

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            simple_bridge._handle_event(event)

        assert len(simple_bridge._on_adjusted_calls) == 1
        adjustment = simple_bridge._on_adjusted_calls[0]
        assert adjustment.config_key == "hex8_2p"


# =============================================================================
# Weight Adjustment Tests
# =============================================================================


class TestWeightAdjustmentLogic:
    """Tests for weight adjustment logic."""

    def test_adjust_weight_increase(self, simple_bridge, mock_curriculum_feedback):
        """Test weight increase."""
        simple_bridge._test_multiplier = 1.5

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            adjustment = simple_bridge._adjust_curriculum_weight(
                "hex8_2p", 1.5, {"severity": "high"}
            )

        assert adjustment is not None
        assert adjustment.old_weight == 1.0
        assert adjustment.new_weight == 1.5
        assert adjustment.was_adjusted is True

    def test_adjust_weight_decrease(self, simple_bridge, mock_curriculum_feedback):
        """Test weight decrease."""
        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            adjustment = simple_bridge._adjust_curriculum_weight(
                "hex8_2p", 0.5, {"severity": "low"}
            )

        assert adjustment is not None
        assert adjustment.old_weight == 1.0
        assert adjustment.new_weight == 0.5

    def test_adjust_weight_clamped_to_max(self, simple_bridge, mock_curriculum_feedback):
        """Test weight is clamped to feedback max."""
        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            adjustment = simple_bridge._adjust_curriculum_weight(
                "hex8_2p", 10.0, {}  # 10x would be 10.0, but max is 5.0
            )

        assert adjustment is not None
        assert adjustment.new_weight == 5.0  # Clamped to weight_max

    def test_adjust_weight_clamped_to_min(self, simple_bridge, mock_curriculum_feedback):
        """Test weight is clamped to feedback min."""
        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            adjustment = simple_bridge._adjust_curriculum_weight(
                "hex8_2p", 0.01, {}  # 0.01 would be 0.01, but min is 0.1
            )

        assert adjustment is not None
        assert adjustment.new_weight == 0.1  # Clamped to weight_min

    def test_adjust_weight_unknown_config(self, simple_bridge, mock_curriculum_feedback):
        """Test adjusting weight for unknown config uses default 1.0."""
        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            adjustment = simple_bridge._adjust_curriculum_weight(
                "unknown_config", 1.5, {}
            )

        assert adjustment is not None
        assert adjustment.old_weight == 1.0  # Default
        assert adjustment.new_weight == 1.5


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for CURRICULUM_REBALANCED event emission."""

    def test_emit_rebalance_event(self, simple_bridge):
        """Test rebalance event is emitted."""
        adjustment = WeightAdjustment(
            config_key="hex8_2p",
            old_weight=1.0,
            new_weight=1.2,
            multiplier=1.2,
            trigger="test_bridge",
            details={"severity": "high"},
        )

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            simple_bridge._emit_rebalance_event(adjustment)

        mock_publish_sync.assert_called_once()
        call_args = mock_publish_sync.call_args

        assert call_args[0][0] == "CURRICULUM_REBALANCED"
        payload = call_args[0][1]
        assert payload["trigger"] == "test_bridge"
        assert payload["changed_configs"] == ["hex8_2p"]
        assert payload["new_weights"] == {"hex8_2p": 1.2}
        assert payload["severity"] == "high"

    def test_emit_disabled_in_config(self, mock_curriculum_feedback):
        """Test event emission can be disabled via config."""
        config = CurriculumSignalConfig(emit_rebalance_events=False)
        bridge = SimpleBridge(config=config)

        event = {"config_key": "hex8_2p", "severity": "high"}

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ), patch(
            "app.coordination.event_router.publish_sync"
        ) as mock_publish_sync:
            bridge._handle_event(event)

        # Event should NOT be emitted
        mock_publish_sync.assert_not_called()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_not_subscribed(self, simple_bridge):
        """Test health check when not subscribed."""
        result = simple_bridge.health_check()

        # Handle both HealthCheckResult and dict
        if hasattr(result, "healthy"):
            assert result.healthy is False
        else:
            assert result["healthy"] is False

    def test_health_check_subscribed(self, simple_bridge, mock_router):
        """Test health check when subscribed."""
        with patch(
            "app.coordination.event_router.get_router", return_value=mock_router
        ):
            simple_bridge.subscribe()

        result = simple_bridge.health_check()

        if hasattr(result, "healthy"):
            assert result.healthy is True
        else:
            assert result["healthy"] is True


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management."""

    def test_get_set_state(self, simple_bridge):
        """Test get/set state."""
        simple_bridge.set_state("counter", 5)
        assert simple_bridge.get_state("counter") == 5

    def test_get_state_default(self, simple_bridge):
        """Test get state with default value."""
        result = simple_bridge.get_state("missing", default=42)
        assert result == 42

    def test_reset_state_all(self, simple_bridge):
        """Test reset all state."""
        simple_bridge.set_state("key1", "value1")
        simple_bridge.set_state("key2", "value2")

        simple_bridge.reset_state()

        assert simple_bridge._state == {}

    def test_reset_state_config_specific(self, simple_bridge):
        """Test reset state for specific config."""
        simple_bridge.set_state("hex8_2p:counter", 5)
        simple_bridge.set_state("hex8_2p:errors", 2)
        simple_bridge.set_state("square8_4p:counter", 10)

        simple_bridge.reset_state("hex8_2p")

        assert "hex8_2p:counter" not in simple_bridge._state
        assert "hex8_2p:errors" not in simple_bridge._state
        assert simple_bridge.get_state("square8_4p:counter") == 10

    def test_stats_property(self, simple_bridge):
        """Test stats property."""
        simple_bridge._event_count = 10
        simple_bridge._adjustment_count = 5

        stats = simple_bridge.stats

        assert stats["watcher_name"] == "test_bridge"
        assert stats["subscribed"] is False
        assert stats["event_count"] == 10
        assert stats["adjustment_count"] == 5


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateSignalBridge:
    """Tests for create_signal_bridge factory function."""

    def test_create_basic_bridge(self):
        """Test creating a basic bridge from functions."""
        bridge = create_signal_bridge(
            watcher_name="factory_test",
            event_types=["TEST_EVENT"],
            compute_multiplier=lambda ck, p: 1.5,
        )

        assert bridge.WATCHER_NAME == "factory_test"
        assert bridge.EVENT_TYPES == ["TEST_EVENT"]

    def test_create_bridge_with_details(self):
        """Test creating bridge with detail extractor."""
        bridge = create_signal_bridge(
            watcher_name="factory_test",
            event_types=["TEST_EVENT"],
            compute_multiplier=lambda ck, p: 1.5,
            extract_details=lambda p: {"custom": p.get("value")},
        )

        details = bridge._extract_event_details({"value": 42})
        assert details == {"custom": 42}

    def test_create_bridge_with_config(self):
        """Test creating bridge with custom config."""
        config = CurriculumSignalConfig(max_weight_multiplier=2.0)
        bridge = create_signal_bridge(
            watcher_name="factory_test",
            event_types=["TEST_EVENT"],
            compute_multiplier=lambda ck, p: 1.5,
            config=config,
        )

        assert bridge._config.max_weight_multiplier == 2.0

    def test_functional_bridge_multiplier(self, mock_curriculum_feedback):
        """Test functional bridge computes multiplier correctly."""
        bridge = create_signal_bridge(
            watcher_name="factory_test",
            event_types=["TEST_EVENT"],
            compute_multiplier=lambda ck, p: 1.0 + p.get("boost", 0),
        )

        result = bridge._compute_weight_multiplier("hex8_2p", {"boost": 0.5})
        assert result == 1.5

    def test_functional_bridge_returns_none(self):
        """Test functional bridge can return None to skip."""
        bridge = create_signal_bridge(
            watcher_name="factory_test",
            event_types=["TEST_EVENT"],
            compute_multiplier=lambda ck, p: None if p.get("skip") else 1.2,
        )

        result = bridge._compute_weight_multiplier("hex8_2p", {"skip": True})
        assert result is None


# =============================================================================
# Bounds Checking Tests
# =============================================================================


class TestBoundsChecking:
    """Tests for multiplier bounds checking."""

    def test_multiplier_clamped_to_config_max(self, mock_curriculum_feedback):
        """Test multiplier is clamped to config max."""
        config = CurriculumSignalConfig(max_weight_multiplier=2.0)
        bridge = SimpleBridge(multiplier=5.0, config=config)

        event = {"config_key": "hex8_2p", "severity": "high"}

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            bridge._handle_event(event)

        # Multiplier should be clamped to 2.0
        assert len(bridge._on_adjusted_calls) == 1
        # New weight = 1.0 * 2.0 = 2.0
        assert bridge._on_adjusted_calls[0].multiplier == 2.0  # Clamped multiplier
        assert bridge._on_adjusted_calls[0].new_weight == 2.0  # Clamped result

    def test_multiplier_clamped_to_config_min(self, mock_curriculum_feedback):
        """Test multiplier is clamped to config min."""
        config = CurriculumSignalConfig(min_weight_multiplier=0.5)
        bridge = SimpleBridge(multiplier=0.1, config=config)

        event = {"config_key": "hex8_2p", "severity": "high"}

        with patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_curriculum_feedback,
        ):
            bridge._handle_event(event)

        # Multiplier should be clamped to 0.5
        assert len(bridge._on_adjusted_calls) == 1
        # New weight = 1.0 * 0.5 = 0.5
        assert bridge._on_adjusted_calls[0].new_weight == 0.5

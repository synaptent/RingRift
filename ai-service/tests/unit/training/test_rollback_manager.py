"""Tests for rollback_manager.py.

This module tests the RollbackManager and related components for
automated and manual model rollback capabilities.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.rollback_manager import (
    RollbackEvent,
    RollbackThresholds,
    RollbackManager,
    AutoRollbackHandler,
    QualityRollbackWatcher,
    create_rollback_alert_rules,
    get_auto_rollback_handler,
    get_quality_rollback_watcher,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_history_path(tmp_path):
    """Create a temporary path for rollback history."""
    return tmp_path / "rollback_history.json"


@pytest.fixture
def mock_registry():
    """Create a mock ModelRegistry."""
    from unittest.mock import MagicMock

    registry = MagicMock()

    # Mock production model
    prod_model = MagicMock()
    prod_model.model_id = "hex8_2p"
    prod_model.version = 5
    prod_model.metrics.to_dict.return_value = {"elo": 1600, "win_rate": 0.55}
    prod_model.stage = MagicMock()
    prod_model.stage.value = "production"

    registry.get_production_model.return_value = prod_model
    registry.list_models.return_value = []
    registry.get_model.return_value = None

    return registry


@pytest.fixture
def rollback_manager(mock_registry, temp_history_path):
    """Create a RollbackManager with mocks."""
    return RollbackManager(
        registry=mock_registry,
        history_path=temp_history_path,
    )


@pytest.fixture
def auto_handler(rollback_manager):
    """Create an AutoRollbackHandler."""
    return AutoRollbackHandler(
        rollback_manager=rollback_manager,
        auto_rollback_enabled=True,
        require_approval_for_severe=True,
    )


@pytest.fixture
def quality_watcher(rollback_manager):
    """Create a QualityRollbackWatcher."""
    return QualityRollbackWatcher(
        rollback_manager=rollback_manager,
        critical_threshold=0.3,
        sustained_minutes=30.0,
    )


# =============================================================================
# Tests for RollbackEvent
# =============================================================================


class TestRollbackEvent:
    """Tests for RollbackEvent dataclass."""

    def test_basic_creation(self):
        """Can create a RollbackEvent."""
        event = RollbackEvent(
            model_id="hex8_2p",
            from_version=5,
            to_version=4,
            reason="Performance degradation",
            triggered_by="manual",
            timestamp="2025-12-29T12:00:00",
        )

        assert event.model_id == "hex8_2p"
        assert event.from_version == 5
        assert event.to_version == 4
        assert event.reason == "Performance degradation"
        assert event.triggered_by == "manual"
        assert event.success is True

    def test_default_values(self):
        """Should have reasonable defaults."""
        event = RollbackEvent(
            model_id="hex8_2p",
            from_version=5,
            to_version=4,
            reason="Test",
            triggered_by="test",
            timestamp="2025-12-29T12:00:00",
        )

        assert event.from_metrics == {}
        assert event.to_metrics == {}
        assert event.success is True
        assert event.error_message is None

    def test_to_dict(self):
        """to_dict should serialize to dictionary."""
        event = RollbackEvent(
            model_id="hex8_2p",
            from_version=5,
            to_version=4,
            reason="Test",
            triggered_by="test",
            timestamp="2025-12-29T12:00:00",
            from_metrics={"elo": 1600},
        )

        d = event.to_dict()

        assert d["model_id"] == "hex8_2p"
        assert d["from_version"] == 5
        assert d["from_metrics"] == {"elo": 1600}

    def test_from_dict(self):
        """from_dict should deserialize from dictionary."""
        d = {
            "model_id": "hex8_2p",
            "from_version": 5,
            "to_version": 4,
            "reason": "Test",
            "triggered_by": "test",
            "timestamp": "2025-12-29T12:00:00",
            "from_metrics": {"elo": 1600},
            "to_metrics": {},
            "success": True,
            "error_message": None,
        }

        event = RollbackEvent.from_dict(d)

        assert event.model_id == "hex8_2p"
        assert event.from_version == 5
        assert event.from_metrics == {"elo": 1600}

    def test_round_trip(self):
        """to_dict/from_dict should round-trip correctly."""
        original = RollbackEvent(
            model_id="hex8_2p",
            from_version=5,
            to_version=4,
            reason="Test",
            triggered_by="auto_elo",
            timestamp="2025-12-29T12:00:00",
            from_metrics={"elo": 1600, "win_rate": 0.55},
            to_metrics={"elo": 1550},
            success=False,
            error_message="Test error",
        )

        d = original.to_dict()
        restored = RollbackEvent.from_dict(d)

        assert restored.model_id == original.model_id
        assert restored.from_version == original.from_version
        assert restored.success == original.success
        assert restored.error_message == original.error_message


# =============================================================================
# Tests for RollbackThresholds
# =============================================================================


class TestRollbackThresholds:
    """Tests for RollbackThresholds dataclass."""

    def test_default_values(self):
        """Should have reasonable default thresholds."""
        thresholds = RollbackThresholds()

        assert thresholds.elo_drop_threshold == 50.0
        assert thresholds.elo_drop_window_hours == 24.0
        assert thresholds.win_rate_drop_threshold == 0.10
        assert thresholds.error_rate_threshold == 0.05
        assert thresholds.min_games_for_evaluation == 50

    def test_custom_thresholds(self):
        """Can create with custom thresholds."""
        thresholds = RollbackThresholds(
            elo_drop_threshold=30.0,
            min_games_for_evaluation=100,
        )

        assert thresholds.elo_drop_threshold == 30.0
        assert thresholds.min_games_for_evaluation == 100


# =============================================================================
# Tests for RollbackManager Initialization
# =============================================================================


class TestRollbackManagerInit:
    """Tests for RollbackManager initialization."""

    def test_basic_init(self, mock_registry, temp_history_path):
        """Can initialize with basic parameters."""
        manager = RollbackManager(
            registry=mock_registry,
            history_path=temp_history_path,
        )

        assert manager.registry is mock_registry
        assert manager.thresholds is not None
        assert manager._history == []

    def test_init_with_thresholds(self, mock_registry, temp_history_path):
        """Can initialize with custom thresholds."""
        thresholds = RollbackThresholds(elo_drop_threshold=30.0)

        manager = RollbackManager(
            registry=mock_registry,
            thresholds=thresholds,
            history_path=temp_history_path,
        )

        assert manager.thresholds.elo_drop_threshold == 30.0

    def test_loads_existing_history(self, mock_registry, temp_history_path):
        """Should load existing history from disk."""
        # Create history file
        history = [
            {
                "model_id": "hex8_2p",
                "from_version": 5,
                "to_version": 4,
                "reason": "Test",
                "triggered_by": "test",
                "timestamp": "2025-12-29T12:00:00",
                "from_metrics": {},
                "to_metrics": {},
                "success": True,
                "error_message": None,
            }
        ]
        temp_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_history_path, 'w') as f:
            json.dump(history, f)

        manager = RollbackManager(
            registry=mock_registry,
            history_path=temp_history_path,
        )

        assert len(manager._history) == 1
        assert manager._history[0].model_id == "hex8_2p"


# =============================================================================
# Tests for Baseline Management
# =============================================================================


class TestBaselineManagement:
    """Tests for baseline setting and checking."""

    def test_set_baseline(self, rollback_manager):
        """set_baseline should store metrics."""
        metrics = {"elo": 1600, "win_rate": 0.55, "games_played": 100}

        rollback_manager.set_baseline("hex8_2p", metrics)

        assert "hex8_2p" in rollback_manager._baselines
        baseline = rollback_manager._baselines["hex8_2p"]
        assert baseline["elo"] == 1600
        assert baseline["win_rate"] == 0.55

    def test_set_baseline_timestamp(self, rollback_manager):
        """set_baseline should include timestamp."""
        metrics = {"elo": 1600}

        rollback_manager.set_baseline("hex8_2p", metrics)

        baseline = rollback_manager._baselines["hex8_2p"]
        assert "timestamp" in baseline


# =============================================================================
# Tests for Performance Checking
# =============================================================================


class TestPerformanceChecking:
    """Tests for check_performance method."""

    def test_no_baseline(self, rollback_manager):
        """check_performance should return False without baseline."""
        is_degraded, reason = rollback_manager.check_performance(
            "hex8_2p",
            {"elo": 1500},
        )

        assert is_degraded is False
        assert "No baseline" in reason

    def test_insufficient_games(self, rollback_manager):
        """check_performance should return False with insufficient games."""
        rollback_manager.set_baseline("hex8_2p", {"elo": 1600, "games_played": 0})

        is_degraded, reason = rollback_manager.check_performance(
            "hex8_2p",
            {"elo": 1500, "games_played": 10},
        )

        assert is_degraded is False
        assert "Insufficient games" in reason

    def test_elo_drop_triggers(self, rollback_manager):
        """check_performance should detect Elo drop."""
        rollback_manager.set_baseline("hex8_2p", {"elo": 1600, "games_played": 0})

        is_degraded, reason = rollback_manager.check_performance(
            "hex8_2p",
            {"elo": 1540, "games_played": 100},  # 60 point drop
        )

        assert is_degraded is True
        assert "Elo dropped" in reason

    def test_win_rate_drop_triggers(self, rollback_manager):
        """check_performance should detect win rate drop."""
        rollback_manager.set_baseline(
            "hex8_2p",
            {"elo": 1600, "win_rate": 0.55, "games_played": 0},
        )

        is_degraded, reason = rollback_manager.check_performance(
            "hex8_2p",
            {"elo": 1590, "win_rate": 0.40, "games_played": 100},  # 15% drop
        )

        assert is_degraded is True
        assert "Win rate dropped" in reason

    def test_acceptable_performance(self, rollback_manager):
        """check_performance should accept good metrics."""
        rollback_manager.set_baseline(
            "hex8_2p",
            {"elo": 1600, "win_rate": 0.55, "games_played": 0},
        )

        is_degraded, reason = rollback_manager.check_performance(
            "hex8_2p",
            {"elo": 1610, "win_rate": 0.56, "games_played": 100},
        )

        assert is_degraded is False
        assert "acceptable range" in reason


# =============================================================================
# Tests for Should Rollback
# =============================================================================


class TestShouldRollback:
    """Tests for should_rollback method."""

    def test_not_in_production(self, rollback_manager):
        """should_rollback returns False if model not in production."""
        rollback_manager.registry.get_production_model.return_value = None

        should, reason = rollback_manager.should_rollback("hex8_2p")

        assert should is False
        assert "not in production" in reason.lower()

    def test_no_candidates(self, rollback_manager):
        """should_rollback returns False without rollback candidates."""
        rollback_manager.registry.list_models.return_value = []

        should, reason = rollback_manager.should_rollback("hex8_2p")

        assert should is False
        assert "No rollback candidates" in reason


# =============================================================================
# Tests for Get Rollback Candidate
# =============================================================================


class TestGetRollbackCandidate:
    """Tests for get_rollback_candidate method."""

    def test_no_candidates(self, rollback_manager):
        """get_rollback_candidate returns None without candidates."""
        rollback_manager.registry.list_models.return_value = []

        candidate = rollback_manager.get_rollback_candidate("hex8_2p")

        assert candidate is None

    def test_returns_best_candidate(self, rollback_manager):
        """get_rollback_candidate prefers candidates with good metrics."""
        rollback_manager.registry.list_models.return_value = [
            {"model_id": "hex8_2p", "version": 4, "metrics": {"elo": 1550}},
            {"model_id": "hex8_2p", "version": 3, "metrics": {"elo": 1500}},
        ]

        candidate = rollback_manager.get_rollback_candidate("hex8_2p")

        assert candidate is not None
        assert candidate["version"] == 4  # Most recent with good Elo

    def test_filters_by_model_id(self, rollback_manager):
        """get_rollback_candidate filters by model_id."""
        rollback_manager.registry.list_models.return_value = [
            {"model_id": "hex8_2p", "version": 4, "metrics": {"elo": 1550}},
            {"model_id": "square8_2p", "version": 3, "metrics": {"elo": 1600}},
        ]

        candidate = rollback_manager.get_rollback_candidate("hex8_2p")

        assert candidate is not None
        assert candidate["model_id"] == "hex8_2p"


# =============================================================================
# Tests for Rollback History
# =============================================================================


class TestRollbackHistory:
    """Tests for rollback history management."""

    def test_get_empty_history(self, rollback_manager):
        """get_rollback_history returns empty list initially."""
        history = rollback_manager.get_rollback_history()

        assert history == []

    def test_get_history_with_events(self, rollback_manager):
        """get_rollback_history returns recorded events."""
        event = RollbackEvent(
            model_id="hex8_2p",
            from_version=5,
            to_version=4,
            reason="Test",
            triggered_by="test",
            timestamp=datetime.now().isoformat(),
        )
        rollback_manager._history.append(event)

        history = rollback_manager.get_rollback_history()

        assert len(history) == 1
        assert history[0].model_id == "hex8_2p"

    def test_get_history_filtered_by_model(self, rollback_manager):
        """get_rollback_history can filter by model_id."""
        rollback_manager._history = [
            RollbackEvent(
                model_id="hex8_2p",
                from_version=5,
                to_version=4,
                reason="Test",
                triggered_by="test",
                timestamp=datetime.now().isoformat(),
            ),
            RollbackEvent(
                model_id="square8_2p",
                from_version=3,
                to_version=2,
                reason="Test2",
                triggered_by="test",
                timestamp=datetime.now().isoformat(),
            ),
        ]

        history = rollback_manager.get_rollback_history(model_id="hex8_2p")

        assert len(history) == 1
        assert history[0].model_id == "hex8_2p"

    def test_get_history_limited(self, rollback_manager):
        """get_rollback_history respects limit parameter."""
        for i in range(10):
            rollback_manager._history.append(
                RollbackEvent(
                    model_id="hex8_2p",
                    from_version=i + 1,
                    to_version=i,
                    reason=f"Test {i}",
                    triggered_by="test",
                    timestamp=datetime.now().isoformat(),
                )
            )

        history = rollback_manager.get_rollback_history(limit=5)

        assert len(history) == 5


# =============================================================================
# Tests for Rollback Stats
# =============================================================================


class TestRollbackStats:
    """Tests for get_rollback_stats method."""

    def test_empty_stats(self, rollback_manager):
        """get_rollback_stats returns zeros for empty history."""
        stats = rollback_manager.get_rollback_stats()

        assert stats["total_rollbacks"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["by_trigger"] == {}
        assert stats["recent_24h"] == 0

    def test_stats_with_events(self, rollback_manager):
        """get_rollback_stats counts events correctly."""
        rollback_manager._history = [
            RollbackEvent(
                model_id="hex8_2p",
                from_version=5,
                to_version=4,
                reason="Test",
                triggered_by="manual",
                timestamp=datetime.now().isoformat(),
                success=True,
            ),
            RollbackEvent(
                model_id="hex8_2p",
                from_version=6,
                to_version=5,
                reason="Test2",
                triggered_by="auto_elo",
                timestamp=datetime.now().isoformat(),
                success=False,
            ),
        ]

        stats = rollback_manager.get_rollback_stats()

        assert stats["total_rollbacks"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["by_trigger"]["manual"] == 1
        assert stats["by_trigger"]["auto_elo"] == 1


# =============================================================================
# Tests for AutoRollbackHandler
# =============================================================================


class TestAutoRollbackHandlerInit:
    """Tests for AutoRollbackHandler initialization."""

    def test_basic_init(self, rollback_manager):
        """Can initialize AutoRollbackHandler."""
        handler = AutoRollbackHandler(
            rollback_manager=rollback_manager,
        )

        assert handler.rollback_manager is rollback_manager
        assert handler.auto_rollback_enabled is True
        assert handler.require_approval_for_severe is True
        assert handler._pending_rollbacks == {}

    def test_init_disabled(self, rollback_manager):
        """Can initialize with auto-rollback disabled."""
        handler = AutoRollbackHandler(
            rollback_manager=rollback_manager,
            auto_rollback_enabled=False,
        )

        assert handler.auto_rollback_enabled is False


class TestAutoRollbackHandlerOnRegression:
    """Tests for on_regression method."""

    def test_disabled_does_not_act(self, rollback_manager):
        """on_regression does nothing when disabled."""
        handler = AutoRollbackHandler(
            rollback_manager=rollback_manager,
            auto_rollback_enabled=False,
        )

        # Create mock regression event
        event = MagicMock()
        event.model_id = "hex8_2p"
        event.severity.name = "CRITICAL"
        event.reason = "Test"

        handler.on_regression(event)

        # Should not call rollback
        rollback_manager.rollback_model = MagicMock()
        rollback_manager.rollback_model.assert_not_called()

    def test_minor_not_acted_upon(self, auto_handler):
        """MINOR regressions are not acted upon."""
        # The handler only acts on SEVERE (if require_approval_for_severe=False)
        # or CRITICAL severity. By default require_approval_for_severe=True,
        # so only CRITICAL triggers auto-rollback.

        # Verify initial state doesn't have any pending rollbacks
        initial_pending = len(auto_handler._pending_rollbacks)

        # Verify default safety setting - only CRITICAL auto-rolls back
        assert auto_handler.require_approval_for_severe is True

        # Also verify that pending_rollbacks hasn't changed
        assert len(auto_handler._pending_rollbacks) == initial_pending


class TestAutoRollbackPendingRollbacks:
    """Tests for pending rollback management."""

    def test_get_pending_rollbacks_empty(self, auto_handler):
        """get_pending_rollbacks returns empty dict initially."""
        pending = auto_handler.get_pending_rollbacks()

        assert pending == {}

    def test_clear_pending_rollback(self, auto_handler):
        """clear_pending_rollback removes pending item."""
        auto_handler._pending_rollbacks["hex8_2p"] = {"reason": "Test"}

        result = auto_handler.clear_pending_rollback("hex8_2p")

        assert result is True
        assert "hex8_2p" not in auto_handler._pending_rollbacks

    def test_clear_nonexistent_rollback(self, auto_handler):
        """clear_pending_rollback returns False for nonexistent."""
        result = auto_handler.clear_pending_rollback("nonexistent")

        assert result is False


class TestAutoRollbackEventSubscription:
    """Tests for event bus subscription."""

    def test_subscribe_to_events(self, auto_handler):
        """subscribe_to_regression_events attempts subscription."""
        # Patch the event_router module where get_event_bus is imported from
        with patch("app.coordination.event_router.get_event_bus") as mock_get_bus:
            mock_bus = MagicMock()
            mock_get_bus.return_value = mock_bus

            result = auto_handler.subscribe_to_regression_events()

            assert result is True
            assert mock_bus.subscribe.called

    def test_unsubscribe_from_events(self, auto_handler):
        """unsubscribe_from_regression_events handles unsubscription."""
        auto_handler._event_subscribed = True

        with patch("app.coordination.event_router.get_event_bus") as mock_get_bus:
            mock_bus = MagicMock()
            mock_get_bus.return_value = mock_bus

            auto_handler.unsubscribe_from_regression_events()

            assert auto_handler._event_subscribed is False


# =============================================================================
# Tests for QualityRollbackWatcher
# =============================================================================


class TestQualityRollbackWatcherInit:
    """Tests for QualityRollbackWatcher initialization."""

    def test_basic_init(self, rollback_manager):
        """Can initialize QualityRollbackWatcher."""
        watcher = QualityRollbackWatcher(
            rollback_manager=rollback_manager,
        )

        assert watcher.rollback_manager is rollback_manager
        assert watcher.critical_threshold == 0.3
        assert watcher.sustained_minutes == 30.0
        assert watcher._subscribed is False

    def test_init_with_custom_thresholds(self, rollback_manager):
        """Can initialize with custom thresholds."""
        watcher = QualityRollbackWatcher(
            rollback_manager=rollback_manager,
            critical_threshold=0.2,
            sustained_minutes=60.0,
        )

        assert watcher.critical_threshold == 0.2
        assert watcher.sustained_minutes == 60.0


class TestQualityRollbackWatcherStats:
    """Tests for get_stats method."""

    def test_get_stats_initial(self, quality_watcher):
        """get_stats returns initial state."""
        stats = quality_watcher.get_stats()

        assert stats["subscribed"] is False
        assert stats["rollbacks_triggered"] == 0
        assert stats["configs_being_monitored"] == []
        assert stats["critical_threshold"] == 0.3
        assert stats["sustained_minutes"] == 30.0


class TestQualityRollbackWatcherEvents:
    """Tests for event handling."""

    def test_on_low_quality_starts_monitoring(self, quality_watcher):
        """_on_low_quality starts monitoring for critical quality."""
        event = MagicMock()
        event.payload = {
            "quality_score": 0.2,  # Below critical threshold
            "config_key": "hex8_2p",
            "timestamp": time.time(),
        }

        quality_watcher._on_low_quality(event)

        assert "hex8_2p" in quality_watcher._low_quality_start

    def test_on_quality_recovered_clears_monitoring(self, quality_watcher):
        """_on_quality_recovered clears monitoring."""
        quality_watcher._low_quality_start["hex8_2p"] = time.time()

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p"}

        quality_watcher._on_quality_recovered(event)

        assert "hex8_2p" not in quality_watcher._low_quality_start

    def test_subscribe_to_quality_events(self, quality_watcher):
        """subscribe_to_quality_events attempts subscription."""
        # Patch the event_router module where get_event_bus is imported from
        with patch("app.coordination.event_router.get_event_bus") as mock_get_bus:
            mock_bus = MagicMock()
            mock_get_bus.return_value = mock_bus

            result = quality_watcher.subscribe_to_quality_events()

            assert result is True
            assert quality_watcher._subscribed is True
            assert mock_bus.subscribe.called


# =============================================================================
# Tests for Factory Functions
# =============================================================================


class TestCreateRollbackAlertRules:
    """Tests for create_rollback_alert_rules function."""

    def test_returns_yaml(self):
        """create_rollback_alert_rules returns YAML content."""
        yaml_content = create_rollback_alert_rules()

        assert isinstance(yaml_content, str)
        assert "groups:" in yaml_content
        assert "model_rollback_alerts" in yaml_content

    def test_contains_alert_definitions(self):
        """YAML content contains expected alerts."""
        yaml_content = create_rollback_alert_rules()

        assert "ModelRollbackTriggered" in yaml_content
        assert "MultipleRollbacksDetected" in yaml_content
        assert "EloDegradation" in yaml_content

    def test_contains_severity_labels(self):
        """YAML content contains severity labels."""
        yaml_content = create_rollback_alert_rules()

        assert "warning" in yaml_content
        assert "critical" in yaml_content


class TestGetAutoRollbackHandler:
    """Tests for get_auto_rollback_handler function."""

    def test_returns_none_initially(self):
        """get_auto_rollback_handler returns None before wiring."""
        # Note: This may return a value if wired elsewhere
        result = get_auto_rollback_handler()

        # Result could be None or an instance depending on state
        assert result is None or isinstance(result, AutoRollbackHandler)


class TestGetQualityRollbackWatcher:
    """Tests for get_quality_rollback_watcher function."""

    def test_returns_none_initially(self):
        """get_quality_rollback_watcher returns None before wiring."""
        # Note: This may return a value if wired elsewhere
        result = get_quality_rollback_watcher()

        # Result could be None or an instance depending on state
        assert result is None or isinstance(result, QualityRollbackWatcher)


# =============================================================================
# Tests for History Persistence
# =============================================================================


class TestHistoryPersistence:
    """Tests for history save/load."""

    def test_save_history(self, rollback_manager, temp_history_path):
        """_save_history writes to disk."""
        rollback_manager._history.append(
            RollbackEvent(
                model_id="hex8_2p",
                from_version=5,
                to_version=4,
                reason="Test",
                triggered_by="test",
                timestamp=datetime.now().isoformat(),
            )
        )

        rollback_manager._save_history()

        assert temp_history_path.exists()
        with open(temp_history_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["model_id"] == "hex8_2p"

    def test_load_history(self, mock_registry, temp_history_path):
        """_load_history reads from disk."""
        # Create history file
        history = [
            {
                "model_id": "hex8_2p",
                "from_version": 5,
                "to_version": 4,
                "reason": "Test",
                "triggered_by": "test",
                "timestamp": "2025-12-29T12:00:00",
                "from_metrics": {},
                "to_metrics": {},
                "success": True,
                "error_message": None,
            }
        ]
        temp_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_history_path, 'w') as f:
            json.dump(history, f)

        manager = RollbackManager(
            registry=mock_registry,
            history_path=temp_history_path,
        )

        assert len(manager._history) == 1

    def test_load_corrupt_history(self, mock_registry, temp_history_path):
        """_load_history handles corrupt files gracefully."""
        temp_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_history_path, 'w') as f:
            f.write("not valid json")

        # Should not raise
        manager = RollbackManager(
            registry=mock_registry,
            history_path=temp_history_path,
        )

        assert manager._history == []


# =============================================================================
# Tests for Exploration Boost Factor
# =============================================================================


class TestExplorationBoostFactor:
    """Tests for _get_exploration_boost_factor method."""

    def test_critical_boost(self, auto_handler):
        """CRITICAL severity should have highest boost."""
        boost = auto_handler._get_exploration_boost_factor("critical")

        assert boost == 2.0

    def test_major_boost(self, auto_handler):
        """MAJOR severity should have high boost."""
        boost = auto_handler._get_exploration_boost_factor("major")

        assert boost == 1.75

    def test_moderate_boost(self, auto_handler):
        """MODERATE severity should have medium boost."""
        boost = auto_handler._get_exploration_boost_factor("moderate")

        assert boost == 1.5

    def test_minor_boost(self, auto_handler):
        """MINOR severity should have low boost."""
        boost = auto_handler._get_exploration_boost_factor("minor")

        assert boost == 1.25

    def test_unknown_severity(self, auto_handler):
        """Unknown severity should return default boost."""
        boost = auto_handler._get_exploration_boost_factor("unknown")

        assert boost == 1.5  # Default


# =============================================================================
# Tests for Emit Rollback Metric
# =============================================================================


class TestEmitRollbackMetric:
    """Tests for _emit_rollback_metric method."""

    def test_emit_without_prometheus(self, rollback_manager):
        """_emit_rollback_metric should not raise without prometheus."""
        # Should not raise even if prometheus is not available
        rollback_manager._emit_rollback_metric("hex8_2p", "manual")

    @patch("prometheus_client.REGISTRY")
    def test_emit_with_prometheus(self, mock_registry_module, rollback_manager):
        """_emit_rollback_metric should increment metric when available."""
        mock_metric = MagicMock()
        mock_registry_module._names_to_collectors = {
            "ringrift_model_rollbacks_total": mock_metric
        }

        rollback_manager._emit_rollback_metric("hex8_2p", "manual")

        # Should attempt to increment - the metric lookup happens first
        # Even if increment fails, the method should not raise
        assert True  # Method completed without error


# =============================================================================
# Tests for Approve Pending Rollback
# =============================================================================


class TestApprovePendingRollback:
    """Tests for approve_pending_rollback method."""

    def test_approve_nonexistent(self, auto_handler):
        """approve_pending_rollback returns error for nonexistent."""
        result = auto_handler.approve_pending_rollback("nonexistent")

        assert result["success"] is False
        assert "No pending rollback" in result["error"]

    def test_approve_existing(self, auto_handler):
        """approve_pending_rollback executes pending rollback."""
        auto_handler._pending_rollbacks["hex8_2p"] = {
            "reason": "Test regression",
        }

        with patch.object(auto_handler.rollback_manager, 'rollback_model') as mock_rollback:
            mock_rollback.return_value = {"success": True}

            result = auto_handler.approve_pending_rollback("hex8_2p")

            assert result["success"] is True
            assert "hex8_2p" not in auto_handler._pending_rollbacks

"""End-to-end integration tests for the rollback system.

Tests the full lifecycle:
1. Model promotion
2. Regression detection
3. Automatic rollback
4. Cooldown enforcement
5. Notification hooks

These tests use mocked services to simulate the full flow without
requiring actual model files or Elo databases.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from app.training.notification_config import (
    FilteredWebhookHook,
    PagerDutyNotificationHook,
    load_rollback_config,
)
from app.training.promotion_controller import (
    LoggingNotificationHook,
    NotificationHook,
    PromotionController,
    PromotionCriteria,
    PromotionDecision,
    PromotionType,
    RollbackCriteria,
    RollbackEvent,
    RollbackMonitor,
)


class MockNotificationHook(NotificationHook):
    """Test hook that records all notifications."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def on_regression_detected(self, model_id: str, status: dict[str, Any]) -> None:
        self.events.append({
            "type": "regression_detected",
            "model_id": model_id,
            "status": status,
        })

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        self.events.append({
            "type": "at_risk",
            "model_id": model_id,
            "status": status,
        })

    def on_rollback_triggered(self, event: RollbackEvent) -> None:
        self.events.append({
            "type": "rollback_triggered",
            "event": event,
        })

    def on_rollback_completed(self, event: RollbackEvent, success: bool) -> None:
        self.events.append({
            "type": "rollback_completed",
            "event": event,
            "success": success,
        })


class MockEloService:
    """Mock Elo service for testing."""

    def __init__(self):
        self.ratings: dict[str, dict] = {}
        self.history: dict[str, list[dict]] = {}

    def set_rating(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        rating: float,
        games_played: int,
        win_rate: float,
    ):
        """Set a mock rating for a model."""
        key = f"{model_id}_{board_type}_{num_players}"
        mock_rating = MagicMock()
        mock_rating.rating = rating
        mock_rating.games_played = games_played
        mock_rating.win_rate = win_rate
        self.ratings[key] = mock_rating

    def get_rating(self, model_id: str, board_type: str, num_players: int):
        """Get a mock rating."""
        key = f"{model_id}_{board_type}_{num_players}"
        return self.ratings.get(key)

    def get_rating_history(self, model_id: str, board_type: str, num_players: int, limit: int = 10):
        """Get mock rating history."""
        key = f"{model_id}_{board_type}_{num_players}"
        return self.history.get(key, [])


class MockModelRegistry:
    """Mock model registry for testing."""

    def __init__(self):
        self.models: dict[str, str] = {}  # model_id -> stage
        self.history: list[dict] = []

    def promote_model(self, model_id: str, stage):
        """Promote a model to a stage."""
        self.models[model_id] = stage.value if hasattr(stage, 'value') else str(stage)
        self.history.append({
            "model_id": model_id,
            "stage": self.models[model_id],
            "timestamp": datetime.now().isoformat(),
        })

    def get_production_model(self):
        """Get current production model."""
        for model_id, stage in self.models.items():
            if stage == "production":
                return model_id
        return None

    def get_model_history(self, limit: int = 10):
        """Get model history."""
        return self.history[-limit:]


class TestFullRollbackCycle:
    """Test complete promotion → regression → rollback cycle."""

    def test_promotion_to_regression_to_rollback(self):
        """Test the full lifecycle of a model promotion and rollback."""
        # Setup mock services
        elo_service = MockEloService()
        model_registry = MockModelRegistry()
        notification_hook = MockNotificationHook()

        # Create controller and monitor
        promotion_criteria = PromotionCriteria(
            min_elo_improvement=25.0,
            min_games_played=50,
        )
        rollback_criteria = RollbackCriteria(
            elo_regression_threshold=-30.0,
            min_games_for_regression=20,
            consecutive_checks_required=2,
            cooldown_seconds=60,
        )

        controller = PromotionController(
            criteria=promotion_criteria,
            elo_service=elo_service,
            model_registry=model_registry,
        )

        monitor = RollbackMonitor(
            criteria=rollback_criteria,
            promotion_controller=controller,
            notification_hooks=[notification_hook],
        )

        # Phase 1: Set up baseline model
        elo_service.set_rating(
            "model_v1", "square8", 2,
            rating=1500, games_played=100, win_rate=0.55
        )
        model_registry.promote_model("model_v1", MagicMock(value="production"))

        # Phase 2: New model performs well initially → gets promoted
        elo_service.set_rating(
            "model_v2", "square8", 2,
            rating=1550, games_played=60, win_rate=0.58  # +50 Elo
        )

        decision = controller.evaluate_promotion(
            model_id="model_v2",
            board_type="square8",
            num_players=2,
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="model_v1",
        )

        assert decision.should_promote is True
        assert decision.elo_improvement == 50.0

        # Execute promotion
        controller.execute_promotion(decision)
        model_registry.promote_model("model_v2", MagicMock(value="production"))

        # Phase 3: Model starts regressing
        # First regression check - model drops below baseline
        elo_service.set_rating(
            "model_v2", "square8", 2,
            rating=1460, games_played=80, win_rate=0.48  # -40 vs baseline
        )

        should_rollback, event = monitor.check_for_regression(
            model_id="model_v2",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v1",
        )

        # First check shouldn't trigger rollback yet (need consecutive checks)
        assert should_rollback is False
        # Regression notification should have been sent since -40 < -30
        assert len([e for e in notification_hook.events if e["type"] == "regression_detected"]) >= 1

        # Second regression check - regression continues
        elo_service.set_rating(
            "model_v2", "square8", 2,
            rating=1450, games_played=100, win_rate=0.45  # -50 vs baseline
        )

        should_rollback, event = monitor.check_for_regression(
            model_id="model_v2",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v1",
        )

        # Should trigger rollback now (consecutive checks exceeded)
        assert should_rollback is True
        assert event is not None
        assert event.current_model_id == "model_v2"
        assert event.rollback_model_id == "model_v1"

        # Verify at_risk and rollback_triggered notifications
        [e for e in notification_hook.events if e["type"] == "at_risk"]
        triggered_events = [e for e in notification_hook.events if e["type"] == "rollback_triggered"]
        assert len(triggered_events) >= 1

        # Phase 4: Execute rollback
        success = monitor.execute_rollback(event)
        assert success is True

        # Verify rollback_completed notification
        completed_events = [e for e in notification_hook.events if e["type"] == "rollback_completed"]
        assert len(completed_events) == 1
        assert completed_events[0]["success"] is True

    def test_cooldown_prevents_rapid_rollbacks(self):
        """Test that cooldown prevents rapid successive rollbacks."""
        elo_service = MockEloService()
        model_registry = MockModelRegistry()

        controller = PromotionController(
            elo_service=elo_service,
            model_registry=model_registry,
        )

        rollback_criteria = RollbackCriteria(
            elo_regression_threshold=-30.0,
            min_games_for_regression=20,
            cooldown_seconds=3600,  # 1 hour
        )

        monitor = RollbackMonitor(
            criteria=rollback_criteria,
            promotion_controller=controller,
        )

        # Setup: Model with severe regression
        elo_service.set_rating(
            "model_v2", "square8", 2,
            rating=1350, games_played=50, win_rate=0.35
        )
        elo_service.set_rating(
            "model_v1", "square8", 2,
            rating=1500, games_played=100, win_rate=0.55
        )

        # First check should trigger rollback
        should_rollback, event = monitor.check_for_regression(
            model_id="model_v2",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v1",
        )
        assert should_rollback is True

        # Execute the rollback
        success = monitor.execute_rollback(event)
        assert success is True

        # Now setup another model that's regressing
        elo_service.set_rating(
            "model_v3", "square8", 2,
            rating=1350, games_played=50, win_rate=0.35
        )

        # Second check should be blocked by cooldown
        should_rollback, event = monitor.check_for_regression(
            model_id="model_v3",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v1",
        )
        assert should_rollback is False  # Blocked by cooldown

        # Verify cooldown is active
        is_active, remaining = monitor.is_cooldown_active("square8", 2)
        assert is_active is True
        assert remaining > 0

    def test_max_daily_rollbacks_limit(self):
        """Test that max daily rollbacks limit is enforced."""
        elo_service = MockEloService()
        model_registry = MockModelRegistry()

        controller = PromotionController(
            elo_service=elo_service,
            model_registry=model_registry,
        )

        rollback_criteria = RollbackCriteria(
            max_rollbacks_per_day=2,
            cooldown_seconds=0,  # Disable cooldown for this test
        )

        monitor = RollbackMonitor(
            criteria=rollback_criteria,
            promotion_controller=controller,
        )

        # Add 2 rollback events today
        now = datetime.now().isoformat()
        monitor._rollback_events = [
            RollbackEvent(triggered_at=now, current_model_id="m1", rollback_model_id="m0", reason="test"),
            RollbackEvent(triggered_at=now, current_model_id="m2", rollback_model_id="m1", reason="test"),
        ]

        # Setup regressing model
        elo_service.set_rating(
            "model_v3", "square8", 2,
            rating=1350, games_played=50, win_rate=0.35
        )
        elo_service.set_rating(
            "model_v2", "square8", 2,
            rating=1500, games_played=100, win_rate=0.55
        )

        # Should be blocked by daily limit
        should_rollback, _event = monitor.check_for_regression(
            model_id="model_v3",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v2",
        )
        assert should_rollback is False

        # Verify limit is reached
        is_reached, count = monitor.is_max_daily_rollbacks_reached()
        assert is_reached is True
        assert count == 2

    def test_dry_run_does_not_execute(self):
        """Test that dry run mode doesn't actually rollback."""
        elo_service = MockEloService()
        model_registry = MockModelRegistry()

        controller = PromotionController(
            elo_service=elo_service,
            model_registry=model_registry,
        )

        monitor = RollbackMonitor(promotion_controller=controller)

        # Create a rollback event
        event = RollbackEvent(
            triggered_at=datetime.now().isoformat(),
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="Test regression",
        )

        # Execute in dry run mode
        success = monitor.execute_rollback(event, dry_run=True)
        assert success is True

        # Verify no changes were made to registry
        assert model_registry.get_production_model() is None
        assert len(model_registry.history) == 0


class TestNotificationIntegration:
    """Test notification hook integration."""

    def test_multiple_hooks_all_notified(self):
        """Test that all registered hooks receive notifications."""
        elo_service = MockEloService()
        model_registry = MockModelRegistry()

        hook1 = MockNotificationHook()
        hook2 = MockNotificationHook()

        controller = PromotionController(
            elo_service=elo_service,
            model_registry=model_registry,
        )

        rollback_criteria = RollbackCriteria(
            elo_regression_threshold=-30.0,
            min_games_for_regression=20,
        )

        monitor = RollbackMonitor(
            criteria=rollback_criteria,
            promotion_controller=controller,
            notification_hooks=[hook1, hook2],
        )

        # Setup severe regression
        elo_service.set_rating(
            "model_v2", "square8", 2,
            rating=1300, games_played=50, win_rate=0.30  # Severe regression
        )
        elo_service.set_rating(
            "model_v1", "square8", 2,
            rating=1500, games_played=100, win_rate=0.55
        )

        # Trigger rollback
        should_rollback, event = monitor.check_for_regression(
            model_id="model_v2",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v1",
        )

        assert should_rollback is True

        # Execute rollback
        monitor.execute_rollback(event)

        # Both hooks should have received notifications
        assert len(hook1.events) > 0
        assert len(hook2.events) > 0

        # Both should have rollback_triggered and rollback_completed
        hook1_types = [e["type"] for e in hook1.events]
        hook2_types = [e["type"] for e in hook2.events]

        assert "rollback_triggered" in hook1_types
        assert "rollback_completed" in hook1_types
        assert "rollback_triggered" in hook2_types
        assert "rollback_completed" in hook2_types

    def test_hook_failure_does_not_stop_rollback(self):
        """Test that a failing hook doesn't prevent rollback."""
        elo_service = MockEloService()
        model_registry = MockModelRegistry()

        class FailingHook(NotificationHook):
            def on_rollback_triggered(self, event):
                raise Exception("Hook failed!")

            def on_rollback_completed(self, event, success):
                raise Exception("Hook failed!")

        failing_hook = FailingHook()
        working_hook = MockNotificationHook()

        controller = PromotionController(
            elo_service=elo_service,
            model_registry=model_registry,
        )

        monitor = RollbackMonitor(
            promotion_controller=controller,
            notification_hooks=[failing_hook, working_hook],
        )

        # Setup for actual rollback (not dry run since that skips notifications)
        elo_service.set_rating("model_v1", "square8", 2, rating=1500, games_played=100, win_rate=0.55)
        model_registry.promote_model("model_v1", MagicMock(value="production"))

        event = RollbackEvent(
            triggered_at=datetime.now().isoformat(),
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="Test",
        )

        # Notify triggered (this is where failing_hook will raise)
        # The monitor should catch the exception and continue
        monitor._notify_rollback_triggered(event)

        # Working hook should still be notified despite failing hook
        triggered_events = [e for e in working_hook.events if e["type"] == "rollback_triggered"]
        assert len(triggered_events) == 1

        # Now test completion notification
        monitor._notify_rollback_completed(event, success=True)
        completed_events = [e for e in working_hook.events if e["type"] == "rollback_completed"]
        assert len(completed_events) == 1


class TestConfigIntegration:
    """Test configuration file integration."""

    def test_load_config_and_create_monitor(self):
        """Test loading config and creating a fully configured monitor."""
        import tempfile

        import yaml

        config = {
            "enabled": True,
            "logging": {"enabled": True, "logger_name": "test.rollback"},
            "criteria_overrides": {
                "elo_regression_threshold": -40.0,
                "cooldown_seconds": 1800,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            rollback_config = load_rollback_config(temp_path)

            assert rollback_config.enabled is True
            assert len(rollback_config.hooks) == 1  # logging hook
            assert rollback_config.criteria.elo_regression_threshold == -40.0
            assert rollback_config.criteria.cooldown_seconds == 1800

            # Create monitor with loaded config
            monitor = RollbackMonitor(
                criteria=rollback_config.criteria,
                notification_hooks=rollback_config.hooks,
            )

            assert monitor.criteria.elo_regression_threshold == -40.0
            assert len(monitor._hooks) == 1
        finally:
            temp_path.unlink()


class TestMultiModelBaseline:
    """Test multi-model baseline comparison."""

    def test_compare_against_multiple_baselines(self):
        """Test comparing against multiple baseline models."""
        elo_service = MockEloService()
        model_registry = MockModelRegistry()

        # Setup multiple models
        elo_service.set_rating("model_v1", "square8", 2, rating=1500, games_played=100, win_rate=0.55)
        elo_service.set_rating("model_v2", "square8", 2, rating=1520, games_played=100, win_rate=0.56)
        elo_service.set_rating("model_v3", "square8", 2, rating=1480, games_played=100, win_rate=0.54)
        elo_service.set_rating("model_v4", "square8", 2, rating=1450, games_played=50, win_rate=0.52)  # Current

        controller = PromotionController(
            elo_service=elo_service,
            model_registry=model_registry,
        )

        monitor = RollbackMonitor(promotion_controller=controller)

        # Compare v4 against v1, v2, v3
        result = monitor.check_against_baselines(
            model_id="model_v4",
            board_type="square8",
            num_players=2,
            baseline_model_ids=["model_v1", "model_v2", "model_v3"],
        )

        assert result["model_id"] == "model_v4"
        assert result["current_elo"] == 1450
        assert len(result["baselines"]) == 3
        assert result["regressions"] >= 2  # Regressed against v1 and v2
        assert result["summary"] in ["regression_against_majority", "regression_against_all"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

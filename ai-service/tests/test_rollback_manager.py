"""Tests for RollbackManager.

Tests rollback detection, execution, history tracking, and
performance monitoring capabilities.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.training.model_registry import (
    ModelRegistry,
    ModelStage,
    ModelMetrics,
)
from app.training.rollback_manager import (
    RollbackManager,
    RollbackThresholds,
    RollbackEvent,
)


class TestRollbackThresholds:
    """Test RollbackThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RollbackThresholds()
        assert thresholds.elo_drop_threshold == 50.0
        assert thresholds.min_games_for_evaluation == 50
        assert thresholds.win_rate_drop_threshold == 0.10

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = RollbackThresholds(
            elo_drop_threshold=30.0,
            min_games_for_evaluation=25,
        )
        assert thresholds.elo_drop_threshold == 30.0
        assert thresholds.min_games_for_evaluation == 25


class TestRollbackEvent:
    """Test RollbackEvent dataclass."""

    def test_event_serialization(self):
        """Test event serialization roundtrip."""
        event = RollbackEvent(
            model_id="test_model",
            from_version=2,
            to_version=1,
            reason="Performance regression",
            triggered_by="auto_elo",
            timestamp="2024-01-01T12:00:00",
            from_metrics={"elo": 1450},
            to_metrics={"elo": 1500},
            success=True,
        )

        d = event.to_dict()
        restored = RollbackEvent.from_dict(d)

        assert restored.model_id == event.model_id
        assert restored.from_version == event.from_version
        assert restored.to_version == event.to_version
        assert restored.success == event.success


class TestRollbackManager:
    """Test RollbackManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.registry = ModelRegistry(self.registry_dir)
        self.manager = RollbackManager(
            self.registry,
            thresholds=RollbackThresholds(
                elo_drop_threshold=30,
                min_games_for_evaluation=10,
            ),
            history_path=Path(self.temp_dir) / "rollback_history.json",
        )

        # Create test model file
        self.model_path = Path(self.temp_dir) / "model.pt"
        self.model_path.write_bytes(b"test model" * 100)

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_baseline(self):
        """Test setting performance baseline."""
        self.manager.set_baseline("test_model", {
            "elo": 1500,
            "win_rate": 0.55,
            "games_played": 100,
        })

        assert "test_model" in self.manager._baselines
        assert self.manager._baselines["test_model"]["elo"] == 1500

    def test_check_performance_within_range(self):
        """Test performance check when within acceptable range."""
        self.manager.set_baseline("test_model", {
            "elo": 1500,
            "games_played": 100,
        })

        is_degraded, reason = self.manager.check_performance(
            "test_model",
            {"elo": 1490, "games_played": 150},  # Only 10 point drop
        )

        assert not is_degraded
        assert "within acceptable range" in reason

    def test_check_performance_degraded_elo(self):
        """Test performance check when Elo degraded."""
        self.manager.set_baseline("test_model", {
            "elo": 1500,
            "games_played": 100,
        })

        is_degraded, reason = self.manager.check_performance(
            "test_model",
            {"elo": 1460, "games_played": 150},  # 40 point drop
        )

        assert is_degraded
        assert "Elo dropped" in reason

    def test_check_performance_degraded_win_rate(self):
        """Test performance check when win rate degraded."""
        self.manager.set_baseline("test_model", {
            "elo": 1500,
            "win_rate": 0.55,
            "games_played": 100,
        })

        is_degraded, reason = self.manager.check_performance(
            "test_model",
            {"elo": 1495, "win_rate": 0.40, "games_played": 150},  # 15% drop
        )

        assert is_degraded
        assert "Win rate dropped" in reason

    def test_check_performance_insufficient_games(self):
        """Test that performance check requires minimum games."""
        self.manager.set_baseline("test_model", {
            "elo": 1500,
            "games_played": 100,
        })

        is_degraded, reason = self.manager.check_performance(
            "test_model",
            {"elo": 1400, "games_played": 105},  # Only 5 games since baseline
        )

        assert not is_degraded
        assert "Insufficient games" in reason

    def test_get_rollback_candidate(self):
        """Test finding rollback candidate."""
        # Create and archive a model
        model_id, v1 = self.registry.register_model(
            name="Test Model",
            model_path=self.model_path,
            metrics=ModelMetrics(elo=1500),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="candidate_test",
        )
        self.registry.promote(model_id, v1, ModelStage.STAGING)
        self.registry.promote(model_id, v1, ModelStage.PRODUCTION)
        self.registry.promote(model_id, v1, ModelStage.ARCHIVED)

        candidate = self.manager.get_rollback_candidate("candidate_test")

        assert candidate is not None
        assert candidate["model_id"] == "candidate_test"
        assert candidate["version"] == 1

    def test_get_rollback_candidate_none(self):
        """Test when no rollback candidate exists."""
        candidate = self.manager.get_rollback_candidate("nonexistent_model")
        assert candidate is None

    def test_should_rollback_no_production(self):
        """Test should_rollback when model not in production."""
        should, reason = self.manager.should_rollback("test_model")
        assert not should
        assert "not in production" in reason

    def test_rollback_model_success(self):
        """Test successful rollback execution."""
        # Setup: Create v1, promote to production, create v2, promote to production
        m1_id, m1_v = self.registry.register_model(
            name="Rollback Test",
            model_path=self.model_path,
            metrics=ModelMetrics(elo=1500),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="rollback_test",
        )
        self.registry.promote(m1_id, m1_v, ModelStage.STAGING)
        self.registry.promote(m1_id, m1_v, ModelStage.PRODUCTION)

        # Create v2
        model_path2 = Path(self.temp_dir) / "model2.pt"
        model_path2.write_bytes(b"model 2" * 100)
        m2_id, m2_v = self.registry.register_model(
            name="Rollback Test",
            model_path=model_path2,
            metrics=ModelMetrics(elo=1450),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="rollback_test",
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)
        self.registry.promote(m2_id, m2_v, ModelStage.PRODUCTION)

        # v1 should now be archived
        v1_model = self.registry.get_model(m1_id, m1_v)
        assert v1_model.stage == ModelStage.ARCHIVED

        # Perform rollback
        result = self.manager.rollback_model(
            "rollback_test",
            to_version=m1_v,
            reason="Performance regression",
        )

        assert result["success"]
        assert result["from_version"] == m2_v
        assert result["to_version"] == m1_v

        # v1 should be back in production
        v1_after = self.registry.get_model(m1_id, m1_v)
        assert v1_after.stage == ModelStage.PRODUCTION

    def test_rollback_model_not_found(self):
        """Test rollback when model doesn't exist."""
        result = self.manager.rollback_model(
            "nonexistent_model",
            reason="Test",
        )
        assert not result["success"]
        assert "error" in result

    def test_rollback_history_tracking(self):
        """Test that rollback history is tracked."""
        # Setup rollback scenario
        m1_id, m1_v = self.registry.register_model(
            name="History Test",
            model_path=self.model_path,
            metrics=ModelMetrics(elo=1500),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="history_test",
        )
        self.registry.promote(m1_id, m1_v, ModelStage.STAGING)
        self.registry.promote(m1_id, m1_v, ModelStage.PRODUCTION)

        model_path2 = Path(self.temp_dir) / "model2.pt"
        model_path2.write_bytes(b"model 2" * 100)
        m2_id, m2_v = self.registry.register_model(
            name="History Test",
            model_path=model_path2,
            metrics=ModelMetrics(elo=1450),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="history_test",
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)
        self.registry.promote(m2_id, m2_v, ModelStage.PRODUCTION)

        # Perform rollback
        self.manager.rollback_model("history_test", to_version=m1_v)

        # Check history
        history = self.manager.get_rollback_history()
        assert len(history) == 1
        assert history[0].model_id == "history_test"
        assert history[0].success

    def test_rollback_stats(self):
        """Test rollback statistics."""
        # Initially empty
        stats = self.manager.get_rollback_stats()
        assert stats["total_rollbacks"] == 0

        # After a rollback
        m1_id, m1_v = self.registry.register_model(
            name="Stats Test",
            model_path=self.model_path,
            metrics=ModelMetrics(elo=1500),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="stats_test",
        )
        self.registry.promote(m1_id, m1_v, ModelStage.STAGING)
        self.registry.promote(m1_id, m1_v, ModelStage.PRODUCTION)

        model_path2 = Path(self.temp_dir) / "model2.pt"
        model_path2.write_bytes(b"model 2" * 100)
        m2_id, m2_v = self.registry.register_model(
            name="Stats Test",
            model_path=model_path2,
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="stats_test",
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)
        self.registry.promote(m2_id, m2_v, ModelStage.PRODUCTION)

        self.manager.rollback_model("stats_test", to_version=m1_v, triggered_by="manual")

        stats = self.manager.get_rollback_stats()
        assert stats["total_rollbacks"] == 1
        assert stats["successful"] == 1
        assert stats["by_trigger"]["manual"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

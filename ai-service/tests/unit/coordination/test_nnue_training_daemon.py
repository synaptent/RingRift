"""Unit tests for NNUETrainingDaemon.

December 2025: Tests for NNUE automatic training daemon.
Covers configuration, state management, training logic, and health checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.nnue_training_daemon import (
    NNUETrainingConfig,
    NNUETrainingDaemon,
    NNUETrainingState,
    get_nnue_training_daemon,
)


@pytest.fixture
def reset_daemon():
    """Reset daemon singleton before and after each test."""
    NNUETrainingDaemon.reset_instance()
    yield
    NNUETrainingDaemon.reset_instance()


@pytest.fixture
def config():
    """Create a test configuration."""
    return NNUETrainingConfig(
        game_thresholds={
            "hex8_2p": 100,
            "hex8_4p": 200,
        },
        default_game_threshold=150,
        check_interval_seconds=60.0,
        max_concurrent_trainings=2,
        min_time_between_trainings=60.0,
        training_timeout_seconds=120.0,
    )


@pytest.fixture
def daemon(reset_daemon, config, tmp_path):
    """Create a daemon with test config and temp state path."""
    daemon = NNUETrainingDaemon(config=config)
    daemon._state_path = tmp_path / "nnue_state.json"
    return daemon


# =============================================================================
# NNUETrainingConfig Tests
# =============================================================================


class TestNNUETrainingConfig:
    """Tests for NNUETrainingConfig dataclass."""

    def test_default_thresholds(self):
        """Default config has thresholds for all 12 canonical configs."""
        config = NNUETrainingConfig()

        assert "hex8_2p" in config.game_thresholds
        assert "hex8_4p" in config.game_thresholds
        assert "square8_2p" in config.game_thresholds
        assert "square19_2p" in config.game_thresholds
        assert "hexagonal_4p" in config.game_thresholds

    def test_get_threshold_known_config(self, config):
        """get_threshold returns configured value for known configs."""
        assert config.get_threshold("hex8_2p") == 100
        assert config.get_threshold("hex8_4p") == 200

    def test_get_threshold_unknown_config(self, config):
        """get_threshold returns default for unknown configs."""
        assert config.get_threshold("unknown_config") == 150

    def test_default_values(self):
        """Default config has expected default values."""
        config = NNUETrainingConfig()

        assert config.default_game_threshold == 5000
        assert config.check_interval_seconds == 3600.0
        assert config.max_concurrent_trainings == 2
        assert config.min_time_between_trainings == 3600.0
        assert config.training_timeout_seconds == 7200.0

    def test_custom_values(self, config):
        """Custom config overrides defaults."""
        assert config.default_game_threshold == 150
        assert config.check_interval_seconds == 60.0
        assert config.max_concurrent_trainings == 2


# =============================================================================
# NNUETrainingState Tests
# =============================================================================


class TestNNUETrainingState:
    """Tests for NNUETrainingState dataclass."""

    def test_empty_state(self):
        """Empty state has empty dictionaries."""
        state = NNUETrainingState()

        assert state.last_training_time == {}
        assert state.last_training_game_count == {}
        assert state.active_trainings == {}
        assert state.training_history == []

    def test_to_dict(self):
        """to_dict serializes state correctly."""
        state = NNUETrainingState(
            last_training_time={"hex8_2p": 1000.0},
            last_training_game_count={"hex8_2p": 500},
            active_trainings={"hex8_4p": 2000.0},
            training_history=[{"config_key": "hex8_2p", "success": True}],
        )

        data = state.to_dict()

        assert data["last_training_time"] == {"hex8_2p": 1000.0}
        assert data["last_training_game_count"] == {"hex8_2p": 500}
        assert data["active_trainings"] == {"hex8_4p": 2000.0}
        assert len(data["training_history"]) == 1

    def test_to_dict_truncates_history(self):
        """to_dict keeps only last 100 history entries."""
        state = NNUETrainingState(
            training_history=[{"i": i} for i in range(150)]
        )

        data = state.to_dict()

        assert len(data["training_history"]) == 100
        # Should keep the last 100
        assert data["training_history"][0]["i"] == 50

    def test_from_dict(self):
        """from_dict deserializes state correctly."""
        data = {
            "last_training_time": {"hex8_2p": 1000.0},
            "last_training_game_count": {"hex8_2p": 500},
            "active_trainings": {"hex8_4p": 2000.0},
            "training_history": [{"config_key": "hex8_2p", "success": True}],
        }

        state = NNUETrainingState.from_dict(data)

        assert state.last_training_time == {"hex8_2p": 1000.0}
        assert state.last_training_game_count == {"hex8_2p": 500}
        assert state.active_trainings == {"hex8_4p": 2000.0}
        assert len(state.training_history) == 1

    def test_from_dict_empty(self):
        """from_dict handles empty data."""
        state = NNUETrainingState.from_dict({})

        assert state.last_training_time == {}
        assert state.last_training_game_count == {}
        assert state.active_trainings == {}
        assert state.training_history == []

    def test_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        original = NNUETrainingState(
            last_training_time={"hex8_2p": 1000.0, "hex8_4p": 2000.0},
            last_training_game_count={"hex8_2p": 500, "hex8_4p": 1000},
            active_trainings={},
            training_history=[
                {"config_key": "hex8_2p", "success": True},
                {"config_key": "hex8_4p", "success": False},
            ],
        )

        restored = NNUETrainingState.from_dict(original.to_dict())

        assert restored.last_training_time == original.last_training_time
        assert restored.last_training_game_count == original.last_training_game_count
        assert restored.training_history == original.training_history


# =============================================================================
# NNUETrainingDaemon Initialization Tests
# =============================================================================


class TestNNUETrainingDaemonInit:
    """Tests for daemon initialization."""

    def test_singleton_pattern(self, reset_daemon):
        """get_instance returns the same instance."""
        daemon1 = NNUETrainingDaemon.get_instance()
        daemon2 = NNUETrainingDaemon.get_instance()

        assert daemon1 is daemon2

    def test_reset_instance(self, reset_daemon):
        """reset_instance creates new instance."""
        daemon1 = NNUETrainingDaemon.get_instance()
        NNUETrainingDaemon.reset_instance()
        daemon2 = NNUETrainingDaemon.get_instance()

        assert daemon1 is not daemon2

    def test_custom_config(self, reset_daemon, config):
        """Custom config is applied to daemon."""
        daemon = NNUETrainingDaemon.get_instance(config=config)

        assert daemon._nnue_config.check_interval_seconds == 60.0
        assert daemon._nnue_config.get_threshold("hex8_2p") == 100

    def test_default_config(self, reset_daemon):
        """Default config is used when none provided."""
        daemon = NNUETrainingDaemon.get_instance()

        assert daemon._nnue_config.default_game_threshold == 5000
        assert daemon._nnue_config.check_interval_seconds == 3600.0

    def test_inherits_handler_base(self, daemon):
        """Daemon inherits from HandlerBase."""
        from app.coordination.handler_base import HandlerBase

        assert isinstance(daemon, HandlerBase)
        assert daemon._name == "nnue_training"


# =============================================================================
# Event Subscriptions Tests
# =============================================================================


class TestEventSubscriptions:
    """Tests for event subscription handling."""

    def test_event_subscriptions(self, daemon):
        """Daemon subscribes to expected events."""
        subs = daemon._get_event_subscriptions()

        assert "NEW_GAMES_AVAILABLE" in subs
        assert "CONSOLIDATION_COMPLETE" in subs
        assert "DATA_SYNC_COMPLETED" in subs
        assert "TRAINING_COMPLETED" in subs

    @pytest.mark.asyncio
    async def test_on_new_games_updates_count(self, daemon):
        """NEW_GAMES_AVAILABLE updates game count."""
        event = {"config_key": "hex8_2p", "game_count": 500}

        await daemon._on_new_games(event)

        assert daemon._current_game_counts.get("hex8_2p") == 500

    @pytest.mark.asyncio
    async def test_on_new_games_ignores_empty(self, daemon):
        """NEW_GAMES_AVAILABLE ignores empty events."""
        event = {}

        await daemon._on_new_games(event)

        assert daemon._current_game_counts == {}


# =============================================================================
# Training Logic Tests
# =============================================================================


class TestShouldTrain:
    """Tests for _should_train logic."""

    def test_should_train_threshold_met(self, daemon):
        """Training triggers when threshold is met."""
        # Config has threshold of 100 for hex8_2p
        daemon._state.last_training_game_count["hex8_2p"] = 0

        result = daemon._should_train("hex8_2p", 100)

        assert result is True

    def test_should_not_train_threshold_not_met(self, daemon):
        """Training doesn't trigger when threshold not met."""
        daemon._state.last_training_game_count["hex8_2p"] = 0

        result = daemon._should_train("hex8_2p", 50)

        assert result is False

    def test_should_not_train_already_active(self, daemon):
        """Training doesn't trigger when already active."""
        daemon._state.active_trainings["hex8_2p"] = time.time()

        result = daemon._should_train("hex8_2p", 500)

        assert result is False

    def test_should_not_train_too_soon(self, daemon):
        """Training doesn't trigger if trained recently."""
        daemon._state.last_training_time["hex8_2p"] = time.time()
        daemon._state.last_training_game_count["hex8_2p"] = 0

        result = daemon._should_train("hex8_2p", 500)

        assert result is False

    def test_should_train_after_cooldown(self, daemon):
        """Training triggers after cooldown period."""
        daemon._state.last_training_time["hex8_2p"] = time.time() - 120
        daemon._state.last_training_game_count["hex8_2p"] = 0

        result = daemon._should_train("hex8_2p", 100)

        assert result is True

    def test_should_train_incremental_games(self, daemon):
        """Training uses incremental game count."""
        daemon._state.last_training_game_count["hex8_2p"] = 1000

        # Only 50 new games since last training
        result = daemon._should_train("hex8_2p", 1050)
        assert result is False

        # 100 new games since last training
        result = daemon._should_train("hex8_2p", 1100)
        assert result is True


class TestCleanupActiveTrainings:
    """Tests for _cleanup_active_trainings."""

    def test_cleanup_removes_timed_out(self, daemon):
        """Cleanup removes timed-out trainings."""
        # Training started 200 seconds ago, timeout is 120
        daemon._state.active_trainings["hex8_2p"] = time.time() - 200

        daemon._cleanup_active_trainings()

        assert "hex8_2p" not in daemon._state.active_trainings

    def test_cleanup_keeps_active(self, daemon):
        """Cleanup keeps non-timed-out trainings."""
        daemon._state.active_trainings["hex8_2p"] = time.time() - 30

        daemon._cleanup_active_trainings()

        assert "hex8_2p" in daemon._state.active_trainings


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state load/save."""

    def test_load_state_file_not_exists(self, daemon, tmp_path):
        """Load handles missing state file."""
        daemon._state_path = tmp_path / "nonexistent.json"

        daemon._load_state()

        # State should remain empty
        assert daemon._state.last_training_time == {}

    def test_save_and_load_state(self, daemon, tmp_path):
        """State can be saved and loaded."""
        daemon._state_path = tmp_path / "state.json"
        daemon._state.last_training_time["hex8_2p"] = 1000.0
        daemon._state.last_training_game_count["hex8_2p"] = 500

        daemon._save_state()

        # Create new daemon and load state
        daemon2 = NNUETrainingDaemon(config=daemon._nnue_config)
        daemon2._state_path = tmp_path / "state.json"
        daemon2._load_state()

        assert daemon2._state.last_training_time["hex8_2p"] == 1000.0
        assert daemon2._state.last_training_game_count["hex8_2p"] == 500

    def test_load_state_handles_corrupt_json(self, daemon, tmp_path):
        """Load handles corrupt JSON gracefully."""
        state_path = tmp_path / "corrupt.json"
        state_path.write_text("not valid json {{{")
        daemon._state_path = state_path

        daemon._load_state()

        # State should remain empty
        assert daemon._state.last_training_time == {}


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_when_running(self, daemon):
        """Health check returns healthy when running."""
        daemon._running = True
        daemon._current_game_counts = {"hex8_2p": 100}

        result = daemon.health_check()

        assert result.healthy is True
        assert result.details["configs_tracked"] == 1

    def test_health_check_when_stopped(self, daemon):
        """Health check returns unhealthy when stopped."""
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False

    def test_health_check_includes_active_trainings(self, daemon):
        """Health check includes active training count."""
        daemon._running = True
        daemon._state.active_trainings["hex8_2p"] = time.time()

        result = daemon.health_check()

        assert result.details["active_trainings"] == 1

    def test_health_check_includes_last_training(self, daemon):
        """Health check includes last training info."""
        daemon._running = True
        daemon._state.training_history.append({
            "config_key": "hex8_2p",
            "success": True,
            "timestamp": 1000.0,
        })

        result = daemon.health_check()

        assert "last_training" in result.details
        assert result.details["last_training"]["config"] == "hex8_2p"


# =============================================================================
# Training Stats Tests
# =============================================================================


class TestTrainingStats:
    """Tests for get_training_stats method."""

    def test_empty_stats(self, daemon):
        """Stats are empty for new daemon."""
        stats = daemon.get_training_stats()

        assert stats["total_trainings"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["success_rate"] == 0.0

    def test_stats_with_history(self, daemon):
        """Stats reflect training history."""
        daemon._state.training_history = [
            {"config_key": "hex8_2p", "success": True},
            {"config_key": "hex8_4p", "success": True},
            {"config_key": "square8_2p", "success": False},
        ]

        stats = daemon.get_training_stats()

        assert stats["total_trainings"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1
        assert stats["success_rate"] == pytest.approx(2 / 3)

    def test_stats_includes_active(self, daemon):
        """Stats include active trainings."""
        daemon._state.active_trainings["hex8_2p"] = time.time()
        daemon._current_game_counts["hex8_2p"] = 100

        stats = daemon.get_training_stats()

        assert "hex8_2p" in stats["active_trainings"]
        assert "hex8_2p" in stats["configs_tracked"]


# =============================================================================
# Module Functions Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_nnue_training_daemon(self, reset_daemon):
        """get_nnue_training_daemon returns singleton."""
        daemon1 = get_nnue_training_daemon()
        daemon2 = get_nnue_training_daemon()

        assert daemon1 is daemon2

    def test_get_nnue_training_daemon_with_config(self, reset_daemon, config):
        """get_nnue_training_daemon uses provided config."""
        daemon = get_nnue_training_daemon(config=config)

        assert daemon._nnue_config.check_interval_seconds == 60.0

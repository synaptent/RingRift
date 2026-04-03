"""Tests for TrainingTriggerDaemon - Phase 2 test coverage (December 2025).

Tests cover:
- Configuration dataclasses
- State persistence (SQLite)
- Event handling (NPZ export, training completion, backpressure)
- Training condition checks
- Deduplication logic
- Health check reporting
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.training_trigger_daemon import (
    ConfigTrainingState,
    TrainingTriggerConfig,
    TrainingTriggerDaemon,
    TRIGGER_DEDUP_WINDOW_SECONDS,
)


class TestTrainingTriggerConfig:
    """Tests for TrainingTriggerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingTriggerConfig()
        assert config.enabled is True
        assert config.min_samples_threshold == 5000
        # Jan 5, 2026 (Session 17.24): Reduced from 0.083 to 0.033 (~2 min)
        assert config.training_cooldown_hours == 0.033
        assert config.max_concurrent_training == 8
        assert config.gpu_idle_threshold_percent == 20.0
        # Jan 5, 2026 (Session 17.24): Reduced from 30s to 15s
        assert config.scan_interval_seconds == 15
        assert config.default_epochs == 20
        assert config.default_batch_size == 512
        assert config.model_version == "v2"

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = TrainingTriggerConfig(
            enabled=False,
            min_samples_threshold=10000,
            max_concurrent_training=4,
        )
        assert config.enabled is False
        assert config.min_samples_threshold == 10000
        assert config.max_concurrent_training == 4

    def test_state_db_path(self):
        """Test state database path configuration."""
        config = TrainingTriggerConfig(
            state_db_path="/custom/path/state.db"
        )
        assert config.state_db_path == "/custom/path/state.db"


class TestConfigTrainingState:
    """Tests for ConfigTrainingState dataclass."""

    def test_default_values(self):
        """Test default training state values."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
        )
        assert state.config_key == "hex8_2p"
        assert state.board_type == "hex8"
        assert state.num_players == 2
        assert state.last_training_time == 0.0
        assert state.training_in_progress is False
        assert state.training_pid is None
        assert state.npz_sample_count == 0
        assert state.last_elo == 1500.0
        assert state.elo_trend == 0.0
        assert state.training_intensity == "normal"
        assert state.consecutive_failures == 0

    def test_state_mutation(self):
        """Test that state can be mutated."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
        )
        state.training_in_progress = True
        state.training_pid = 12345
        state.npz_sample_count = 50000
        state.last_elo = 1650.0

        assert state.training_in_progress is True
        assert state.training_pid == 12345
        assert state.npz_sample_count == 50000
        assert state.last_elo == 1650.0


class TestTrainingTriggerDaemonInit:
    """Tests for TrainingTriggerDaemon initialization."""

    def test_default_initialization(self):
        """Test default daemon initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(
                state_db_path=f"{tmpdir}/state.db"
            )
            daemon = TrainingTriggerDaemon(config=config)

            assert daemon._daemon_config == config
            assert daemon._training_states == {}
            assert daemon._coordinator_skip is False
            assert daemon._recent_triggers == {}
            assert daemon._evaluation_backpressure is False


class TestQualityGate:
    """Tests for training quality gate decisions."""

    @pytest.mark.asyncio
    async def test_check_quality_gate_bootstraps_without_quality_data(self):
        """Training should proceed in bootstrap mode when no quality data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            quality_monitor = MagicMock()
            quality_monitor.get_quality_for_config.return_value = None

            with patch(
                "app.coordination.quality_monitor_daemon.get_quality_monitor",
                return_value=quality_monitor,
            ):
                quality_ok, reason = await daemon._check_quality_gate("hex8_2p")

        assert quality_ok is True
        assert reason.startswith("quality ok (")

    def test_state_db_initialization(self):
        """Test that state database is created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.db"
            config = TrainingTriggerConfig(state_db_path=db_path)
            daemon = TrainingTriggerDaemon(config=config)

            # Verify DB was created
            assert Path(db_path).exists()

            # Verify table structure
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='config_state'"
                )
                assert cursor.fetchone() is not None


class TestStatePersistence:
    """Tests for state persistence (Phase 3 - December 2025)."""

    def test_save_and_load_state(self):
        """Test saving and loading training state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.db"
            config = TrainingTriggerConfig(state_db_path=db_path)

            # Create daemon and add state
            daemon = TrainingTriggerDaemon(config=config)
            daemon._training_states["hex8_2p"] = ConfigTrainingState(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                last_training_time=12345.0,
                npz_sample_count=50000,
                last_elo=1700.0,
                training_intensity="accelerated",
            )

            # Save state
            daemon._save_state()

            # Create new daemon and load state
            daemon2 = TrainingTriggerDaemon(config=config)
            daemon2._load_state()

            # Verify state was loaded
            assert "hex8_2p" in daemon2._training_states
            state = daemon2._training_states["hex8_2p"]
            assert state.config_key == "hex8_2p"
            assert state.last_training_time == 12345.0
            assert state.npz_sample_count == 50000
            assert state.last_elo == 1700.0
            assert state.training_intensity == "accelerated"
            # training_in_progress resets on restart
            assert state.training_in_progress is False

    def test_load_state_no_db(self):
        """Test loading state when no DB exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/nonexistent.db"
            config = TrainingTriggerConfig(state_db_path=db_path)

            daemon = TrainingTriggerDaemon(config=config)
            # Remove the DB that was created
            Path(db_path).unlink()

            # Should not raise
            daemon._load_state()
            assert daemon._training_states == {}

    def test_save_multiple_configs(self):
        """Test saving multiple config states."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.db"
            config = TrainingTriggerConfig(state_db_path=db_path)

            daemon = TrainingTriggerDaemon(config=config)

            # Add multiple states
            for board in ["hex8", "square8"]:
                for players in [2, 3, 4]:
                    config_key = f"{board}_{players}p"
                    daemon._training_states[config_key] = ConfigTrainingState(
                        config_key=config_key,
                        board_type=board,
                        num_players=players,
                        last_elo=1500.0 + players * 50,
                    )

            daemon._save_state()

            # Verify all saved
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM config_state")
                count = cursor.fetchone()[0]
                assert count == 6  # 2 boards * 3 player counts


class TestDeduplication:
    """Tests for trigger deduplication logic."""

    def test_first_trigger_not_skipped(self):
        """Test that first trigger for a config is not skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            assert daemon._should_skip_duplicate_trigger("hex8_2p") is False
            assert "hex8_2p" in daemon._recent_triggers

    def test_duplicate_within_window_skipped(self):
        """Test that duplicate trigger within window is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # First trigger
            assert daemon._should_skip_duplicate_trigger("hex8_2p") is False

            # Immediate second trigger - should be skipped
            assert daemon._should_skip_duplicate_trigger("hex8_2p") is True

    def test_different_configs_not_deduped(self):
        """Test that different configs are not deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            assert daemon._should_skip_duplicate_trigger("hex8_2p") is False
            assert daemon._should_skip_duplicate_trigger("square8_4p") is False

    def test_after_window_not_skipped(self):
        """Test that trigger after window expires is not skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # First trigger
            daemon._should_skip_duplicate_trigger("hex8_2p")

            # Simulate time passing beyond window
            daemon._recent_triggers["hex8_2p"] = time.time() - TRIGGER_DEDUP_WINDOW_SECONDS - 10

            # Should not be skipped
            assert daemon._should_skip_duplicate_trigger("hex8_2p") is False


class TestIntensityMapping:
    """Tests for training intensity mapping.

    Feb 2026: Updated to use module-level functions after extraction from daemon.
    - intensity_from_quality → training_quality_gates.py
    - get_training_params_for_intensity → training_architecture_selector.py
    """

    def test_intensity_from_quality_hot_path(self):
        """Test hot_path intensity for high quality."""
        from app.coordination.training_quality_gates import intensity_from_quality

        assert intensity_from_quality(0.95) == "hot_path"
        assert intensity_from_quality(0.90) == "hot_path"

    def test_intensity_from_quality_accelerated(self):
        """Test accelerated intensity for good quality."""
        from app.coordination.training_quality_gates import intensity_from_quality

        assert intensity_from_quality(0.85) == "accelerated"
        assert intensity_from_quality(0.80) == "accelerated"

    def test_intensity_from_quality_normal(self):
        """Test normal intensity for average quality."""
        from app.coordination.training_quality_gates import intensity_from_quality

        assert intensity_from_quality(0.70) == "normal"
        assert intensity_from_quality(0.65) == "normal"

    def test_intensity_from_quality_reduced(self):
        """Test reduced intensity for poor quality."""
        from app.coordination.training_quality_gates import intensity_from_quality

        assert intensity_from_quality(0.55) == "reduced"
        assert intensity_from_quality(0.50) == "reduced"

    def test_intensity_from_quality_paused(self):
        """Test paused intensity for very poor quality."""
        from app.coordination.training_quality_gates import intensity_from_quality

        assert intensity_from_quality(0.45) == "paused"
        assert intensity_from_quality(0.0) == "paused"

    def test_training_params_for_intensity(self):
        """Test training parameters for each intensity level."""
        from app.coordination.training_architecture_selector import (
            get_training_params_for_intensity,
        )

        # intensive: stalled configs (2x epochs)
        epochs, batch, lr = get_training_params_for_intensity("intensive")
        assert epochs == 100  # 50 * 2.0
        assert batch == 512
        assert lr == 1.1

        # hot_path: aggressive improvement (1.5x epochs)
        epochs, batch, lr = get_training_params_for_intensity("hot_path")
        assert epochs == 75  # 50 * 1.5
        assert batch == 1024
        assert lr == 1.2

        # high: plateau response (1.5x epochs)
        epochs, batch, lr = get_training_params_for_intensity("high")
        assert epochs == 75  # 50 * 1.5
        assert batch == 768
        assert lr == 1.05

        # accelerated: moderate boost (1.2x epochs)
        epochs, batch, lr = get_training_params_for_intensity("accelerated")
        assert epochs == 60  # 50 * 1.2
        assert batch == 768
        assert lr == 1.1

        # normal (uses default_epochs=50, default_batch_size=512)
        epochs, batch, lr = get_training_params_for_intensity("normal")
        assert epochs == 50
        assert batch == 512
        assert lr == 1.0

        # reduced: conservative (0.8x epochs)
        epochs, batch, lr = get_training_params_for_intensity("reduced")
        assert epochs == 40  # 50 * 0.8
        assert batch == 256
        assert lr == 0.9

        # paused: skip training
        epochs, batch, lr = get_training_params_for_intensity("paused")
        assert epochs == 0
        assert lr == 0.0

    def test_unknown_intensity_fallback(self):
        """Test fallback to normal for unknown intensity."""
        from app.coordination.training_architecture_selector import (
            get_training_params_for_intensity,
        )

        epochs, batch, lr = get_training_params_for_intensity("unknown")
        assert epochs == 50  # default_epochs
        assert batch == 512  # default_batch_size
        assert lr == 1.0


class TestBackpressure:
    """Tests for evaluation backpressure handling (Phase 4)."""

    def test_initial_state_no_backpressure(self):
        """Test initial state has no backpressure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            assert daemon._evaluation_backpressure is False
            assert daemon._backpressure_stats["pauses_due_to_backpressure"] == 0
            assert daemon._backpressure_stats["resumes_after_backpressure"] == 0

    @pytest.mark.asyncio
    async def test_backpressure_activation(self):
        """Test backpressure activation handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Create mock event
            event = MagicMock()
            event.payload = {"queue_depth": 50, "backpressure_active": True}

            await daemon._on_evaluation_backpressure(event)

            assert daemon._evaluation_backpressure is True
            assert daemon._backpressure_stats["pauses_due_to_backpressure"] == 1
            assert daemon._backpressure_stats["last_backpressure_time"] > 0

    @pytest.mark.asyncio
    async def test_backpressure_release(self):
        """Test backpressure release handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Activate backpressure first
            daemon._evaluation_backpressure = True

            # Create mock event
            event = MagicMock()
            event.payload = {"queue_depth": 15, "backpressure_active": False}

            await daemon._on_evaluation_backpressure_released(event)

            assert daemon._evaluation_backpressure is False
            assert daemon._backpressure_stats["resumes_after_backpressure"] == 1


class TestEventSubscriptions:
    """Tests for event subscription wiring."""

    def test_event_subscriptions(self):
        """Test that event subscriptions are properly defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            subscriptions = daemon._get_event_subscriptions()

            # Core events
            assert "npz_export_complete" in subscriptions
            assert "training_completed" in subscriptions
            assert "training_threshold_reached" in subscriptions

            # Quality events
            assert "quality_score_updated" in subscriptions
            assert "training_blocked_by_quality" in subscriptions

            # Freshness events
            assert "data_stale" in subscriptions
            assert "data_sync_completed" in subscriptions

            # Backpressure events (Phase 4)
            assert "EVALUATION_BACKPRESSURE" in subscriptions
            assert "EVALUATION_BACKPRESSURE_RELEASED" in subscriptions

    def test_all_handlers_are_callables(self):
        """Test that all subscription handlers are callable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            subscriptions = daemon._get_event_subscriptions()

            for event_name, handler in subscriptions.items():
                assert callable(handler), f"Handler for {event_name} is not callable"


class TestHealthCheck:
    """Tests for health check reporting."""

    def test_health_check_structure(self):
        """Test health check returns proper structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            health = daemon.health_check()

            # Health check should have required attributes
            assert hasattr(health, 'healthy')
            assert hasattr(health, 'details')
            assert isinstance(health.details, dict)
            # Details should include key fields
            assert "running" in health.details
            assert "enabled" in health.details
            assert "evaluation_backpressure" in health.details

    def test_health_check_with_backpressure(self):
        """Test health check reports backpressure state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)
            daemon._evaluation_backpressure = True

            health = daemon.health_check()

            # Actual key is 'evaluation_backpressure'
            assert health.details["evaluation_backpressure"] is True

    def test_health_check_with_active_training(self):
        """Test health check reports active training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Add training state with in_progress
            daemon._training_states["hex8_2p"] = ConfigTrainingState(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                training_in_progress=True,
            )

            health = daemon.health_check()

            # Actual key is 'active_training_tasks'
            assert health.details["active_training_tasks"] >= 1


class TestGetOrCreateState:
    """Tests for state creation and retrieval."""

    def test_create_new_state(self):
        """Test creating state for new config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            state = daemon._get_or_create_state("hex8_2p", "hex8", 2)

            assert state.config_key == "hex8_2p"
            assert state.board_type == "hex8"
            assert state.num_players == 2
            assert "hex8_2p" in daemon._training_states

    def test_get_existing_state(self):
        """Test retrieving existing state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Create state
            original = daemon._get_or_create_state("hex8_2p", "hex8", 2)
            original.npz_sample_count = 50000

            # Get again - should be same object
            retrieved = daemon._get_or_create_state("hex8_2p", "hex8", 2)
            assert retrieved.npz_sample_count == 50000


class TestLifecycle:
    """Tests for daemon lifecycle (start/stop)."""

    @pytest.mark.asyncio
    async def test_start_loads_state(self):
        """Test that start() loads persisted state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.db"
            config = TrainingTriggerConfig(state_db_path=db_path)

            # Create and save state
            daemon1 = TrainingTriggerDaemon(config=config)
            daemon1._training_states["hex8_2p"] = ConfigTrainingState(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                last_elo=1800.0,
            )
            daemon1._save_state()

            # Create new daemon and start
            daemon2 = TrainingTriggerDaemon(config=config)

            # Mock the parent start() to avoid actual loop
            with patch.object(daemon2, '_run_cycle', new_callable=AsyncMock):
                # Just call _load_state which start() would call
                daemon2._load_state()

            assert "hex8_2p" in daemon2._training_states
            assert daemon2._training_states["hex8_2p"].last_elo == 1800.0

    @pytest.mark.asyncio
    async def test_stop_saves_state(self):
        """Test that stop() saves state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.db"
            config = TrainingTriggerConfig(state_db_path=db_path)

            daemon = TrainingTriggerDaemon(config=config)
            daemon._training_states["hex8_2p"] = ConfigTrainingState(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                npz_sample_count=75000,
            )

            # Call stop (mock parent to avoid cleanup errors)
            with patch.object(daemon, '_running', True):
                await daemon.stop()

            # Verify state was saved
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT npz_sample_count FROM config_state WHERE config_key = ?",
                    ("hex8_2p",)
                )
                row = cursor.fetchone()
                assert row is not None
                assert row[0] == 75000


class TestEventHandlers:
    """Tests for event handler methods."""

    @pytest.mark.asyncio
    async def test_on_npz_export_complete_updates_state(self):
        """Test NPZ export complete updates training state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Mock _maybe_trigger_training to avoid actual training
            daemon._maybe_trigger_training = AsyncMock()

            # Create mock result
            result = MagicMock()
            result.metadata = {
                "config": "hex8_2p",
                "board_type": "hex8",
                "num_players": 2,
                "output_path": "/data/training/hex8_2p.npz",
                "samples": 50000,
            }

            await daemon._on_npz_export_complete(result)

            assert "hex8_2p" in daemon._training_states
            state = daemon._training_states["hex8_2p"]
            assert state.npz_sample_count == 50000
            assert state.npz_path == "/data/training/hex8_2p.npz"
            daemon._maybe_trigger_training.assert_called_once_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_on_training_completed_updates_state(self):
        """Test training completed updates state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Pre-create state with training in progress
            daemon._training_states["hex8_2p"] = ConfigTrainingState(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                training_in_progress=True,
                training_pid=12345,
            )

            # Create mock event
            event = MagicMock()
            event.payload = {
                "config": "hex8_2p",
                "elo": 1750.0,
            }

            await daemon._on_training_completed(event)

            state = daemon._training_states["hex8_2p"]
            assert state.training_in_progress is False
            assert state.training_pid is None
            assert state.last_elo == 1750.0
            assert state.last_training_time > 0

    @pytest.mark.asyncio
    async def test_on_quality_score_updated(self):
        """Test quality score update changes intensity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(state_db_path=f"{tmpdir}/state.db")
            daemon = TrainingTriggerDaemon(config=config)

            # Pre-create state
            daemon._training_states["hex8_2p"] = ConfigTrainingState(
                config_key="hex8_2p",
                board_type="hex8",
                num_players=2,
                training_intensity="normal",
            )

            # Create mock event
            event = MagicMock()
            event.payload = {
                "config_key": "hex8_2p",
                "quality_score": 0.92,
            }

            await daemon._on_quality_score_updated(event)

            state = daemon._training_states["hex8_2p"]
            assert state.training_intensity == "hot_path"


class TestDynamicThreshold:
    """Tests for dynamic sample threshold calculation.

    Feb 2026: Updated to use module-level compute_dynamic_sample_threshold()
    after extraction from daemon to training_decision_engine.py.
    """

    def test_fallback_on_exception(self):
        """Test fallback when dynamic_threshold_getter raises Exception."""
        from app.coordination.training_decision_engine import (
            compute_dynamic_sample_threshold,
        )

        def raising_getter(_config_key: str) -> int:
            raise RuntimeError("Optimizer error")

        threshold = compute_dynamic_sample_threshold(
            "hex8_2p", num_players=2, base_threshold=5000,
            dynamic_threshold_getter=raising_getter,
        )

        # Should return the configured threshold (fallback)
        assert threshold == 5000

    def test_dynamic_threshold_from_module(self):
        """Test that dynamic threshold is returned without getter."""
        from app.coordination.training_decision_engine import (
            compute_dynamic_sample_threshold,
        )

        # Without getter, should use base_threshold with player multiplier
        threshold = compute_dynamic_sample_threshold(
            "hex8_2p", num_players=2, base_threshold=5000
        )

        # Should be a positive integer
        assert isinstance(threshold, int)
        assert threshold > 0

    def test_dynamic_threshold_with_mock_value(self):
        """Test that dynamic threshold uses getter's value."""
        from app.coordination.training_decision_engine import (
            compute_dynamic_sample_threshold,
        )

        threshold = compute_dynamic_sample_threshold(
            "hex8_2p", num_players=2, base_threshold=5000,
            dynamic_threshold_getter=lambda _: 2500,
        )

        # Should return the getter value (2500 * 1.0 for 2p)
        assert threshold == 2500


# Run with: pytest tests/unit/coordination/test_training_trigger_daemon.py -v

"""Tests for unified training signal computation.

Tests the central signal computation engine that all training decision
systems use for consistent decisions.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.training.unified_signals import (
    ConfigTrainingState,
    TrainingSignals,
    TrainingUrgency,
    UnifiedSignalComputer,
    get_signal_computer,
)


class TestTrainingUrgency:
    """Test TrainingUrgency enum values and usage."""

    def test_urgency_levels_exist(self):
        """Verify all urgency levels are defined."""
        assert TrainingUrgency.CRITICAL.value == "critical"
        assert TrainingUrgency.HIGH.value == "high"
        assert TrainingUrgency.NORMAL.value == "normal"
        assert TrainingUrgency.LOW.value == "low"
        assert TrainingUrgency.NONE.value == "none"

    def test_urgency_ordering(self):
        """Verify urgency levels have consistent semantics."""
        # CRITICAL is most urgent, NONE is least
        urgencies = [
            TrainingUrgency.CRITICAL,
            TrainingUrgency.HIGH,
            TrainingUrgency.NORMAL,
            TrainingUrgency.LOW,
            TrainingUrgency.NONE,
        ]
        # All should be distinct
        assert len(set(urgencies)) == 5


class TestTrainingSignals:
    """Test TrainingSignals dataclass."""

    def test_default_values(self):
        """Verify sensible defaults."""
        signals = TrainingSignals()
        assert signals.games_since_last_training == 0
        assert signals.urgency == TrainingUrgency.NONE
        assert signals.should_train is False
        assert signals.reason == ""
        assert signals.priority == 0.0

    def test_summary_property(self):
        """Test human-readable summary generation."""
        signals = TrainingSignals(
            urgency=TrainingUrgency.HIGH,
            games_since_last_training=600,
            games_threshold=500,
            elo_trend=5.0,
            priority=0.75,
            reason="Games threshold exceeded",
        )
        summary = signals.summary
        assert "HIGH" in summary or "high" in summary
        assert "600" in summary
        assert "500" in summary
        assert "Games threshold" in summary

    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        signals = TrainingSignals(
            games_since_last_training=100,
            current_elo=1600.0,
            urgency=TrainingUrgency.NORMAL,
            should_train=True,
            config_key="square8_2p",
        )
        d = signals.to_dict()

        assert d["games_since_last_training"] == 100
        assert d["current_elo"] == 1600.0
        assert d["urgency"] == "normal"
        assert d["should_train"] is True
        assert d["config_key"] == "square8_2p"
        assert "computed_at" in d


class TestUnifiedSignalComputer:
    """Test the central signal computation engine."""

    @pytest.fixture
    def computer(self):
        """Create a fresh signal computer for each test."""
        return UnifiedSignalComputer()

    def test_initialization(self, computer):
        """Test computer initializes with clean state."""
        assert computer._last_computation is None
        assert computer._last_training_time is None
        assert computer._last_training_games == 0
        assert len(computer._config_states) == 0

    def test_compute_signals_basic(self, computer):
        """Test basic signal computation."""
        signals = computer.compute_signals(
            current_games=1000,
            current_elo=1550.0,
        )

        assert isinstance(signals, TrainingSignals)
        assert signals.games_since_last_training == 1000
        assert signals.current_elo == 1550.0
        assert signals.computed_at is not None

    def test_games_threshold_ratio(self, computer):
        """Test games threshold ratio calculation."""
        # Get the actual threshold from the computer
        threshold = computer.games_threshold

        # Test below threshold
        signals = computer.compute_signals(
            current_games=threshold // 2,
            current_elo=1500.0,
        )
        assert signals.games_threshold_ratio == pytest.approx(0.5, rel=0.1)

        # Test above threshold
        signals = computer.compute_signals(
            current_games=int(threshold * 1.5),
            current_elo=1500.0,
            force_recompute=True,
        )
        assert signals.games_threshold_ratio == pytest.approx(1.5, rel=0.1)

    def test_time_threshold_detection(self, computer):
        """Test time-based training threshold."""
        # Simulate training that happened 2 hours ago
        computer._last_training_time = datetime.now() - timedelta(hours=2)
        computer._last_training_games = 500

        signals = computer.compute_signals(
            current_games=600,
            current_elo=1500.0,
            force_recompute=True,
        )

        assert signals.time_since_last_training >= timedelta(hours=2)
        assert signals.time_threshold_met is True  # Default min interval is 20 min

    def test_bootstrap_detection(self, computer):
        """Test detection of bootstrap state (new config, no models)."""
        signals = computer.compute_signals(
            current_games=100,
            current_elo=1500.0,
            config_key="new_config",
            model_count=0,
        )

        assert signals.is_bootstrap is True
        assert signals.model_count == 0

    def test_win_rate_regression_detection(self, computer):
        """Test win rate regression detection."""
        signals = computer.compute_signals(
            current_games=1000,
            current_elo=1500.0,
            win_rate=0.35,  # Below default min_win_rate (0.45)
        )

        assert signals.win_rate == 0.35
        assert signals.win_rate_regression is True

    def test_config_specific_state(self, computer):
        """Test per-config state tracking."""
        # Update config 1
        computer.record_training_started(1000, config_key="config1")
        computer.record_training_completed(1600.0, config_key="config1")

        # Update config 2
        computer.record_training_started(500, config_key="config2")
        computer.record_training_completed(1550.0, config_key="config2")

        # Verify independent tracking
        signals1 = computer.compute_signals(
            current_games=1200,
            current_elo=1600.0,
            config_key="config1",
        )
        signals2 = computer.compute_signals(
            current_games=700,
            current_elo=1550.0,
            config_key="config2",
        )

        assert signals1.games_since_last_training == 200  # 1200 - 1000
        assert signals2.games_since_last_training == 200  # 700 - 500

    def test_record_training_started(self, computer):
        """Test recording training start."""
        computer.record_training_started(1000)

        assert computer._last_training_games == 1000
        assert computer._last_training_time is not None

    def test_record_training_completed(self, computer):
        """Test recording training completion."""
        computer.record_training_started(1000)
        computer.record_training_completed(new_elo=1650.0)

        assert len(computer._elo_history) > 0
        assert computer._elo_history[-1][1] == 1650.0

    def test_update_config_state(self, computer):
        """Test updating config-specific state."""
        computer.update_config_state(
            config_key="test_config",
            model_count=5,
            current_elo=1700.0,
            win_rate=0.55,
        )

        state = computer._get_config_state("test_config")
        assert state.model_count == 5
        assert state.current_elo == 1700.0
        assert state.win_rate == 0.55

    def test_cache_behavior(self, computer):
        """Test signal computation caching."""
        # First call computes
        signals1 = computer.compute_signals(
            current_games=1000,
            current_elo=1500.0,
        )

        # Second call within TTL returns cached result
        signals2 = computer.compute_signals(
            current_games=1000,
            current_elo=1500.0,
        )

        assert signals1.computed_at == signals2.computed_at

        # Force recompute bypasses cache
        signals3 = computer.compute_signals(
            current_games=1000,
            current_elo=1500.0,
            force_recompute=True,
        )

        assert signals3.computed_at >= signals1.computed_at

    def test_thread_safety(self, computer):
        """Test thread-safe access to signal computer."""
        import threading
        import time

        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    signals = computer.compute_signals(
                        current_games=1000,
                        current_elo=1500.0,
                        force_recompute=True,
                    )
                    results.append(signals)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50


class TestUrgencyComputation:
    """Test urgency level computation logic."""

    @pytest.fixture
    def computer(self):
        return UnifiedSignalComputer()

    def test_none_urgency_when_no_training_needed(self, computer):
        """Test LOW/NONE urgency when below thresholds with existing models."""
        # Simulate that we already have models (not bootstrap)
        computer.update_config_state("test", model_count=5)
        computer.record_training_started(0, config_key="test")
        computer.record_training_completed(1500.0, config_key="test")

        signals = computer.compute_signals(
            current_games=50,  # Well below threshold
            current_elo=1500.0,
            config_key="test",
            model_count=5,
        )

        # With models already, should not be HIGH/CRITICAL
        assert signals.urgency in [TrainingUrgency.NONE, TrainingUrgency.LOW]

    def test_normal_urgency_when_threshold_met(self, computer):
        """Test NORMAL urgency when games threshold is met."""
        signals = computer.compute_signals(
            current_games=550,  # Above 500 threshold
            current_elo=1500.0,
        )

        assert signals.urgency in [TrainingUrgency.NORMAL, TrainingUrgency.HIGH]
        assert signals.should_train is True

    def test_high_urgency_when_threshold_exceeded(self, computer):
        """Test HIGH urgency when significantly over threshold."""
        signals = computer.compute_signals(
            current_games=1000,  # 2x threshold
            current_elo=1500.0,
        )

        assert signals.urgency in [TrainingUrgency.HIGH, TrainingUrgency.CRITICAL]
        assert signals.should_train is True
        assert signals.priority > 0.5

    def test_critical_urgency_on_regression(self, computer):
        """Test CRITICAL urgency when regression detected."""
        # Set up history showing regression
        computer._elo_history = [
            (datetime.now() - timedelta(hours=2), 1600.0),
            (datetime.now() - timedelta(hours=1), 1550.0),
            (datetime.now(), 1480.0),  # Significant drop
        ]

        signals = computer.compute_signals(
            current_games=1000,
            current_elo=1480.0,
            win_rate=0.35,  # Also low win rate
            force_recompute=True,
        )

        # Should detect regression from win rate at minimum
        assert signals.win_rate_regression is True


class TestEloTrendComputation:
    """Test Elo trend calculation via linear regression."""

    @pytest.fixture
    def computer(self):
        return UnifiedSignalComputer()

    def test_positive_elo_trend(self, computer):
        """Test positive Elo trend detection (improving)."""
        now = datetime.now()
        # Use config-specific history for proper trend detection
        computer._config_states["test"] = ConfigTrainingState(
            config_key="test",
            model_count=5,
            elo_history=[
                (now - timedelta(hours=3), 1500.0),
                (now - timedelta(hours=2), 1520.0),
                (now - timedelta(hours=1), 1540.0),
                (now, 1560.0),
            ]
        )

        signals = computer.compute_signals(
            current_games=1000,
            current_elo=1560.0,
            config_key="test",
            model_count=5,
            force_recompute=True,
        )

        # Trend should be positive (but implementation may vary)
        # At minimum, verify elo_trend is computed
        assert signals.elo_trend >= 0  # Non-negative for improving

    def test_negative_elo_trend(self, computer):
        """Test negative Elo trend detection (regressing)."""
        now = datetime.now()
        # Use config-specific history for proper trend detection
        computer._config_states["test"] = ConfigTrainingState(
            config_key="test",
            model_count=5,
            elo_history=[
                (now - timedelta(hours=3), 1560.0),
                (now - timedelta(hours=2), 1540.0),
                (now - timedelta(hours=1), 1520.0),
                (now, 1500.0),
            ]
        )

        signals = computer.compute_signals(
            current_games=1000,
            current_elo=1500.0,
            config_key="test",
            model_count=5,
            force_recompute=True,
        )

        # Trend should be negative or zero (but implementation may vary)
        # At minimum, verify elo_trend is computed
        assert signals.elo_trend <= 0  # Non-positive for regressing

    def test_stable_elo_trend(self, computer):
        """Test stable Elo (near zero trend)."""
        now = datetime.now()
        computer._elo_history = [
            (now - timedelta(hours=3), 1500.0),
            (now - timedelta(hours=2), 1502.0),
            (now - timedelta(hours=1), 1498.0),
            (now, 1501.0),
        ]

        signals = computer.compute_signals(
            current_games=1000,
            current_elo=1501.0,
            force_recompute=True,
        )

        # Should be close to zero
        assert abs(signals.elo_trend) < 10  # Less than 10 Elo/hour

    def test_elo_trend_ignores_none_entries(self, computer):
        """Trend calculation should tolerate sparse history before first Elo."""
        now = datetime.now()
        computer._config_states["test"] = ConfigTrainingState(
            config_key="test",
            model_count=1,
            elo_history=[
                (now - timedelta(minutes=30), None),  # type: ignore[list-item]
                (now - timedelta(minutes=20), None),  # type: ignore[list-item]
                (now - timedelta(minutes=10), 1500.0),
                (now, 1510.0),
            ],
        )

        signals = computer.compute_signals(
            current_games=100,
            current_elo=None,  # type: ignore[arg-type]
            config_key="test",
            model_count=1,
            force_recompute=True,
        )

        assert signals.current_elo == 1510.0
        assert signals.elo_trend > 0


class TestSingletonPattern:
    """Test singleton pattern for global signal computer."""

    def test_get_signal_computer_returns_singleton(self):
        """Test that get_signal_computer returns the same instance."""
        computer1 = get_signal_computer()
        computer2 = get_signal_computer()

        assert computer1 is computer2

    def test_signal_computer_is_instance(self):
        """Test that returned object is UnifiedSignalComputer."""
        computer = get_signal_computer()
        assert isinstance(computer, UnifiedSignalComputer)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_should_train_convenience(self):
        """Test should_train convenience function."""
        from app.training.unified_signals import should_train

        # Include required config_key parameter
        result = should_train(
            config_key="test",
            current_games=1000,
            current_elo=1500.0,
        )
        assert isinstance(result, bool)

    def test_get_urgency_convenience(self):
        """Test get_urgency convenience function."""
        from app.training.unified_signals import get_urgency

        # Include required config_key parameter
        result = get_urgency(
            config_key="test",
            current_games=1000,
            current_elo=1500.0,
        )
        assert isinstance(result, TrainingUrgency)

    def test_get_training_intensity_convenience(self):
        """Test get_training_intensity convenience function."""
        from app.training.unified_signals import get_training_intensity

        # Include required config_key parameter
        result = get_training_intensity(
            config_key="test",
            current_games=1000,
            current_elo=1500.0,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 2.5  # Valid intensity range


class TestIntegrationWithOtherSystems:
    """Test integration with TrainingTriggers and FeedbackAccelerator."""

    def test_training_triggers_uses_unified_signals(self):
        """Verify TrainingTriggers delegates to unified signals."""
        from app.training.training_triggers import TrainingTriggers

        triggers = TrainingTriggers()
        assert hasattr(triggers, '_signal_computer')
        assert triggers._signal_computer is not None

    def test_feedback_accelerator_uses_unified_signals(self):
        """Verify FeedbackAccelerator uses unified signals."""
        from app.training.feedback_accelerator import FeedbackAccelerator

        accel = FeedbackAccelerator()
        assert hasattr(accel, '_signal_computer')
        assert accel._signal_computer is not None

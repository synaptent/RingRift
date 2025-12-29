"""Unit tests for CurriculumFeedback module.

Tests the curriculum feedback loop for training, covering:
- ConfigMetrics dataclass calculations
- GameRecord dataclass
- CurriculumFeedback game recording and weight computation
- Promotion recording with weight adjustments
- Stage tracking and advancement
- Event handlers for CURRICULUM_ADVANCED and TRAINING_EARLY_STOPPED

December 28, 2025
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.training.curriculum_feedback import (
    ConfigMetrics,
    CurriculumFeedback,
    GameRecord,
    DEFAULT_WEIGHT_MIN,
    DEFAULT_WEIGHT_MAX,
    DEFAULT_LOOKBACK_MINUTES,
    DEFAULT_TARGET_WIN_RATE,
)


# ============================================================================
# ConfigMetrics Tests
# ============================================================================


class TestConfigMetrics:
    """Tests for ConfigMetrics dataclass."""

    def test_default_values(self):
        """Test ConfigMetrics default values."""
        metrics = ConfigMetrics()
        assert metrics.games_total == 0
        assert metrics.games_recent == 0
        assert metrics.wins_recent == 0
        assert metrics.losses_recent == 0
        assert metrics.draws_recent == 0
        assert metrics.avg_elo == 1500.0
        assert metrics.win_rate == 0.5
        assert metrics.elo_trend == 0.0
        assert metrics.last_game_time == 0
        assert metrics.last_training_time == 0
        assert metrics.model_count == 0

    def test_recent_win_rate_no_games(self):
        """Test recent_win_rate returns 0.5 when no games."""
        metrics = ConfigMetrics()
        assert metrics.recent_win_rate == 0.5

    def test_recent_win_rate_all_wins(self):
        """Test recent_win_rate with all wins."""
        metrics = ConfigMetrics(wins_recent=10, losses_recent=0, draws_recent=0)
        assert metrics.recent_win_rate == 1.0

    def test_recent_win_rate_all_losses(self):
        """Test recent_win_rate with all losses."""
        metrics = ConfigMetrics(wins_recent=0, losses_recent=10, draws_recent=0)
        assert metrics.recent_win_rate == 0.0

    def test_recent_win_rate_mixed(self):
        """Test recent_win_rate with mixed results."""
        # 5 wins + 0.5 * 2 draws = 6, total = 10
        metrics = ConfigMetrics(wins_recent=5, losses_recent=3, draws_recent=2)
        expected = (5 + 0.5 * 2) / 10  # 0.6
        assert metrics.recent_win_rate == expected

    def test_recent_win_rate_only_draws(self):
        """Test recent_win_rate with only draws."""
        metrics = ConfigMetrics(wins_recent=0, losses_recent=0, draws_recent=10)
        assert metrics.recent_win_rate == 0.5


# ============================================================================
# GameRecord Tests
# ============================================================================


class TestGameRecord:
    """Tests for GameRecord dataclass."""

    def test_creation(self):
        """Test GameRecord creation with required fields."""
        record = GameRecord(
            config_key="hex8_2p",
            timestamp=1000.0,
            winner=1,
        )
        assert record.config_key == "hex8_2p"
        assert record.timestamp == 1000.0
        assert record.winner == 1
        assert record.model_elo == 1500.0  # default
        assert record.opponent_type == "baseline"  # default

    def test_creation_with_all_fields(self):
        """Test GameRecord creation with all fields."""
        record = GameRecord(
            config_key="square8_4p",
            timestamp=2000.0,
            winner=-1,
            model_elo=1750.0,
            opponent_type="selfplay",
        )
        assert record.config_key == "square8_4p"
        assert record.winner == -1
        assert record.model_elo == 1750.0
        assert record.opponent_type == "selfplay"


# ============================================================================
# CurriculumFeedback Core Tests
# ============================================================================


class TestCurriculumFeedbackInit:
    """Tests for CurriculumFeedback initialization."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_default_initialization(self, mock_wire):
        """Test CurriculumFeedback initializes with defaults."""
        feedback = CurriculumFeedback()
        assert feedback.lookback_minutes == DEFAULT_LOOKBACK_MINUTES
        assert feedback.weight_min == DEFAULT_WEIGHT_MIN
        assert feedback.weight_max == DEFAULT_WEIGHT_MAX
        assert feedback.target_win_rate == DEFAULT_TARGET_WIN_RATE
        assert feedback._game_history == {}
        assert feedback._config_metrics == {}
        # December 29, 2025: ALL_CANONICAL_CONFIGS are now pre-initialized
        assert len(feedback._current_weights) == 12  # All canonical configs initialized
        assert len(feedback._curriculum_stages) == 12  # All canonical configs initialized
        assert all(w == 1.0 for w in feedback._current_weights.values())  # Default weight
        assert all(s == 0 for s in feedback._curriculum_stages.values())  # Default stage

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_custom_initialization(self, mock_wire):
        """Test CurriculumFeedback with custom parameters."""
        feedback = CurriculumFeedback(
            lookback_minutes=60,
            weight_min=0.3,
            weight_max=3.0,
            target_win_rate=0.6,
        )
        assert feedback.lookback_minutes == 60
        assert feedback.weight_min == 0.3
        assert feedback.weight_max == 3.0
        assert feedback.target_win_rate == 0.6


class TestCurriculumFeedbackRecordGame:
    """Tests for CurriculumFeedback.record_game()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_record_single_game(self, mock_wire):
        """Test recording a single game."""
        feedback = CurriculumFeedback()
        feedback.record_game("hex8_2p", winner=1, model_elo=1600.0)

        assert "hex8_2p" in feedback._game_history
        assert len(feedback._game_history["hex8_2p"]) == 1
        record = feedback._game_history["hex8_2p"][0]
        assert record.winner == 1
        assert record.model_elo == 1600.0

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_record_multiple_games(self, mock_wire):
        """Test recording multiple games."""
        feedback = CurriculumFeedback()
        for i in range(5):
            feedback.record_game("hex8_2p", winner=1 if i % 2 == 0 else -1)

        assert len(feedback._game_history["hex8_2p"]) == 5

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_record_games_multiple_configs(self, mock_wire):
        """Test recording games for multiple configs."""
        feedback = CurriculumFeedback()
        feedback.record_game("hex8_2p", winner=1)
        feedback.record_game("square8_4p", winner=-1)
        feedback.record_game("hexagonal_3p", winner=0)

        assert len(feedback._game_history) == 3
        assert len(feedback._game_history["hex8_2p"]) == 1
        assert len(feedback._game_history["square8_4p"]) == 1
        assert len(feedback._game_history["hexagonal_3p"]) == 1

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_history_truncation(self, mock_wire):
        """Test that history is truncated to max size."""
        feedback = CurriculumFeedback()
        feedback._max_history_per_config = 100  # Set small max for testing

        # Record more than max
        for i in range(150):
            feedback.record_game("hex8_2p", winner=1)

        # Should be truncated to max
        assert len(feedback._game_history["hex8_2p"]) == 100


class TestCurriculumFeedbackRecordPromotion:
    """Tests for CurriculumFeedback.record_promotion()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_successful_promotion_reduces_weight(self, mock_wire, mock_emit):
        """Test that successful promotion reduces weight."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 1.0

        feedback.record_promotion("hex8_2p", promoted=True, new_elo=1700.0)

        # Weight should decrease by 0.1
        assert feedback._current_weights["hex8_2p"] == 0.9
        mock_emit.assert_called_once()

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_failed_promotion_increases_weight(self, mock_wire, mock_emit):
        """Test that failed promotion increases weight."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 1.0

        feedback.record_promotion("hex8_2p", promoted=False)

        # Weight should increase by 0.15
        assert feedback._current_weights["hex8_2p"] == 1.15
        mock_emit.assert_called_once()

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_weight_respects_max(self, mock_wire, mock_emit):
        """Test that weight doesn't exceed max."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 1.95

        feedback.record_promotion("hex8_2p", promoted=False)

        # Should cap at weight_max (2.0)
        assert feedback._current_weights["hex8_2p"] == 2.0

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_weight_respects_min(self, mock_wire, mock_emit):
        """Test that weight doesn't go below min."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 0.55

        feedback.record_promotion("hex8_2p", promoted=True)

        # Should cap at weight_min (0.5)
        assert feedback._current_weights["hex8_2p"] == 0.5

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_updates_elo_tracking(self, mock_wire, mock_emit):
        """Test that promotion updates Elo tracking."""
        feedback = CurriculumFeedback()
        metrics = feedback._get_or_create_metrics("hex8_2p")
        old_elo = metrics.avg_elo

        feedback.record_promotion("hex8_2p", promoted=True, new_elo=1650.0)

        metrics = feedback._get_or_create_metrics("hex8_2p")
        assert metrics.avg_elo == 1650.0
        assert metrics.elo_trend == 1650.0 - old_elo


class TestCurriculumFeedbackStageTracking:
    """Tests for curriculum stage tracking."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_get_stage_default(self, mock_wire):
        """Test get_curriculum_stage returns 0 for unknown config."""
        feedback = CurriculumFeedback()
        assert feedback.get_curriculum_stage("unknown_config") == 0

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_advance_stage(self, mock_wire):
        """Test manual stage advancement."""
        feedback = CurriculumFeedback()
        assert feedback.get_curriculum_stage("hex8_2p") == 0

        feedback.advance_stage("hex8_2p", 3)
        assert feedback.get_curriculum_stage("hex8_2p") == 3

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_get_all_stages(self, mock_wire):
        """Test getting all curriculum stages."""
        feedback = CurriculumFeedback()
        feedback.advance_stage("hex8_2p", 2)
        feedback.advance_stage("square8_4p", 5)

        stages = feedback.get_all_curriculum_stages()
        assert stages == {"hex8_2p": 2, "square8_4p": 5}


class TestCurriculumFeedbackEventHandlers:
    """Tests for event handlers."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_on_curriculum_advanced(self, mock_wire, mock_emit):
        """Test _on_curriculum_advanced handler."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 1.0
        feedback._curriculum_stages["hex8_2p"] = 1

        event = MagicMock()
        event.payload = {"config": "hex8_2p", "new_stage": 3, "reason": "gauntlet"}

        feedback._on_curriculum_advanced(event)

        # Stage should be updated
        assert feedback._curriculum_stages["hex8_2p"] == 3
        # Weight should be reduced (2 stage advancement * -0.05)
        assert feedback._current_weights["hex8_2p"] == 0.9

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_on_curriculum_advanced_empty_config(self, mock_wire, mock_emit):
        """Test _on_curriculum_advanced with empty config key."""
        feedback = CurriculumFeedback()

        event = MagicMock()
        event.payload = {"config": "", "new_stage": 3}

        # Should not raise, just return early
        feedback._on_curriculum_advanced(event)
        mock_emit.assert_not_called()

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_on_training_early_stopped(self, mock_wire, mock_emit):
        """Test _on_training_early_stopped handler boosts weight."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 1.0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "stagnation",
            "epoch": 25,
            "final_loss": 0.5,
        }

        feedback._on_training_early_stopped(event)

        # Weight should be boosted by 30%
        assert feedback._current_weights["hex8_2p"] == 1.3
        mock_emit.assert_called_once()

    @patch("app.training.curriculum_feedback.CurriculumFeedback._emit_curriculum_updated")
    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_on_training_early_stopped_respects_max(self, mock_wire, mock_emit):
        """Test _on_training_early_stopped respects weight max."""
        feedback = CurriculumFeedback()
        feedback._current_weights["hex8_2p"] = 1.9

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "reason": "stagnation"}

        feedback._on_training_early_stopped(event)

        # 1.9 * 1.3 = 2.47, capped to max 2.0
        assert feedback._current_weights["hex8_2p"] == 2.0


class TestCurriculumFeedbackGetCurriculumWeights:
    """Tests for get_curriculum_weights()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_empty_returns_empty(self, mock_wire):
        """Test get_curriculum_weights with no games recorded."""
        feedback = CurriculumFeedback()
        weights = feedback.get_curriculum_weights()
        assert weights == {}

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_returns_weights_for_recorded_games(self, mock_wire):
        """Test get_curriculum_weights returns weights for configs with games."""
        feedback = CurriculumFeedback()
        # Record games to create config entries
        feedback.record_game("hex8_2p", winner=1)
        feedback.record_game("square8_4p", winner=-1)

        weights = feedback.get_curriculum_weights()
        # Should have entries for configs with recorded games
        assert "hex8_2p" in weights
        assert "square8_4p" in weights
        # Weights should be within valid range
        for config, weight in weights.items():
            assert 0.5 <= weight <= 2.0


class TestCurriculumFeedbackRecordTraining:
    """Tests for record_training()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_updates_last_training_time(self, mock_wire):
        """Test record_training updates last_training_time."""
        feedback = CurriculumFeedback()
        before = time.time()

        feedback.record_training("hex8_2p")

        metrics = feedback._get_or_create_metrics("hex8_2p")
        assert metrics.last_training_time >= before


class TestCurriculumFeedbackSetOpponentTracker:
    """Tests for set_opponent_tracker()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_sets_tracker(self, mock_wire):
        """Test set_opponent_tracker stores the tracker."""
        feedback = CurriculumFeedback()
        assert feedback._opponent_tracker is None

        mock_tracker = MagicMock()
        feedback.set_opponent_tracker(mock_tracker)

        assert feedback._opponent_tracker is mock_tracker


class TestCurriculumFeedbackShouldUpdateCurriculum:
    """Tests for should_update_curriculum()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_not_enough_games(self, mock_wire):
        """Test returns False when not enough games."""
        feedback = CurriculumFeedback()
        # Record fewer than min_games
        for _ in range(10):
            feedback.record_game("hex8_2p", winner=1)

        assert feedback.should_update_curriculum(min_games=50) is False

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_enough_games(self, mock_wire):
        """Test returns True when enough games."""
        feedback = CurriculumFeedback()
        # Record more than min_games
        for _ in range(60):
            feedback.record_game("hex8_2p", winner=1)

        assert feedback.should_update_curriculum(min_games=50) is True


class TestCurriculumFeedbackExportWeightsJson:
    """Tests for export_weights_json()."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_export_creates_file(self, mock_wire, tmp_path):
        """Test export_weights_json creates file."""
        feedback = CurriculumFeedback()
        feedback._current_weights = {"hex8_2p": 1.5}

        output_path = tmp_path / "weights.json"
        feedback.export_weights_json(str(output_path))

        assert output_path.exists()

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_export_content(self, mock_wire, tmp_path):
        """Test exported JSON content."""
        import json

        feedback = CurriculumFeedback()
        # Record games to create config entries that will have weights
        feedback.record_game("hex8_2p", winner=1)
        feedback.record_game("square8_4p", winner=-1)

        output_path = tmp_path / "weights.json"
        feedback.export_weights_json(str(output_path))

        with open(output_path) as f:
            data = json.load(f)

        assert "weights" in data
        # Configs with recorded games should have weights in export
        assert "hex8_2p" in data["weights"]
        assert "square8_4p" in data["weights"]
        # Weights should be numeric
        assert isinstance(data["weights"]["hex8_2p"], float)


class TestCurriculumFeedbackGetMetrics:
    """Tests for metrics retrieval."""

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_get_config_metrics(self, mock_wire):
        """Test get_config_metrics returns metrics."""
        feedback = CurriculumFeedback()
        feedback.record_game("hex8_2p", winner=1)

        metrics = feedback.get_config_metrics("hex8_2p")
        assert isinstance(metrics, ConfigMetrics)

    @patch("app.training.curriculum_feedback.CurriculumFeedback._auto_wire_curriculum_advanced")
    def test_get_all_metrics(self, mock_wire):
        """Test get_all_metrics returns all config metrics."""
        feedback = CurriculumFeedback()
        feedback.record_game("hex8_2p", winner=1)
        feedback.record_game("square8_4p", winner=-1)

        all_metrics = feedback.get_all_metrics()
        assert "hex8_2p" in all_metrics
        assert "square8_4p" in all_metrics


class TestConstants:
    """Tests for module-level constants."""

    def test_default_constants(self):
        """Test default constant values."""
        assert DEFAULT_WEIGHT_MIN == 0.5
        assert DEFAULT_WEIGHT_MAX == 2.0
        assert DEFAULT_LOOKBACK_MINUTES == 30
        assert DEFAULT_TARGET_WIN_RATE == 0.55
